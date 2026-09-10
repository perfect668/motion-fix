"""Geometry-only source contact detection for WholeBody V5.

The detector never receives robot qpos.  It creates stable per-frame contact
records from canonical human landmarks and the source/target terrain field;
the QP consumes the resulting plan without changing its topology mid-frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.schemas import ContactEpisode, ContactPlan
from ..motion_adapters import CanonicalMotion
from ..terrain_geometry import SceneTransform, TerrainField


CHANNELS = (
    "left_heel", "right_heel", "left_toe", "right_toe",
    "left_palm", "right_palm", "left_knee", "right_knee",
    "left_shin", "right_shin", "left_butt", "right_butt",
    "lower_back", "upper_back",
)


def _points(motion: CanonicalMotion) -> list[dict[str, np.ndarray]]:
    frames = motion.canonical_named_positions()
    result = []
    for frame in frames:
        value = {str(k): np.asarray(v, dtype=float) for k, v in frame.items()}
        # Build an anatomical frame from the measured pelvis/hip/spine
        # landmarks.  All butt/back proxies are expressed in this frame; a
        # fixed world-Z offset is invalid once the person bends or rotates.
        pelvis = value.get("pelvis")
        spine = value.get("spine3", pelvis)
        lhip, rhip = value.get("left_hip", pelvis), value.get("right_hip", pelvis)
        up = np.asarray(spine - pelvis, dtype=float)
        up /= max(float(np.linalg.norm(up)), 1e-12)
        lateral = np.asarray(lhip - rhip, dtype=float)
        lateral /= max(float(np.linalg.norm(lateral)), 1e-12)
        backward = -np.cross(lateral, up)
        backward /= max(float(np.linalg.norm(backward)), 1e-12)
        # Candidate provenance is kept separate from canonical landmark names
        # so adapters remain format agnostic and diagnostics can explain which
        # geometric proxy generated a contact.
        candidates: dict[str, tuple[np.ndarray, list[str]]] = {}
        for side in ("left", "right"):
            hip = value.get(f"{side}_hip", value.get("pelvis"))
            knee = value.get(f"{side}_knee", hip)
            ankle = value.get(f"{side}_ankle", value.get(f"{side}_foot", knee))
            value.setdefault(f"{side}_shin", .5 * (knee + ankle))
            # The pelvis joint is not a butt surface.  This proxy is built
            # once here (rather than independently in a task and a solver)
            # and its provenance is carried into the contact record.
            # Three-dimensional candidates cover the actual pelvis surface
            # under arbitrary seated/leaning orientations.  The solver sees
            # one selected point per channel, while the detector chooses the
            # closest surface-consistent candidate.
            side_sign = 1.0 if side == "left" else -1.0
            butt_base = np.asarray(hip, dtype=float)
            butt_offsets = (
                -0.10 * up + side_sign * 0.025 * lateral,
                -0.14 * up + side_sign * 0.045 * lateral,
                -0.17 * up + side_sign * 0.055 * lateral,
                -0.14 * up + side_sign * 0.035 * lateral + 0.035 * backward,
                -0.14 * up + side_sign * 0.035 * lateral - 0.025 * backward,
            )
            candidates[f"{side}_butt"] = (
                np.asarray([butt_base + offset for offset in butt_offsets]),
                ["hip_derived_butt_proxy"] * len(butt_offsets),
            )
            value.setdefault(f"{side}_butt", candidates[f"{side}_butt"][0][0].copy())
            value.setdefault(f"{side}_palm", value.get(f"{side}_wrist", hip))
            # Heel/toe are independent surface landmarks. Never substitute
            # an ankle/foot center: that creates false support episodes for
            # formats that do not provide true surface markers.
            if f"{side}_heel" not in value:
                value.pop(f"{side}_heel", None)
            if f"{side}_toe" not in value:
                value.pop(f"{side}_toe", None)
        # Back surface candidates similarly follow the torso frame instead of
        # assuming the source world axes are anatomical.
        lower_offsets = tuple(
            height * up + depth * backward
            for height, depth in (
                (0.02, -0.06), (0.02, 0.06),
                (0.05, -0.09), (0.05, 0.09),
                (0.08, 0.04),
            )
        )
        upper_offsets = tuple(
            height * up + depth * backward
            for height, depth in (
                (0.00, -0.08), (0.00, 0.08),
                (0.04, -0.10), (0.04, 0.10),
                (0.08, 0.06),
            )
        )
        candidates["lower_back"] = (
            np.asarray([pelvis + offset for offset in lower_offsets]),
            ["spine_surface_proxy"] * len(lower_offsets),
        )
        candidates["upper_back"] = (
            np.asarray([spine + offset for offset in upper_offsets]),
            ["spine_surface_proxy"] * len(upper_offsets),
        )
        value.setdefault("lower_back", candidates["lower_back"][0][0].copy())
        value.setdefault("upper_back", candidates["upper_back"][0][0].copy())
        value["__proxy_candidates__"] = candidates
        result.append(value)
    return result


class SourceContactDetector:
    def __init__(self, terrain: TerrainField, fps: float, config: dict[str, Any] | None = None):
        self.terrain = terrain; self.fps=float(fps); self.config=dict(config or {})
        self.enter=float(self.config.get("contact_enter_distance", .03)); self.exit=float(self.config.get("contact_exit_distance", .055))
        self.static_speed=float(self.config.get("static_tangent_speed", .08)); self.normal_speed=float(self.config.get("normal_speed_limit", .20)); self.min_score=float(self.config.get("contact_activation_score", .15))
        self.max_penetration=float(self.config.get("max_allowed_penetration", .02))
        self.surface_normal_cos=float(self.config.get("surface_normal_cos", .95))

    def detect(self, frames: list[dict[str, np.ndarray]], transform: SceneTransform | None = None) -> ContactPlan:
        transformed=[]
        for frame in frames:
            transformed_frame = {}
            for key, value in frame.items():
                if key == "__proxy_candidates__":
                    transformed_candidates = {}
                    for channel, payload in value.items():
                        candidate_points, provenances = payload
                        candidate_points = np.asarray(candidate_points, dtype=float).reshape((-1, 3))
                        if transform is not None:
                            candidate_points = transform.transform_points(candidate_points)
                        transformed_candidates[channel] = (candidate_points, list(provenances))
                    transformed_frame[key] = transformed_candidates
                else:
                    array = np.asarray(value, dtype=float).reshape(3)
                    transformed_frame[key] = transform.transform_points(array) if transform is not None else array.copy()
            transformed.append(transformed_frame)
        records = []
        previous = {name: False for name in CHANNELS}
        previous_states = {name: "NONE" for name in CHANNELS}
        blend_frames = max(1, int(self.config.get("contact_blend_frames", 7)))
        blend_alpha = 1.0 / float(blend_frames)
        blended_scores = {name: 0.0 for name in CHANNELS}
        last_points: dict[str, np.ndarray] = {}
        # A STATIC contact owns a world-space surface anchor for its complete
        # episode.  Nearest-surface queries are still useful for deciding when
        # an episode ends, but they must never make a supporting foot/palm
        # drift along a tessellated mesh while the source is stationary.
        static_anchors: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
        episode_surfaces: dict[str, str] = {}
        episodes: list[ContactEpisode] = []
        for index, frame in enumerate(transformed):
            if hasattr(self.terrain, "set_time"):
                self.terrain.set_time(index / self.fps)
            current={}
            proxy_candidates = frame.get("__proxy_candidates__", {})
            for name in CHANNELS:
                if name not in frame and "pelvis" not in frame:
                    raise ValueError(f"Canonical source frame lacks {name!r} and pelvis fallback")
                # Missing anatomical landmarks are a capability limitation,
                # not evidence of contact.  In particular, a position-only
                # source without toe markers must never turn the pelvis (or
                # ankle) into a fabricated toe contact.
                if name not in frame and name not in proxy_candidates:
                    current[name] = {
                        "score": 0.0, "activation": 0.0,
                        "state": "NONE", "source_state": "NONE",
                        "surface_id": "unavailable",
                        "object_id": "unavailable",
                        "signed_distance": float("inf"),
                        "contact_distance": float("inf"),
                        "normal_speed": 0.0, "tangential_speed": 0.0,
                        "provenance": "missing_landmark",
                    }
                    previous[name] = False
                    previous_states[name] = "NONE"
                    continue
                raw_candidates = proxy_candidates.get(name)
                if raw_candidates is None:
                    points_for_channel = [np.asarray(frame.get(name, frame.get("pelvis")), dtype=float)]
                    provenance_for_channel = [self._provenance(name)]
                else:
                    points_for_channel = [np.asarray(item, dtype=float) for item in np.asarray(raw_candidates[0])]
                    provenance_for_channel = list(raw_candidates[1])
                # Evaluate all candidates before selecting one.  A small
                # continuity penalty prevents a proxy from jumping between
                # opposite pelvis/back samples when a mesh edge is crossed.
                previous_point = last_points.get(name)
                evaluated = []
                for candidate_index, candidate_point in enumerate(points_for_channel):
                    if "heel" in name or "toe" in name:
                        candidate_hit = self.terrain.support_surface(candidate_point)
                    else:
                        query = getattr(self.terrain, "nearest_contact_surface", self.terrain.nearest_surface)
                        candidate_hit = query(candidate_point)
                    candidate_distance = (
                        float(np.linalg.norm(candidate_point - candidate_hit.closest_point))
                        if candidate_hit.surface_type == "mesh" else float(candidate_hit.signed_distance)
                    )
                    continuity = 0.015 * float(np.linalg.norm(candidate_point - previous_point)) if previous_point is not None else 0.0
                    evaluated.append((candidate_distance + continuity, candidate_index, candidate_point, candidate_hit))
                _, selected_index, point, hit = min(evaluated, key=lambda item: (item[0], item[1]))
                provenance = provenance_for_channel[min(selected_index, len(provenance_for_channel) - 1)]
                old = last_points.get(name, point)
                velocity = (point - old) * self.fps
                last_points[name] = point.copy()
                normal_speed = float(velocity @ hit.normal)
                normal_speed_abs = abs(normal_speed)
                tangent_speed = float(np.linalg.norm(velocity - hit.normal * normal_speed))
                signed_distance = float(hit.signed_distance)
                # The mesh field supplies a stable nearest vertex/sample for
                # contact references, not a watertight signed-distance field.
                # Its normal projection may be deeply negative beside a chair
                # even when that body part is far away.  Use Euclidean distance
                # for mesh-contact timing and preserve the signed value only
                # for diagnostics; MuJoCo collision remains authoritative.
                distance = (
                    float(np.linalg.norm(point - hit.closest_point))
                    if hit.surface_type == "mesh"
                    else signed_distance
                )
                # Deep penetration is not evidence of desired contact.  For
                # analytic solids retain the signed-distance lower bound; mesh
                # queries expose Euclidean contact distance and leave collision
                # penetration to MuJoCo.
                deep_penetration = hit.surface_type != "mesh" and signed_distance < -self.max_penetration
                support_ok = (not ("heel" in name or "toe" in name)) or bool(hit.supportable)
                in_contact = support_ok and distance <= self.enter and normal_speed_abs <= self.normal_speed and not deep_penetration
                if previous[name] and distance <= self.exit and normal_speed_abs <= self.normal_speed:
                    in_contact = not deep_penetration
                if distance > self.exit or normal_speed_abs > 2 * self.normal_speed:
                    in_contact = False
                score = (
                    float(np.clip(1.0 - max(distance, 0.0) / max(self.exit, 1e-6), 0.0, 1.0))
                    * float(np.clip(1.0 - normal_speed_abs / max(self.normal_speed, 1e-6), 0.0, 1.0))
                    if in_contact
                    else 0.0
                )
                if score < self.min_score:
                    in_contact = False
                    score = 0.0
                # Keep contact state semantic and independent from task
                # activation.  A newly detected heel/toe or butt contact is
                # blended over several solver frames; release uses the same
                # ramp, preventing a single frame from pulling the leg into a
                # different IK branch.
                blended_scores[name] += blend_alpha * (float(score) - blended_scores[name])
                activation = float(np.clip(blended_scores[name], 0.0, 1.0))
                state="NONE" if not in_contact else ("STATIC" if tangent_speed <= self.static_speed else "SLIDING")
                previous_surface = episode_surfaces.get(name)
                previous_normal = current.get(name, {}).get("surface_normal_solver")
                if previous_normal is None:
                    previous_normal = frame.get(f"{name}_normal")
                same_asset_surface = False
                if previous[name] and previous_surface is not None:
                    previous_object = str(previous_surface).split(":", 1)[0]
                    current_object = str(hit.surface_id).split(":", 1)[0]
                    # Curved/triangulated surfaces commonly expose a new
                    # patch every frame.  Keep one contact episode when the
                    # asset and outward normal remain continuous; an actual
                    # asset transfer or sharp edge still starts a new one.
                    if previous_object == current_object:
                        prior_normal = last_points.get(f"{name}::__normal")
                        if prior_normal is not None:
                            same_asset_surface = float(np.asarray(prior_normal) @ hit.normal) >= self.surface_normal_cos
                surface_changed = bool(previous[name] and previous_surface is not None and str(previous_surface) != str(hit.surface_id) and not same_asset_surface)
                if surface_changed:
                    # A contact episode is surface-owned.  Do not keep an
                    # anchor from the previous chair face/platform when the
                    # source crosses an edge or transfers to another support.
                    static_anchors.pop(name, None)
                    episode_surfaces.pop(name, None)
                if state != "NONE":
                    episode_surfaces.setdefault(name, str(hit.surface_id))
                    if previous[name] and not surface_changed and name in episode_surfaces:
                        # Preserve the source episode's surface identity at
                        # tessellation/box edges; a new surface is accepted
                        # only after the contact has actually exited.
                        locked_surface = episode_surfaces[name]
                    else:
                        locked_surface = str(hit.surface_id)
                else:
                    locked_surface = str(hit.surface_id)
                    episode_surfaces.pop(name, None)
                if state == "STATIC":
                    anchor = static_anchors.get(name)
                    if anchor is None or (not previous[name]):
                        anchor = (hit.closest_point.copy(), hit.normal.copy(), str(hit.surface_id))
                        static_anchors[name] = anchor
                    anchor_point, anchor_normal, anchor_surface = anchor
                else:
                    static_anchors.pop(name, None)
                    anchor_point, anchor_normal, anchor_surface = (
                        hit.closest_point.copy(), hit.normal.copy(), str(hit.surface_id)
                    )

                current[name] = {
                    "score": score,
                    "activation": activation,
                    "state": state,
                    "source_state": state,
                    "human_point_source": point.copy(),
                    "human_point_solver": point.copy(),
                    "surface_point_source": hit.closest_point.copy(),
                    "surface_point_solver": hit.closest_point.copy(),
                    "surface_normal_source": hit.normal.copy(),
                    "surface_normal_solver": hit.normal.copy(),
                    "tangent_anchor_source": anchor_point.copy(),
                    "tangent_anchor_solver": anchor_point.copy(),
                    "anchor_normal_source": anchor_normal.copy(),
                    "anchor_normal_solver": anchor_normal.copy(),
                    "anchor_surface_id": anchor_surface,
                    "surface_id": locked_surface,
                    "object_id": locked_surface.split(":", 1)[0],
                    "signed_distance": signed_distance,
                    "contact_distance": distance,
                    "normal_speed": normal_speed,
                    "tangential_speed": tangent_speed,
                    "provenance": provenance,
                    "triangle_id": getattr(hit, "triangle_id", None),
                    "barycentric": getattr(hit, "barycentric", None),
                    "asset_local_anchor": getattr(hit, "asset_local_anchor", None),
                    "asset_local_normal": getattr(hit, "asset_local_normal", None),
                }
                # Keep the last normal in the detector state without exposing
                # it as a synthetic motion landmark.
                last_points[f"{name}::__normal"] = hit.normal.copy()
                changed = previous_states[name] != "NONE" and (
                    state == "NONE" or state != previous_states[name] or surface_changed
                )
                if changed:
                    for ep_i in range(len(episodes) - 1, -1, -1):
                        ep = episodes[ep_i]
                        if ep.body_channel == name and ep.end_frame == index - 1:
                            episodes[ep_i] = ContactEpisode(
                                ep.body_channel, ep.phase, ep.start_frame, index - 1,
                                ep.object_id, ep.surface_id, ep.source_anchor, ep.normal,
                                triangle_id=ep.triangle_id, barycentric=ep.barycentric,
                                asset_local_anchor=ep.asset_local_anchor,
                                confidence=ep.confidence,
                                source_provenance=ep.source_provenance,
                                target_provenance=ep.target_provenance,
                            )
                            break
                if state != "NONE" and (previous_states[name] == "NONE" or changed):
                    episodes.append(
                        ContactEpisode(
                            name,
                            state,
                            index,
                            index,
                            str(current[name]["object_id"]),
                            str(anchor_surface),
                            anchor_point.copy(),
                            anchor_normal.copy(),
                            triangle_id=getattr(hit, "triangle_id", None),
                            barycentric=getattr(hit, "barycentric", None),
                            asset_local_anchor=getattr(hit, "asset_local_anchor", anchor_point.copy()),
                            confidence=float(score),
                            source_provenance=provenance,
                            target_provenance="surface_anchor",
                        )
                    )
                elif state != "NONE" and previous_states[name] != "NONE" and not changed:
                    for ep_i in range(len(episodes)-1,-1,-1):
                        ep=episodes[ep_i]
                        if ep.body_channel==name and ep.end_frame==index-1:
                            episodes[ep_i]=ContactEpisode(
                                ep.body_channel, state, ep.start_frame, index,
                                ep.object_id, ep.surface_id, ep.source_anchor, ep.normal,
                                triangle_id=ep.triangle_id,
                                barycentric=ep.barycentric,
                                asset_local_anchor=ep.asset_local_anchor,
                                confidence=max(ep.confidence, float(score)),
                                source_provenance=ep.source_provenance,
                                target_provenance=ep.target_provenance,
                            ); break
                previous[name]=state != "NONE"
                previous_states[name] = state
            records.append({"contacts":current,"flat_foot":self._flat(current)})
        return ContactPlan(tuple(episodes),tuple(records),self.fps,metadata={"channels":list(CHANNELS)})

    @staticmethod
    def _provenance(channel: str) -> str:
        if channel.endswith("butt"):
            return "hip_derived_butt_proxy"
        if channel.endswith("back"):
            return "spine_surface_proxy"
        if channel.endswith("shin"):
            return "knee_ankle_midpoint_proxy"
        return "canonical_landmark"

    @staticmethod
    def _flat(contacts):
        result={}
        for side in ("left","right"):
            heel=contacts.get(f"{side}_heel",{}); toe=contacts.get(f"{side}_toe",{})
            valid=heel.get("state","NONE")!="NONE" and toe.get("state","NONE")!="NONE" and heel.get("surface_id")==toe.get("surface_id")
            result[side]=float(min(heel.get("score",0.),toe.get("score",0.))) if valid else 0.
        return result


def build_contact_plan(motion: CanonicalMotion, terrain: TerrainField, *, fps: float | None = None, transform: SceneTransform | None = None, config: dict[str, Any] | None = None) -> ContactPlan:
    return SourceContactDetector(terrain, fps or motion.fps, config).detect(_points(motion), transform)
