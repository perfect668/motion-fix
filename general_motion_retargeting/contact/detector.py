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


_FOOT_LABEL_CHANNELS = ("left_heel", "right_heel", "left_toe", "right_toe")


def _verified_foot_label_order(motion: CanonicalMotion) -> tuple[str, ...] | None:
    """Accept source label columns only when their semantics are declared."""
    value = motion.metadata.get("foot_contact_channel_order")
    if not isinstance(value, (list, tuple)):
        return None
    order = tuple(str(item) for item in value)
    return order if len(order) >= 4 and set(order[:4]) == set(_FOOT_LABEL_CHANNELS) else None


def _points(motion: CanonicalMotion) -> list[dict[str, np.ndarray]]:
    frames = motion.canonical_named_positions()
    result = []
    foot_probabilities = motion.metadata.get("foot_contact_probs")
    if foot_probabilities is not None:
        foot_probabilities = np.asarray(foot_probabilities, dtype=float)
        if foot_probabilities.ndim != 2 or foot_probabilities.shape[0] != len(frames):
            foot_probabilities = None
    foot_label_order = _verified_foot_label_order(motion)
    for frame_index, frame in enumerate(frames):
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
        # Position-only formats may not expose heel/toe surface landmarks.
        # Retain an ankle/foot support probe strictly for SUPPORTED versus
        # FLIGHT classification; it is never emitted as a fabricated contact
        # channel or used as a desired-contact target.
        value["__support_probes__"] = {
            side: np.asarray(value[f"{side}_foot"], dtype=float).copy()
            for side in ("left", "right")
            if f"{side}_foot" in value
        }
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
        if foot_probabilities is not None:
            value["__contact_probs__"] = foot_probabilities[frame_index].copy()
            if foot_label_order is not None:
                value["__contact_prob_order__"] = foot_label_order
        result.append(value)
    return result


class SourceContactDetector:
    def __init__(self, terrain: TerrainField, fps: float, config: dict[str, Any] | None = None):
        self.terrain = terrain; self.fps=float(fps); self.config=dict(config or {})
        self.enter=float(self.config.get("contact_enter_distance", .03)); self.exit=float(self.config.get("contact_exit_distance", .055))
        self.static_speed=float(self.config.get("static_tangent_speed", .08)); self.normal_speed=float(self.config.get("normal_speed_limit", .20)); self.min_score=float(self.config.get("contact_activation_score", .15))
        self.max_penetration=float(self.config.get("max_allowed_penetration", .02))
        self.surface_normal_cos=float(self.config.get("surface_normal_cos", .95))
        self.label_threshold=float(self.config.get("source_contact_label_threshold", .75))
        # A declared source contact label may refer to a body marker with a
        # systematic surface offset (GRAIL heel/ankle proxies are one
        # example).  It can extend the *source contact episode* search radius,
        # but never disables target collision/non-penetration constraints.
        self.label_contact_max_distance=float(self.config.get("label_contact_max_distance", .18))

    @staticmethod
    def _surface_distance(point: np.ndarray, hit) -> float:
        """Return the contact timing distance used by the source detector."""
        return (
            float(np.linalg.norm(np.asarray(point, dtype=float) - hit.closest_point))
            if hit.surface_type == "mesh"
            else float(hit.signed_distance)
        )

    def _label_surface_offsets(
        self, frames: list[dict[str, Any]]
    ) -> dict[str, dict[str, float | int]]:
        """Estimate stable landmark-to-surface offsets from declared labels.

        Some position pipelines expose anatomical regressors rather than true
        skin/sole vertices.  Their world distance can therefore carry a stable
        bias even while the corresponding surface is in contact.  Calibrating
        that observation model is different from moving the actor or scene:
        CanonicalMotion and all interaction geometry remain untouched.

        Only explicitly named label channels, slowly moving support points and
        a dense bounded distance cluster are accepted.  Unlabelled or
        inconsistent sources retain the strict raw signed-distance test.
        """
        policy = self.config.get("label_surface_offset_calibration", {})
        if not isinstance(policy, dict) or not bool(policy.get("enabled", True)):
            return {}
        minimum_samples = max(2, int(policy.get("min_samples", 8)))
        cluster_width = float(policy.get("cluster_width", 0.04))
        minimum_ratio = float(policy.get("min_inlier_ratio", 0.25))
        maximum_offset = float(policy.get("max_abs_offset", 0.20))
        normal_speed_limit = float(policy.get("normal_speed", 0.08))
        tangent_speed_limit = float(policy.get("tangent_speed", 0.08))
        if cluster_width <= 0.0 or maximum_offset <= 0.0:
            return {}

        samples: dict[str, list[float]] = {name: [] for name in _FOOT_LABEL_CHANNELS}
        previous_points: dict[str, np.ndarray] = {}
        for frame_index, frame in enumerate(frames):
            if hasattr(self.terrain, "set_time"):
                self.terrain.set_time(frame_index / self.fps)
            probabilities = np.asarray(frame.get("__contact_probs__", []), dtype=float).reshape(-1)
            order = tuple(frame.get("__contact_prob_order__", ()))
            if len(probabilities) < 4 or set(order[:4]) != set(_FOOT_LABEL_CHANNELS):
                continue
            for name in _FOOT_LABEL_CHANNELS:
                if name not in frame:
                    continue
                point = np.asarray(frame[name], dtype=float).reshape(3)
                hit = self.terrain.support_surface(point)
                previous = previous_points.get(name)
                velocity = (
                    np.zeros(3, dtype=float)
                    if previous is None else (point - previous) * self.fps
                )
                previous_points[name] = point.copy()
                normal = np.asarray(hit.normal, dtype=float).reshape(3)
                normal_speed = abs(float(velocity @ normal))
                tangent_speed = float(np.linalg.norm(velocity - normal * (velocity @ normal)))
                probability = float(probabilities[order.index(name)])
                distance = self._surface_distance(point, hit)
                if (
                    probability >= self.label_threshold
                    and bool(hit.supportable)
                    and np.isfinite(distance)
                    and abs(distance) <= maximum_offset
                    and normal_speed <= normal_speed_limit
                    and tangent_speed <= tangent_speed_limit
                ):
                    samples[name].append(distance)

        result: dict[str, dict[str, float | int]] = {}
        for name, values in samples.items():
            ordered = np.sort(np.asarray(values, dtype=float))
            if len(ordered) < minimum_samples:
                continue
            best: tuple[int, float, int, int] | None = None
            for start, lower in enumerate(ordered):
                end = int(np.searchsorted(ordered, lower + cluster_width, side="right"))
                count = end - start
                median = float(np.median(ordered[start:end]))
                candidate = (count, -abs(median), -start, end)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
            assert best is not None
            count, _, negative_start, end = best
            start = -negative_start
            inliers = ordered[start:end]
            if count < minimum_samples or count / len(ordered) < minimum_ratio:
                continue
            offset = float(np.median(inliers))
            spread = float(1.4826 * np.median(np.abs(inliers - offset)))
            result[name] = {
                "offset": offset,
                "sample_count": int(len(ordered)),
                "inlier_count": int(count),
                "inlier_ratio": float(count / len(ordered)),
                "robust_spread": spread,
            }
        return result

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
                elif key == "__contact_probs__":
                    transformed_frame[key] = np.asarray(value, dtype=float).reshape(-1).copy()
                elif key == "__contact_prob_order__":
                    transformed_frame[key] = tuple(str(item) for item in value)
                elif key == "__support_probes__":
                    transformed_frame[key] = {
                        str(side): (
                            transform.transform_points(np.asarray(point, dtype=float).reshape(3))
                            if transform is not None else np.asarray(point, dtype=float).reshape(3).copy()
                        )
                        for side, point in value.items()
                    }
                else:
                    array = np.asarray(value, dtype=float).reshape(3)
                    transformed_frame[key] = transform.transform_points(array) if transform is not None else array.copy()
            transformed.append(transformed_frame)
        label_surface_offsets = self._label_surface_offsets(transformed)
        records = []
        previous = {name: False for name in CHANNELS}
        previous_states = {name: "NONE" for name in CHANNELS}
        blend_frames = max(1, int(self.config.get("contact_blend_frames", 7)))
        blend_alpha = 1.0 / float(blend_frames)
        blended_scores = {name: 0.0 for name in CHANNELS}
        last_points: dict[str, np.ndarray] = {}
        last_surface_points: dict[str, np.ndarray] = {}
        last_surface_ids: dict[str, str] = {}
        last_support_probes: dict[str, np.ndarray] = {}
        # A STATIC contact owns a world-space surface anchor for its complete
        # episode.  Nearest-surface queries are still useful for deciding when
        # an episode ends, but they must never make a supporting foot/palm
        # drift along a tessellated mesh while the source is stationary.
        static_anchors: dict[
            str,
            tuple[
                np.ndarray, np.ndarray, str,
                np.ndarray | None, np.ndarray | None,
            ],
        ] = {}
        episode_surfaces: dict[str, str] = {}
        episodes: list[ContactEpisode] = []
        for index, frame in enumerate(transformed):
            if hasattr(self.terrain, "set_time"):
                self.terrain.set_time(index / self.fps)
            current={}
            proxy_candidates = frame.get("__proxy_candidates__", {})
            contact_probs = np.asarray(frame.get("__contact_probs__", []), dtype=float).reshape(-1)
            contact_prob_order = tuple(frame.get("__contact_prob_order__", ()))
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
                previous_surface_point = last_surface_points.get(name)
                surface_jump = (
                    0.0
                    if previous_surface_point is None
                    else float(np.linalg.norm(
                        np.asarray(hit.closest_point, dtype=float)
                        - previous_surface_point
                    ))
                )
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
                raw_distance = self._surface_distance(point, hit)
                calibration = label_surface_offsets.get(name)
                calibrated_distance = (
                    raw_distance - float(calibration["offset"])
                    if calibration is not None else raw_distance
                )
                # Deep penetration is not evidence of desired contact.  For
                # analytic solids retain the signed-distance lower bound; mesh
                # queries expose Euclidean contact distance and leave collision
                # penetration to MuJoCo.
                deep_penetration = hit.surface_type != "mesh" and signed_distance < -self.max_penetration
                support_ok = (not ("heel" in name or "toe" in name)) or bool(hit.supportable)
                label_value = None
                if name in contact_prob_order:
                    label_index = contact_prob_order.index(name)
                    if label_index < len(contact_probs):
                        label_value = float(contact_probs[label_index])
                label_contact = bool(
                    label_value is not None
                    and label_value >= self.label_threshold
                    and support_ok
                    and abs(raw_distance) <= self.label_contact_max_distance
                    and normal_speed_abs <= self.normal_speed
                    # A dataset label may compensate for a bounded, clip-level
                    # landmark offset only after the independent calibration
                    # above found a stable low-speed surface-distance cluster.
                    # Without that observation model, deep penetration remains
                    # contradictory geometry and cannot become contact.
                    and (
                        calibration is not None
                        and abs(calibrated_distance) <= float(
                            self.config.get("label_calibrated_max_residual", 0.08)
                        )
                        or not deep_penetration
                    )
                )
                geometric_contact = bool(
                    support_ok
                    and raw_distance <= self.enter
                    and normal_speed_abs <= self.normal_speed
                    and not deep_penetration
                )
                # Geometry remains the primary signal.  A verified source
                # label is an auxiliary signal only within a bounded search
                # radius, useful when the marker is consistently offset from
                # the actual sole.  The residual is retained below for QA.
                in_contact = geometric_contact or label_contact
                detection_distance = calibrated_distance if label_contact else raw_distance
                if previous[name] and detection_distance <= self.exit and normal_speed_abs <= self.normal_speed:
                    in_contact = (not deep_penetration) or label_contact
                if detection_distance > self.exit or normal_speed_abs > 2 * self.normal_speed:
                    in_contact = bool(label_contact)
                geometry_score = (
                    float(np.clip(1.0 - max(raw_distance, 0.0) / max(self.exit, 1e-6), 0.0, 1.0))
                    * float(np.clip(1.0 - normal_speed_abs / max(self.normal_speed, 1e-6), 0.0, 1.0))
                )
                label_score = (
                    float(np.clip(label_value, 0.0, 1.0))
                    * float(np.clip(
                        1.0 - abs(calibrated_distance)
                        / max(float(self.config.get("label_calibrated_max_residual", 0.08)), 1e-6),
                        0.0,
                        1.0,
                    ))
                    * float(np.clip(1.0 - normal_speed_abs / max(self.normal_speed, 1e-6), 0.0, 1.0))
                    if label_contact else 0.0
                )
                score = max(geometry_score if geometric_contact else 0.0, label_score) if in_contact else 0.0
                if score < self.min_score:
                    if label_contact and score > 0.0:
                        # A verified source label already passed the bounded
                        # distance, penetration, support-normal and velocity
                        # gates above.  Keep it as a low-confidence desired
                        # contact instead of turning the frame into NONE just
                        # because its speed factor lowered the blended score.
                        # The continuous score still controls task weight.
                        in_contact = True
                    else:
                        in_contact = False
                        score = 0.0
                # Keep contact state semantic and independent from task
                # activation.  A newly detected heel/toe or butt contact is
                # blended over several solver frames; release uses the same
                # ramp, preventing a single frame from pulling the leg into a
                # different IK branch.
                blended_scores[name] += blend_alpha * (float(score) - blended_scores[name])
                activation = float(np.clip(blended_scores[name], 0.0, 1.0))
                state = "NONE" if not in_contact else ("STATIC" if tangent_speed <= self.static_speed else "SLIDING")
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
                surface_transition = bool(
                    state != "NONE"
                    and last_surface_ids.get(name) is not None
                    and str(last_surface_ids[name]) != str(hit.surface_id)
                    and surface_jump >= float(self.config.get(
                        "surface_transition_distance", 0.04
                    ))
                )
                # Only a confirmed query-surface change is a terrain
                # transition.  Low activation or normal motion alone is not
                # enough: ordinary walking contacts use the regular blend
                # and support replay rather than being exempted wholesale.
                support_transition = surface_transition
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
                        local_anchor = getattr(hit, "asset_local_anchor", None)
                        local_normal = getattr(hit, "asset_local_normal", None)
                        anchor = (
                            hit.closest_point.copy(),
                            hit.normal.copy(),
                            str(hit.surface_id),
                            None if local_anchor is None else np.asarray(local_anchor, dtype=float).copy(),
                            None if local_normal is None else np.asarray(local_normal, dtype=float).copy(),
                        )
                        static_anchors[name] = anchor
                    (
                        anchor_point, anchor_normal, anchor_surface,
                        anchor_asset_local, anchor_asset_normal,
                    ) = anchor
                else:
                    static_anchors.pop(name, None)
                    anchor_point, anchor_normal, anchor_surface = (
                        hit.closest_point.copy(), hit.normal.copy(), str(hit.surface_id)
                    )
                    local_anchor = getattr(hit, "asset_local_anchor", None)
                    local_normal = getattr(hit, "asset_local_normal", None)
                    anchor_asset_local = (
                        None if local_anchor is None
                        else np.asarray(local_anchor, dtype=float).copy()
                    )
                    anchor_asset_normal = (
                        None if local_normal is None
                        else np.asarray(local_normal, dtype=float).copy()
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
                    "surface_transition": surface_transition,
                    "surface_jump": surface_jump,
                    "support_transition": support_transition,
                    "object_id": locked_surface.split(":", 1)[0],
                    "signed_distance": signed_distance,
                    "contact_distance": detection_distance,
                    "raw_contact_distance": raw_distance,
                    "landmark_surface_offset": (
                        None if calibration is None else float(calibration["offset"])
                    ),
                    "normal_speed": normal_speed,
                    "tangential_speed": tangent_speed,
                    "provenance": provenance,
                    "triangle_id": getattr(hit, "triangle_id", None),
                    "barycentric": getattr(hit, "barycentric", None),
                    # STATIC world and asset-local anchors are one immutable
                    # surface frame.  Combining a locked world point with the
                    # current frame's nearest-triangle local coordinate makes
                    # the target drift when it is materialized by the solver.
                    "asset_local_anchor": anchor_asset_local,
                    "asset_local_normal": anchor_asset_normal,
                    "source_contact_label": label_value,
                    "label_contact": bool(label_contact),
                    "geometry_contact": bool(geometric_contact),
                    "contact_geometry_residual": float(abs(calibrated_distance)) if label_contact else 0.0,
                }
                # Keep the last normal in the detector state without exposing
                # it as a synthetic motion landmark.
                last_points[f"{name}::__normal"] = hit.normal.copy()
                last_surface_points[name] = np.asarray(
                    hit.closest_point, dtype=float
                ).copy()
                last_surface_ids[name] = str(hit.surface_id)
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
                            asset_local_anchor=(
                                anchor_point.copy()
                                if anchor_asset_local is None
                                else anchor_asset_local.copy()
                            ),
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
            foot_items = [
                current[name] for name in
                ("left_heel", "left_toe", "right_heel", "right_toe")
                if name in current and current[name].get("provenance") != "missing_landmark"
            ]
            expected_distance = float(self.config.get(
                "expected_support_distance", self.exit
            ))
            expected_penetration = float(self.config.get(
                "expected_support_penetration", self.max_penetration
            ))
            support_probe_distance = float(self.config.get(
                "support_probe_distance", max(self.exit, 0.12)
            ))
            support_probe_speed = float(self.config.get(
                "support_probe_normal_speed", self.normal_speed
            ))
            support_probes = []
            for side, probe_value in frame.get("__support_probes__", {}).items():
                probe = np.asarray(probe_value, dtype=float).reshape(3)
                hit_probe = self.terrain.support_surface(probe)
                previous_probe = last_support_probes.get(str(side), probe)
                probe_velocity = (probe - previous_probe) * self.fps
                last_support_probes[str(side)] = probe.copy()
                probe_distance = (
                    float(np.linalg.norm(probe - hit_probe.closest_point))
                    if hit_probe.surface_type == "mesh"
                    else float(hit_probe.signed_distance)
                )
                probe_normal_speed = abs(float(probe_velocity @ hit_probe.normal))
                if (
                    hit_probe.supportable
                    and -expected_penetration <= probe_distance <= support_probe_distance
                    and probe_normal_speed <= support_probe_speed
                ):
                    support_probes.append({
                        "side": str(side),
                        "distance": probe_distance,
                        "surface_id": str(hit_probe.surface_id),
                        "normal_speed": probe_normal_speed,
                        "provenance": "ankle_foot_support_probe",
                    })
            label_threshold = self.label_threshold
            # This is source evidence, not a robot-state feedback loop.  It
            # tells final validation when the source was close enough to a
            # support surface that a zero robot support would be suspicious.
            signed_distances = np.asarray(
                [float(item.get("contact_distance", np.nan)) for item in foot_items],
                dtype=float,
            )
            label_support_distance = self.label_contact_max_distance
            def _support_distance_ok(item):
                value = float(item.get("contact_distance", np.inf))
                if -expected_penetration <= value <= expected_distance:
                    return True
                return bool(
                    item.get("label_contact", False)
                    and abs(value) <= label_support_distance
                )

            near_support = bool(
                any(
                    (-expected_penetration <= value <= expected_distance)
                    or abs(value) <= label_support_distance
                    for value in signed_distances if np.isfinite(value)
                )
            )
            label_support = bool(any(
                item.get("label_contact", False) and _support_distance_ok(item)
                for item in foot_items
            ))
            # A transient toe crossing a stair edge is not, by itself, a
            # reliable support episode.  ``support_expected`` is consumed by
            # final robot replay, so require the same evidence used to score
            # contact: a geometrically close foot with a meaningful blended
            # activation, or an explicitly high source label.  This keeps
            # flight/transfer frames out of support validation without
            # disabling collision or contact detection for those frames.
            reliable_activation = float(self.config.get(
                "support_activation_threshold", max(self.min_score, 0.35)
            ))
            dynamic_support = bool(any(
                str(item.get("state", "NONE")) != "NONE"
                and float(item.get("activation", 0.0)) >= reliable_activation
                and _support_distance_ok(item)
                for item in foot_items
            ))
            # Activation is intentionally blended over several frames and is
            # therefore not valid evidence for the first frame of a support
            # episode.  Geometry itself is authoritative here: a measured
            # heel/toe with a non-NONE state, a close support surface and a
            # bounded normal speed is enough to classify the source as
            # SUPPORTED.  This prevents the initial standing frame from being
            # marked UNKNOWN merely because its blend weight is 1/7.
            geometry_support = bool(any(
                str(item.get("state", "NONE")) != "NONE"
                and float(item.get("score", 0.0)) >= self.min_score
                and float(item.get("normal_speed", np.inf)) <= support_probe_speed
                and _support_distance_ok(item)
                for item in foot_items
            ))
            probe_support = bool(support_probes) if not foot_items else False
            support_expected = bool(
                (dynamic_support or geometry_support or label_support)
                and any(
                    bool(item.get("surface_id"))
                    and _support_distance_ok(item)
                    for item in foot_items
                )
            )
            if not foot_items and probe_support:
                support_expected = True
            has_declared_labels = bool(contact_prob_order)
            declared_label_support = bool(any(
                item.get("label_contact", False) for item in foot_items
            ))
            # The solver must distinguish a source flight phase from absent
            # or inconsistent source geometry.  In particular, a heel deeply
            # inside an inferred floor/mesh is not valid evidence that the
            # robot may float or that it has a usable support contact.
            if support_expected:
                support_state = "SUPPORTED"
            elif not foot_items or not np.isfinite(signed_distances).any():
                support_state = "UNKNOWN"
            elif index > 0 and near_support and all(
                str(item.get("state", "NONE")) == "NONE" for item in foot_items
            ) and any(
                max(
                    abs(float(item.get("normal_speed", 0.0))),
                    float(item.get("tangential_speed", 0.0)),
                ) > self.static_speed
                for item in foot_items
            ):
                # A foot crossing a tread/edge can be geometrically close to
                # a support surface while its smooth contact score is below
                # the enter threshold.  If it is clearly moving and no foot
                # marker is deeply inside the surface, this is a flight/
                # transfer frame rather than unknown support evidence.  A
                # stationary near-surface frame remains UNKNOWN and therefore
                # fails closed in final replay.
                support_state = "FLIGHT"
            elif np.any(signed_distances < -expected_penetration) and not label_support:
                # A deeply inconsistent stationary sample is still UNKNOWN,
                # but a moving sample was handled above as a transfer/flight
                # frame.  The ordering is intentional: source landmark bias
                # during a swing must not turn an otherwise valid flight phase
                # into a required support frame.
                support_state = "UNKNOWN"
            elif all(str(item.get("state", "NONE")) == "NONE" for item in foot_items) and (
                has_declared_labels and not declared_label_support
            ) and all(
                str(item.get("surface_id", "")) == "floor" for item in foot_items
            ):
                # With an explicit source contact schema, a frame in which
                # every label is below the enter threshold is a source flight
                # transition even if a noisy body marker lies below the
                # analytic floor.  Unlabelled formats remain fail-closed.
                # This does not disable target collision; it only avoids
                # treating absent source contact as unknown support evidence.
                support_state = "FLIGHT"
            elif np.all(signed_distances > expected_distance):
                support_state = "FLIGHT"
            else:
                support_state = "UNKNOWN"
            records.append({
                "contacts": current,
                "support_transition": bool(any(
                    item.get("support_transition", False)
                    for item in current.values()
                )),
                "support_transition_channels": tuple(
                    name for name, item in current.items()
                    if item.get("support_transition", False)
                ),
                "flat_foot": self._flat(current),
                "support_expected": support_expected,
                "support_state": support_state,
                "support_evidence": {
                    "dynamic_support": dynamic_support,
                    "geometry_support": geometry_support,
                    "label_support": label_support,
                    "probe_support": probe_support,
                    "support_probes": support_probes,
                    "reliable_activation_threshold": reliable_activation,
                    "foot_channels": {
                        name: {
                            "state": str(current[name].get("state", "NONE")),
                            "activation": float(current[name].get("activation", 0.0)),
                            "contact_distance": float(current[name].get("contact_distance", np.inf)),
                            "surface_id": str(current[name].get("surface_id", "")),
                        }
                        for name in ("left_heel", "left_toe", "right_heel", "right_toe")
                        if name in current
                    },
                },
            })
        return ContactPlan(
            tuple(episodes), tuple(records), self.fps,
            metadata={
                "channels": list(CHANNELS),
                "foot_channels_available": bool(any(
                    frame.get("support_expected", False) or any(
                        item.get("provenance") != "missing_landmark"
                        for name, item in frame.get("contacts", {}).items()
                        if name in {"left_heel", "left_toe", "right_heel", "right_toe"}
                    ) for frame in records
                )),
                "support_expected_frames": int(sum(
                    frame.get("support_expected", False) for frame in records
                )),
                "support_states": {
                    state: int(sum(frame.get("support_state") == state for frame in records))
                    for state in ("SUPPORTED", "FLIGHT", "UNKNOWN")
                },
                "label_surface_offsets": label_surface_offsets,
            },
        )

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
