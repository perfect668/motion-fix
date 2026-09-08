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
        for side in ("left", "right"):
            hip = value.get(f"{side}_hip", value.get("pelvis"))
            knee = value.get(f"{side}_knee", hip)
            ankle = value.get(f"{side}_ankle", value.get(f"{side}_foot", knee))
            value.setdefault(f"{side}_shin", .5 * (knee + ankle))
            # The pelvis joint is not a butt surface.  This proxy is built
            # once here (rather than independently in a task and a solver)
            # and its provenance is carried into the contact record.
            value.setdefault(f"{side}_butt", hip - np.array([0., 0., .08]))
            value.setdefault(f"{side}_palm", value.get(f"{side}_wrist", hip))
            value.setdefault(f"{side}_heel", value.get(f"{side}_foot", ankle))
            value.setdefault(f"{side}_toe", value.get(f"{side}_foot", ankle))
        spine = value.get("spine3", value.get("pelvis"))
        pelvis = value.get("pelvis", spine)
        value.setdefault("lower_back", pelvis + np.array([-0.02, 0., .07]))
        value.setdefault("upper_back", spine + np.array([-0.02, 0., .03]))
        result.append(value)
    return result


class SourceContactDetector:
    def __init__(self, terrain: TerrainField, fps: float, config: dict[str, Any] | None = None):
        self.terrain = terrain; self.fps=float(fps); self.config=dict(config or {})
        self.enter=float(self.config.get("contact_enter_distance", .03)); self.exit=float(self.config.get("contact_exit_distance", .055))
        self.static_speed=float(self.config.get("static_tangent_speed", .08)); self.normal_speed=float(self.config.get("normal_speed_limit", .20)); self.min_score=float(self.config.get("contact_activation_score", .15))

    def detect(self, frames: list[dict[str, np.ndarray]], transform: SceneTransform | None = None) -> ContactPlan:
        transformed=[]
        for frame in frames:
            transformed.append({k:(transform.transform_points(v) if transform is not None else v.copy()) for k,v in frame.items()})
        records = []
        previous = {name: False for name in CHANNELS}
        last_points: dict[str, np.ndarray] = {}
        # A STATIC contact owns a world-space surface anchor for its complete
        # episode.  Nearest-surface queries are still useful for deciding when
        # an episode ends, but they must never make a supporting foot/palm
        # drift along a tessellated mesh while the source is stationary.
        static_anchors: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
        episodes: list[ContactEpisode] = []
        for index, frame in enumerate(transformed):
            current={}
            for name in CHANNELS:
                point=np.asarray(frame.get(name, frame.get("pelvis")),dtype=float); hit=self.terrain.support_surface(point) if ("heel" in name or "toe" in name) else self.terrain.nearest_surface(point)
                old = last_points.get(name, point)
                velocity = (point - old) * self.fps
                last_points[name] = point.copy()
                normal_speed = float(velocity @ hit.normal)
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
                    if getattr(self.terrain, "is_mesh_scene", False)
                    and hit.surface_type != "floor"
                    else signed_distance
                )
                in_contact = distance <= self.enter and normal_speed <= self.normal_speed
                if previous[name] and distance <= self.exit and normal_speed <= self.normal_speed:
                    in_contact = True
                if distance > self.exit or normal_speed > 2 * self.normal_speed:
                    in_contact = False
                score = (
                    float(np.clip(1.0 - max(distance, 0.0) / max(self.exit, 1e-6), 0.0, 1.0))
                    * float(np.clip(1.0 - max(normal_speed, 0.0) / max(self.normal_speed, 1e-6), 0.0, 1.0))
                    if in_contact
                    else 0.0
                )
                state="NONE" if not in_contact else ("STATIC" if tangent_speed <= self.static_speed else "SLIDING")
                provenance = self._provenance(name)
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
                    "surface_id": hit.surface_id,
                    "object_id": hit.surface_id.split(":", 1)[0],
                    "signed_distance": signed_distance,
                    "contact_distance": distance,
                    "normal_speed": normal_speed,
                    "tangential_speed": tangent_speed,
                    "provenance": provenance,
                }
                if state != "NONE" and not previous[name]:
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
                        )
                    )
                elif state != "NONE" and previous[name]:
                    for ep_i in range(len(episodes)-1,-1,-1):
                        ep=episodes[ep_i]
                        if ep.body_channel==name and ep.end_frame==index-1:
                            episodes[ep_i]=ContactEpisode(ep.body_channel, state, ep.start_frame,index,ep.object_id,ep.surface_id,ep.source_anchor,ep.normal); break
                previous[name]=state != "NONE"
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
