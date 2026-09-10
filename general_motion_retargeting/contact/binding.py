"""Target-scene contact binding and robot contact realization for V5.

SourceContactDetector produces evidence in the source scene.  This module is
the only boundary that turns that evidence into target-scene anchors; the
solver never performs an implicit source-to-target lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.schemas import ContactPlan, SceneRelation


class ContactBindingError(ValueError):
    pass


@dataclass(frozen=True)
class TargetContactSchedule:
    per_frame_states: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


def _copy_item(item: dict[str, Any]) -> dict[str, Any]:
    result = dict(item)
    for key, value in list(result.items()):
        if isinstance(value, np.ndarray):
            result[key] = value.copy()
    return result


class ContactBinder:
    """Bind source surface identity/anchors to a target scene explicitly."""

    def bind(
        self,
        plan: ContactPlan,
        *,
        relation: SceneRelation,
        binding: dict[str, Any] | None = None,
        target_terrain=None,
    ) -> TargetContactSchedule:
        binding = dict(binding or {})
        if relation == SceneRelation.SOURCE_TO_TARGET and not binding:
            raise ContactBindingError("source_to_target requires explicit contact_binding")
        surface_map = binding.get("surface_map", binding.get("surfaces", {}))
        transform = np.asarray(binding["matrix"], dtype=float).reshape(4, 4) if binding.get("matrix") is not None else None
        frames = []
        for frame in plan.per_frame_states:
            contacts = {}
            for channel, original in frame.get("contacts", {}).items():
                item = _copy_item(original)
                source_surface = str(item.get("surface_id", ""))
                target_surface = surface_map.get(source_surface, source_surface)
                if relation == SceneRelation.SOURCE_TO_TARGET and source_surface and source_surface not in surface_map:
                    raise ContactBindingError(
                        f"No target surface mapping for source surface {source_surface!r}"
                    )
                for key in ("surface_point_solver", "tangent_anchor_solver", "human_point_solver"):
                    if transform is not None and key in item:
                        point = np.asarray(item[key], dtype=float).reshape(3)
                        item[key] = (transform[:3, :3] @ point) + transform[:3, 3]
                if transform is not None:
                    for key in ("surface_normal_solver", "anchor_normal_solver"):
                        if key in item:
                            normal = transform[:3, :3] @ np.asarray(item[key], dtype=float)
                            item[key] = normal / max(float(np.linalg.norm(normal)), 1e-12)
                item["source_surface_id"] = source_surface
                item["surface_id"] = str(target_surface)
                item["target_provenance"] = "explicit_surface_binding" if relation == SceneRelation.SOURCE_TO_TARGET else "shared_scene_anchor"
                if target_terrain is not None and item.get("state", "NONE") != "NONE":
                    # Validate the bound target is a real surface.  We do not
                    # replace the explicit anchor with a nearest point.
                    target_point = np.asarray(item.get("surface_point_solver", [0, 0, 0]), dtype=float)
                    target_hit = target_terrain.nearest_surface(target_point)
                    item["target_surface_distance"] = float(target_hit.signed_distance)
                contacts[channel] = item
            frames.append({**{k: v for k, v in frame.items() if k != "contacts"}, "contacts": contacts})
        return TargetContactSchedule(tuple(frames), {
            "relation": relation.value,
            "binding": binding,
            "source_episode_count": len(plan.episodes),
        })


class RobotContactRealizer:
    """Attach canonical contact channels to configured robot proxy points."""

    def realize(self, schedule: TargetContactSchedule, robot_points: dict[str, Any]) -> TargetContactSchedule:
        available = set(robot_points)
        frames = []
        for frame in schedule.per_frame_states:
            contacts = {}
            for channel, item in frame.get("contacts", {}).items():
                value = _copy_item(item)
                value["robot_proxy"] = channel if channel in available else None
                value["robot_provenance"] = "configured_robot_proxy" if channel in available else "unconfigured_channel"
                contacts[channel] = value
            frames.append({**{k: v for k, v in frame.items() if k != "contacts"}, "contacts": contacts})
        metadata = dict(schedule.metadata)
        metadata["robot_channels"] = sorted(available)
        return TargetContactSchedule(tuple(frames), metadata)
