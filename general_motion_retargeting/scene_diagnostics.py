"""Diagnostics for complex-scene retargeting.

The helpers in this module are deliberately solver agnostic.  They compare
the geometry used by rendering, interaction sampling and MuJoCo, and reduce
per-frame V4 records to a reproducible sequence summary.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np


def _finite_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    if not len(points) or not np.all(np.isfinite(points)):
        raise ValueError("scene diagnostic points must be finite and non-empty")
    return points.min(axis=0), points.max(axis=0)


def alignment_sanity_check(
    visual_points: np.ndarray,
    interaction_points: np.ndarray,
    collision_points: np.ndarray,
    *,
    max_bound_error: float = 1e-3,
    max_scale_error: float = 0.01,
) -> dict[str, Any]:
    """Check that three scene representations share pose and scale.

    Points must already be expressed in the same world frame.  The check uses
    robust AABB extents rather than point-to-point correspondence (CoACD
    pieces and surface samples intentionally have different point counts).
    """
    arrays = {
        "visual": np.asarray(visual_points, dtype=float),
        "interaction": np.asarray(interaction_points, dtype=float),
        "collision": np.asarray(collision_points, dtype=float),
    }
    bounds = {name: _finite_bounds(value) for name, value in arrays.items()}
    reference_min, reference_max = bounds["visual"]
    reference_extent = reference_max - reference_min
    result: dict[str, Any] = {"passed": True, "max_bound_error": 0.0, "max_scale_error": 0.0, "representations": {}}
    for name, (lower, upper) in bounds.items():
        bound_error = float(max(np.max(np.abs(lower - reference_min)), np.max(np.abs(upper - reference_max))))
        extent = upper - lower
        valid_extent = reference_extent > 1e-8
        scale_error = float(np.max(np.abs(extent[valid_extent] / reference_extent[valid_extent] - 1.0))) if np.any(valid_extent) else 0.0
        result["representations"][name] = {"min": lower.tolist(), "max": upper.tolist(), "extent": extent.tolist(), "bound_error": bound_error, "scale_error": scale_error}
        result["max_bound_error"] = max(result["max_bound_error"], bound_error)
        result["max_scale_error"] = max(result["max_scale_error"], scale_error)
    result["passed"] = result["max_bound_error"] <= max_bound_error and result["max_scale_error"] <= max_scale_error
    if not result["passed"]:
        result["error"] = "visual, interaction and collision scene geometry are not aligned"
    return result


def summarize_scene_diagnostics(
    frame_records: Iterable[dict[str, Any]],
    contact_schedule: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate V4 frame diagnostics and generic contact channels."""
    records = list(frame_records)
    contacts = list(contact_schedule or [])
    def values(key: str) -> list[float]:
        return [float(record[key]) for record in records if np.isfinite(float(record.get(key, np.nan)))]
    collision_runtime = values("scene_collision_query_runtime_seconds") or values("collision_query_time")
    qp_runtime = values("qp_solve_runtime_seconds") or values("qp_solve_time")
    active = [float(record.get("active_scene_collision_pairs", record.get("scene_collision_active_pairs", 0))) for record in records]
    failures = sum(bool(record.get("qp_failures") or record.get("qp_failure")) for record in records)
    summary: dict[str, Any] = {
        "frames": len(records),
        "qp_failure_count": int(failures),
        "mean_active_scene_collision_pairs": float(np.mean(active)) if active else 0.0,
        "max_active_scene_collision_pairs": int(max(active, default=0)),
        "minimum_scene_distance": float(min(values("minimum_scene_distance"), default=np.inf)),
        "maximum_penetration": float(max(values("maximum_penetration") + values("maximum_scene_penetration"), default=0.0)),
        "mean_collision_query_runtime_seconds": float(np.mean(collision_runtime)) if collision_runtime else 0.0,
        "mean_qp_runtime_seconds": float(np.mean(qp_runtime)) if qp_runtime else 0.0,
    }
    selected_scene = [
        float(record.get("interaction_scene_selected_points", 0.0))
        for record in records
    ]
    selected_terrain = [
        float(record.get("interaction_terrain_selected_points", 0.0))
        for record in records
    ]
    summary["interaction_scene_selected_points_mean"] = float(np.mean(selected_scene)) if selected_scene else 0.0
    summary["interaction_scene_selected_points_max"] = int(max(selected_scene, default=0.0))
    summary["interaction_terrain_selected_points_mean"] = float(np.mean(selected_terrain)) if selected_terrain else 0.0
    summary["interaction_terrain_selected_points_max"] = int(max(selected_terrain, default=0.0))
    summary["interaction_scene_selected_frames"] = int(sum(value > 0.0 for value in selected_scene))
    channel_names = sorted({name for frame in contacts for name in frame.get("contacts", frame).keys()})
    channel_summary: dict[str, Any] = {}
    for name in channel_names:
        items = []
        for index, frame in enumerate(contacts):
            source_item = frame.get("contacts", frame).get(name, {})
            # The schedule describes the source body.  V4 diagnostics carry
            # the independently measured final robot point/state; merge both
            # so summaries never mistake source NONE for robot NONE.
            robot_item = records[index].get("contact_states", {}).get(name, {}) if index < len(records) else {}
            items.append({**source_item, **robot_item})
        states = [str(item.get("state", "NONE")) for item in items]
        robot_states = [str(item.get("robot_state", "NONE")) for item in items]
        distances = [float(item["signed_distance"]) for item in items if np.isfinite(float(item.get("signed_distance", np.nan)))]
        object_distances = []
        for item in items:
            if not item.get("object_id") or item.get("state", item.get("source_state", "NONE")) == "NONE":
                continue
            value = item.get("contact_distance", item.get("signed_distance", np.nan))
            if np.isfinite(float(value)):
                object_distances.append(float(value))
        static_slip = 0.0
        max_static_step = 0.0
        previous_static = None
        for item in items:
            point_value = item.get("robot_point")
            is_static = (
                item.get("robot_state", item.get("state")) == "STATIC"
                and point_value is not None
                and np.asarray(point_value, dtype=float).shape == (3,)
            )
            surface_id = str(item.get("surface_id", ""))
            if not is_static:
                previous_static = None
                continue
            current = np.asarray(point_value, dtype=float)
            current_normal = np.asarray(item.get("surface_normal_solver", [0.0, 0.0, 1.0]), dtype=float)
            if previous_static is None or previous_static[2] != surface_id:
                previous_static = (current, current_normal, surface_id)
                continue
            previous, previous_normal, _ = previous_static
            normal = previous_normal + current_normal
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            displacement = current - previous
            tangent_displacement = displacement - normal * float(displacement @ normal)
            step = float(np.linalg.norm(tangent_displacement))
            static_slip += step
            max_static_step = max(max_static_step, step)
            previous_static = (current, current_normal, surface_id)
        channel_summary[name] = {
            "contact_ratio": float(np.mean([state != "NONE" for state in states])) if states else 0.0,
            "state_counts": dict(Counter(states)),
            "robot_state_counts": dict(Counter(robot_states)),
            "surface_ids": sorted({str(item.get("surface_id", "")) for item in items if item.get("surface_id")}),
            "median_signed_distance": float(np.median(distances)) if distances else np.inf,
            "max_signed_distance": float(np.max(distances)) if distances else np.inf,
            "object_contact_count": len(object_distances),
            "median_object_signed_distance": float(np.median(object_distances)) if object_distances else np.inf,
            "max_object_signed_distance": float(np.max(object_distances)) if object_distances else np.inf,
            "static_tangent_slip_m": static_slip,
            "max_static_step_m": max_static_step,
        }
    summary["contact_channels"] = channel_summary
    for name in ("left_butt", "right_butt", "lower_back", "upper_back"):
        summary[f"{name}_contact_ratio"] = channel_summary.get(name, {}).get("contact_ratio", 0.0)
    seated_frames = [
        index for index, frame in enumerate(contacts)
        if any(frame.get("contacts", {}).get(name, {}).get("object_id")
               and frame.get("contacts", {}).get(name, {}).get("state", "NONE") != "NONE"
               for name in ("left_butt", "right_butt"))
    ]
    if seated_frames:
        summary["seated_phase_start_frame"] = int(min(seated_frames))
        summary["seated_phase_end_frame"] = int(max(seated_frames))
        summary["seated_phase_frame_count"] = int(len(seated_frames))
    else:
        summary["seated_phase_start_frame"] = None
        summary["seated_phase_end_frame"] = None
        summary["seated_phase_frame_count"] = 0
    # Chair-agnostic contact distances are reported for every surface type;
    # the familiar butt/back keys are retained for seated-sequence reports.
    for name in ("left_butt", "right_butt", "lower_back", "upper_back"):
        item = channel_summary.get(name, {})
        summary[f"median_{name}_surface_distance"] = item.get("median_signed_distance", np.inf)
        summary[f"max_{name}_surface_distance"] = item.get("max_signed_distance", np.inf)
        summary[f"median_{name}_object_distance"] = item.get("median_object_signed_distance", np.inf)
        summary[f"max_{name}_object_distance"] = item.get("max_object_signed_distance", np.inf)
    return summary
