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
    active = [float(record.get("active_scene_collision_pairs", 0)) for record in records]
    failures = sum(bool(record.get("qp_failures") or record.get("qp_failure")) for record in records)
    summary: dict[str, Any] = {
        "frames": len(records),
        "qp_failure_count": int(failures),
        "mean_active_scene_collision_pairs": float(np.mean(active)) if active else 0.0,
        "max_active_scene_collision_pairs": int(max(active, default=0)),
        "minimum_scene_distance": float(min(values("minimum_scene_distance"), default=np.inf)),
        "maximum_penetration": float(max(values("maximum_penetration"), default=0.0)),
        "mean_collision_query_runtime_seconds": float(np.mean(values("scene_collision_query_runtime_seconds"))) if values("scene_collision_query_runtime_seconds") else 0.0,
        "mean_qp_runtime_seconds": float(np.mean(values("qp_solve_runtime_seconds"))) if values("qp_solve_runtime_seconds") else 0.0,
    }
    channel_names = sorted({name for frame in contacts for name in frame.get("contacts", frame).keys()})
    channel_summary: dict[str, Any] = {}
    for name in channel_names:
        items = [frame.get("contacts", frame).get(name, {}) for frame in contacts]
        states = [str(item.get("state", "NONE")) for item in items]
        distances = [float(item["signed_distance"]) for item in items if np.isfinite(float(item.get("signed_distance", np.nan)))]
        channel_summary[name] = {
            "contact_ratio": float(np.mean([state != "NONE" for state in states])) if states else 0.0,
            "state_counts": dict(Counter(states)),
            "surface_ids": sorted({str(item.get("surface_id", "")) for item in items if item.get("surface_id")}),
            "median_signed_distance": float(np.median(distances)) if distances else np.inf,
        }
    summary["contact_channels"] = channel_summary
    for name in ("left_butt", "right_butt", "lower_back"):
        summary[f"{name}_contact_ratio"] = channel_summary.get(name, {}).get("contact_ratio", 0.0)
    return summary
