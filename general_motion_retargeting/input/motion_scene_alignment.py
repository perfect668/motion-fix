"""Evidence-based source-motion alignment for paired scene datasets.

Some reconstructions store a metrically valid object track but an SMPL-X
stream whose vertical origin is shifted by a constant fitting offset.  The
offset must be resolved before contact detection, not hidden in robot root Z
or by moving the scene.  This module accepts only a rigid vertical correction
supported by independent source contact labels *and* support-surface geometry.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..motion_adapters import CanonicalMotion
from ..terrain_geometry import SceneTransform


_FOOT_CHANNELS = ("left_heel", "right_heel", "left_toe", "right_toe")


def _verified_label_order(motion: CanonicalMotion) -> tuple[str, ...] | None:
    """Return an explicit four-channel schema, never a guessed one."""
    value = motion.metadata.get("foot_contact_channel_order")
    if not isinstance(value, (list, tuple)):
        return None
    names = tuple(str(item) for item in value)
    return names if len(names) >= 4 and set(names[:4]) == set(_FOOT_CHANNELS) else None


def align_motion_to_scene_support(
    motion: CanonicalMotion,
    terrain,
    transform: SceneTransform,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Optionally reconcile a constant human/scene vertical frame offset.

    No robot state participates.  The correction is accepted only when at
    least ``min_samples`` labelled foot points agree on the same upward world
    translation.  The scene itself is never moved or rescaled.
    """
    config = dict(config or {})
    report: dict[str, Any] = {
        "status": "NOT_AVAILABLE",
        "translation_source": [0.0, 0.0, 0.0],
        "translation_solver": [0.0, 0.0, 0.0],
        "sample_count": 0,
    }
    if not bool(config.get("enabled", True)):
        report["status"] = "DISABLED"
        return report
    labels = motion.metadata.get("foot_contact_probs")
    if labels is None:
        return report
    labels = np.asarray(labels, dtype=float)
    if labels.ndim != 2 or labels.shape[0] != motion.frame_count or labels.shape[1] < 4:
        report["status"] = "INVALID_LABELS"
        return report
    label_threshold = float(config.get("label_threshold", 0.75))
    min_samples = max(1, int(config.get("min_samples", 8)))
    max_offset = float(config.get("max_vertical_offset", 0.25))
    max_spread = float(config.get("max_residual_spread", 0.035))
    max_span = float(config.get("max_candidate_span", 0.08))
    min_normal_z = float(config.get("support_normal_min_z", 0.6))
    # Contact labels can be trustworthy for episode timing while the source
    # body surface still has a pose-dependent marker offset.  Only use them
    # to solve a rigid SceneTransform when explicitly requested; otherwise
    # alignment remains geometry-only and cannot turn non-rigid marker error
    # into a global scene translation.
    label_order = (
        _verified_label_order(motion)
        if bool(config.get("use_labels_for_alignment", True))
        else None
    )
    # A source file with unnamed classifier columns must not be interpreted
    # as heel/toe labels.  It is still possible to establish a coordinate
    # correction from geometry alone: stationary, near-support surface
    # markers across an entire clip are independent evidence of the same
    # rigid source-to-scene transform.
    geometry_search_distance = float(config.get("geometry_search_distance", 0.08))
    geometry_normal_speed = float(config.get("geometry_normal_speed", 0.08))
    geometry_tangent_speed = float(config.get("geometry_tangent_speed", 0.08))
    label_mode = "explicit" if label_order is not None else "geometry_stationary"
    report.update({
        "label_schema": label_mode,
        "evidence_mode": "labelled_support" if label_order is not None else "stationary_support_geometry",
    })
    candidates: list[float] = []
    previous_points: dict[str, np.ndarray] = {}
    for frame, probabilities in zip(motion.canonical_named_positions(), labels):
        for channel in _FOOT_CHANNELS:
            if channel not in frame:
                continue
            point = transform.transform_points(np.asarray(frame[channel], dtype=float))
            hit = terrain.support_surface(point)
            normal = np.asarray(hit.normal, dtype=float)
            if not hit.supportable or normal[2] < min_normal_z:
                continue
            distance = float(np.linalg.norm(point - hit.closest_point))
            previous = previous_points.get(channel)
            velocity = (
                np.zeros(3, dtype=float)
                if previous is None else (point - previous) * float(motion.fps)
            )
            previous_points[channel] = point.copy()
            normal_speed = abs(float(velocity @ normal))
            tangent_speed = float(np.linalg.norm(velocity - normal * (velocity @ normal)))
            if label_order is not None:
                probability = float(probabilities[label_order.index(channel)])
                accepted = probability >= label_threshold
            else:
                # This path deliberately ignores unnamed score columns.  The
                # threshold is a search radius for a pre-alignment surface
                # marker, not an instruction to declare a contact episode.
                accepted = (
                    distance <= geometry_search_distance
                    and normal_speed <= geometry_normal_speed
                    and tangent_speed <= geometry_tangent_speed
                )
            if not accepted:
                continue
            # Solve n^T (p + z e_z - s) = 0 for the required world-z shift.
            shift = -float(normal @ (point - hit.closest_point)) / float(normal[2])
            if np.isfinite(shift) and abs(shift) <= max_offset:
                candidates.append(shift)
    report["sample_count"] = len(candidates)
    if len(candidates) < min_samples:
        report["status"] = "INSUFFICIENT_EVIDENCE"
        return report
    values = np.asarray(candidates, dtype=float)
    median = float(np.median(values))
    spread = float(1.4826 * np.median(np.abs(values - median)))
    lower, upper = (float(np.quantile(values, q)) for q in (0.05, 0.95))
    report.update({
        "vertical_offset_solver": median,
        "robust_spread": spread,
        "candidate_quantiles_solver": {"p05": lower, "p95": upper},
    })
    # A single rigid correction is meaningful only when the complete labelled
    # support evidence agrees.  MAD alone can hide a second, incompatible
    # contact regime (for example an unregistered moving scene/object track),
    # so also reject a broad central span.
    if spread > max_spread or upper - lower > max_span:
        report["status"] = "INCONSISTENT_EVIDENCE"
        return report
    # Recover the source-frame vector through the inverse linear similarity;
    # translation has no place in a vector transform.
    solver_vector = np.array([0.0, 0.0, median])
    source_vector = transform.rotation.T @ (solver_vector / transform.scale)
    motion.positions = motion.positions + source_vector.reshape(1, 1, 3)
    report.update({
        "status": "APPLIED",
        "translation_source": source_vector.tolist(),
        "translation_solver": solver_vector.tolist(),
        "label_threshold": label_threshold,
        "max_vertical_offset": max_offset,
        "geometry_search_distance": geometry_search_distance,
        "geometry_normal_speed": geometry_normal_speed,
        "geometry_tangent_speed": geometry_tangent_speed,
    })
    motion.metadata["motion_scene_alignment"] = dict(report)
    return report
