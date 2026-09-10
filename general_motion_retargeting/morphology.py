"""Human-to-robot morphology mapping for V5.

Morphology is deliberately separate from scene coordinates.  In paired
interaction tasks the asset remains in its metric source world and only
body-relative target vectors are mapped; motion-only clips may opt into a
uniform height fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .core.schemas import MorphologyTargets


def _distance(frame: dict[str, np.ndarray], first: str, second: str) -> float | None:
    if first not in frame or second not in frame:
        return None
    value = float(np.linalg.norm(np.asarray(frame[first]) - np.asarray(frame[second])))
    return value if np.isfinite(value) and value > 1e-6 else None


def build_morphology_targets(motion, robot_height: float, policy: dict[str, Any] | None = None) -> MorphologyTargets:
    policy = dict(policy or {})
    mode = str(policy.get("mode", "scene_preserving"))
    if mode not in {"scene_preserving", "motion_only_uniform", "chain"}:
        raise ValueError(f"Unknown morphology mode: {mode}")
    frames = motion.canonical_named_positions()
    samples: dict[str, list[float]] = {
        "left_upper_leg": [], "right_upper_leg": [],
        "left_lower_leg": [], "right_lower_leg": [],
        "left_upper_arm": [], "right_upper_arm": [],
        "right_lower_arm": [], "left_lower_arm": [],
    }
    pairs = {
        "left_upper_leg": ("left_hip", "left_knee"),
        "right_upper_leg": ("right_hip", "right_knee"),
        "left_lower_leg": ("left_knee", "left_ankle"),
        "right_lower_leg": ("right_knee", "right_ankle"),
        "left_upper_arm": ("left_shoulder", "left_elbow"),
        "right_upper_arm": ("right_shoulder", "right_elbow"),
        "left_lower_arm": ("left_elbow", "left_wrist"),
        "right_lower_arm": ("right_elbow", "right_wrist"),
    }
    for frame in frames:
        for name, (first, second) in pairs.items():
            value = _distance(frame, first, second)
            if value is not None:
                samples[name].append(value)
    reference_height = float(motion.human_height or policy.get("default_human_height", 1.78))
    if reference_height <= 0.0:
        raise ValueError("Morphology source human height must be positive")
    if mode == "motion_only_uniform":
        uniform = float(robot_height) / reference_height
        scales = {name: uniform for name in samples}
    elif mode == "chain":
        # Chain ratios are optional and only affect relative vectors in the
        # solver.  Missing robot measurements remain identity until a robot
        # profile supplies them explicitly.
        scales = {name: float(value) for name, value in policy.get("chain_scales", {}).items()}
    else:
        scales = {name: 1.0 for name in samples}
    return MorphologyTargets(
        human_height=motion.human_height,
        robot_height=float(robot_height),
        chain_scales=scales,
        mode=mode,
        provenance={
            "policy": policy,
            "source_height": reference_height,
            "scene_preserved": mode == "scene_preserving",
        },
    )


def map_semantic_frame(frame: dict[str, np.ndarray], targets: MorphologyTargets) -> dict[str, np.ndarray]:
    """Map only non-contact body-relative vectors, preserving root position."""
    if targets.mode == "scene_preserving":
        return {name: np.asarray(value, dtype=float).copy() for name, value in frame.items()}
    result = {name: np.asarray(value, dtype=float).copy() for name, value in frame.items()}
    pelvis = result.get("pelvis")
    if pelvis is None:
        return result
    for side in ("left", "right"):
        for proximal, distal, key in (
            (f"{side}_hip", f"{side}_knee", f"{side}_upper_leg"),
            (f"{side}_knee", f"{side}_ankle", f"{side}_lower_leg"),
            (f"{side}_shoulder", f"{side}_elbow", f"{side}_upper_arm"),
            (f"{side}_elbow", f"{side}_wrist", f"{side}_lower_arm"),
        ):
            if proximal not in result or distal not in result:
                continue
            scale = float(targets.chain_scales.get(key, 1.0))
            result[distal] = result[proximal] + scale * (result[distal] - result[proximal])
    return result
