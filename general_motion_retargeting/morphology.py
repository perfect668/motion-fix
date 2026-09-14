"""Human-to-robot morphology mapping for V5.

Morphology is deliberately separate from scene coordinates.  In paired
interaction tasks the asset remains in its metric source world and only
body-relative target vectors are mapped; motion-only clips may opt into a
uniform height fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np

from .core.schemas import MorphologyTargets


def _distance(frame: dict[str, np.ndarray], first: str, second: str) -> float | None:
    if first not in frame or second not in frame:
        return None
    value = float(np.linalg.norm(np.asarray(frame[first]) - np.asarray(frame[second])))
    return value if np.isfinite(value) and value > 1e-6 else None


_CHAIN_PAIRS = {
    "left_hip_offset": ("pelvis", "left_hip"),
    "right_hip_offset": ("pelvis", "right_hip"),
    "left_upper_leg": ("left_hip", "left_knee"),
    "right_upper_leg": ("right_hip", "right_knee"),
    "left_lower_leg": ("left_knee", "left_foot"),
    "right_lower_leg": ("right_knee", "right_foot"),
    "left_foot_length": ("left_foot", "left_toe"),
    "right_foot_length": ("right_foot", "right_toe"),
    "left_shoulder_offset": ("pelvis", "left_shoulder"),
    "right_shoulder_offset": ("pelvis", "right_shoulder"),
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "left_lower_arm": ("left_elbow", "left_wrist"),
    "right_lower_arm": ("right_elbow", "right_wrist"),
}


def _point_from_model(model: mj.MjModel, data: mj.MjData, specification: dict[str, Any]) -> np.ndarray:
    """Measure one configured semantic proxy in the robot's rest pose.

    The model, rather than a hand-maintained table of NE01 dimensions, is the
    source of truth.  This also makes the morphology layer usable by a future
    robot profile so long as its semantic point configuration is supplied.
    """
    point = specification.get("robot", specification)
    sites = tuple(str(value) for value in point.get("sites", ()))
    if sites:
        positions = []
        for site in sites:
            site_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site))
            if site_id < 0:
                raise KeyError(f"Morphology semantic point references missing site {site!r}")
            positions.append(data.site_xpos[site_id])
        return np.mean(positions, axis=0)
    body_name = point.get("robot_body", point.get("body"))
    if body_name is None:
        raise ValueError("Morphology semantic point requires robot body or sites")
    body_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, str(body_name)))
    if body_id < 0:
        raise KeyError(f"Morphology semantic point references missing body {body_name!r}")
    offsets = np.asarray(
        point.get("robot_offset", point.get("offset", point.get("offsets", [[0.0, 0.0, 0.0]]))),
        dtype=float,
    ).reshape((-1, 3))
    offset = offsets.mean(axis=0)
    return data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ offset


def measure_robot_chain_lengths(
    robot_xml: str | Path,
    semantic_points: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Measure semantic chain lengths from the actual MuJoCo rest model.

    Returned provenance is exported with the retarget result.  Missing
    semantic points are reported instead of silently replaced with a guessed
    robot dimension.
    """
    model = mj.MjModel.from_xml_path(str(Path(robot_xml).expanduser().resolve()))
    data = mj.MjData(model)
    data.qpos[:] = model.qpos0
    mj.mj_forward(model, data)
    points: dict[str, np.ndarray] = {}
    missing: list[str] = []
    specifications = dict(semantic_points)
    # V5's task profile names the terminal proxy ``*_hand`` while canonical
    # human kinematics calls the equivalent landmark ``*_wrist``.  Measure it
    # once under both semantic aliases so the lower-arm ratio has a real NE01
    # end point rather than silently falling back to unity.
    for side in ("left", "right"):
        hand = f"{side}_hand"
        wrist = f"{side}_wrist"
        if wrist not in specifications and hand in specifications:
            specifications[wrist] = specifications[hand]
    for name, specification in specifications.items():
        try:
            points[str(name)] = _point_from_model(model, data, specification)
        except (KeyError, ValueError):
            missing.append(str(name))
    lengths: dict[str, float] = {}
    for name, (first, second) in _CHAIN_PAIRS.items():
        if first in points and second in points:
            value = float(np.linalg.norm(points[first] - points[second]))
            if np.isfinite(value) and value > 1e-6:
                lengths[name] = value
    return lengths, {
        "robot_xml": str(Path(robot_xml).expanduser().resolve()),
        "rest_pose": "qpos0",
        "semantic_points_measured": sorted(points),
        "semantic_points_unavailable": sorted(missing),
        "robot_chain_lengths": dict(lengths),
    }


def build_morphology_targets(
    motion,
    robot_height: float,
    policy: dict[str, Any] | None = None,
    *,
    robot_chain_lengths: dict[str, float] | None = None,
    robot_provenance: dict[str, Any] | None = None,
) -> MorphologyTargets:
    policy = dict(policy or {})
    # Scene preservation means assets retain their physical metric scale.  It
    # must not mean human limb lengths are copied verbatim into a different
    # robot.  Chain mapping is therefore the safe V5 default in every scene
    # relation; ``scene_preserving`` remains an explicit diagnostic opt-out.
    mode = str(policy.get("mode", "chain"))
    if mode not in {"scene_preserving", "motion_only_uniform", "chain"}:
        raise ValueError(f"Unknown morphology mode: {mode}")
    frames = motion.canonical_named_positions()
    samples: dict[str, list[float]] = {name: [] for name in _CHAIN_PAIRS}
    for frame in frames:
        for name, (first, second) in _CHAIN_PAIRS.items():
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
        measured = {str(name): float(value) for name, value in (robot_chain_lengths or {}).items()}
        overrides = {str(name): float(value) for name, value in policy.get("chain_scales", {}).items()}
        lower = float(policy.get("min_chain_scale", 0.35))
        upper = float(policy.get("max_chain_scale", 2.5))
        if not (0.0 < lower <= upper):
            raise ValueError("Morphology chain-scale bounds must be finite positive ordered values")
        scales = {}
        for name, observations in samples.items():
            if name in overrides:
                value = overrides[name]
            elif observations and name in measured:
                value = measured[name] / float(np.median(observations))
            else:
                # Absent semantic landmarks cannot be fitted safely.  Keep
                # that vector unscaled and expose the absence in provenance.
                value = 1.0
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"Invalid morphology chain scale for {name}: {value}")
            scales[name] = float(np.clip(value, lower, upper))
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
            "source_chain_medians": {
                name: (float(np.median(values)) if values else None)
                for name, values in samples.items()
            },
            "robot": dict(robot_provenance or {}),
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
        # Build every chain from already mapped proximal points.  Using raw
        # coordinates for a downstream segment was the subtle source of
        # inconsistent knee/ankle targets when hip width was scaled.
        for proximal, distal, key in (
            ("pelvis", f"{side}_hip", f"{side}_hip_offset"),
            (f"{side}_hip", f"{side}_knee", f"{side}_upper_leg"),
            (f"{side}_knee", f"{side}_foot", f"{side}_lower_leg"),
            (f"{side}_foot", f"{side}_toe", f"{side}_foot_length"),
            ("pelvis", f"{side}_shoulder", f"{side}_shoulder_offset"),
            (f"{side}_shoulder", f"{side}_elbow", f"{side}_upper_arm"),
            (f"{side}_elbow", f"{side}_wrist", f"{side}_lower_arm"),
        ):
            if proximal not in result or distal not in result:
                continue
            scale = float(targets.chain_scales.get(key, 1.0))
            raw_vector = np.asarray(frame[distal]) - np.asarray(frame[proximal])
            result[distal] = result[proximal] + scale * raw_vector
        # Keep aliases generated by CanonicalMotion semantically identical.
        if f"{side}_foot" in result:
            result[f"{side}_ankle"] = result[f"{side}_foot"].copy()
        if f"{side}_wrist" in result:
            result[f"{side}_hand"] = result[f"{side}_wrist"].copy()
    return result
