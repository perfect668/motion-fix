"""Format-neutral conversion from CanonicalMotion to solver frame targets."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _normalised(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    return vector / length if length > 1e-9 else np.asarray(fallback, dtype=float)


def _pelvis_orientation(source: dict[str, np.ndarray], previous: np.ndarray | None) -> np.ndarray:
    """Build a stable Z-up pelvis frame from landmarks.

    Position-only inputs (including many FBX/BVH exports) must not inject a
    unit quaternion.  That silently forces the NE01 base to a world-aligned
    orientation even when the actor is turning, which is a common cause of
    mirrored hips and a backwards knee branch.  The lateral hip axis and the
    pelvis-to-spine axis are available in every V5-required landmark set.
    """
    lateral = _normalised(
        source["left_hip"] - source["right_hip"], np.array([0.0, 1.0, 0.0])
    )
    up = _normalised(
        source.get("spine3", source["pelvis"]) - source["pelvis"],
        np.array([0.0, 0.0, 1.0]),
    )
    forward = np.cross(lateral, up)
    if np.linalg.norm(forward) < 1e-8:
        forward = np.array([1.0, 0.0, 0.0])
    forward = _normalised(forward, np.array([1.0, 0.0, 0.0]))
    up = _normalised(np.cross(forward, lateral), np.array([0.0, 0.0, 1.0]))
    matrix = np.column_stack((forward, lateral, up))
    quaternion = Rotation.from_matrix(matrix).as_quat(scalar_first=True)
    if previous is not None and float(np.dot(previous, quaternion)) < 0.0:
        quaternion *= -1.0
    return quaternion


def build_solver_inputs(motion):
    """Return source dictionaries and position/quaternion target dictionaries.

    Canonical adapters already normalized names.  This module deliberately
    contains no V3/V4 imports and no dataset-specific branches.
    """
    names = set(motion.joint_names); index = {n: i for i, n in enumerate(motion.joint_names)}
    aliases = {
        "pelvis": ("pelvis", "Hips", "hips"), "spine3": ("spine3", "Spine1", "spine"),
        "left_hip": ("left_hip", "LeftUpLeg"), "right_hip": ("right_hip", "RightUpLeg"),
        "left_knee": ("left_knee", "LeftLeg"), "right_knee": ("right_knee", "RightLeg"),
        "left_foot": ("left_foot", "LeftFoot"), "right_foot": ("right_foot", "RightFoot"),
        "left_toe": ("left_toe", "LeftToeBase", "LeftToe"), "right_toe": ("right_toe", "RightToeBase", "RightToe"),
        "left_shoulder": ("left_shoulder", "LeftArm"), "right_shoulder": ("right_shoulder", "RightArm"),
        "left_elbow": ("left_elbow", "LeftForeArm"), "right_elbow": ("right_elbow", "RightForeArm"),
        "left_wrist": ("left_wrist", "LeftHandMiddle3", "LeftHand"), "right_wrist": ("right_wrist", "RightHandMiddle3", "RightHand"),
    }
    resolved = {key: next((name for name in values if name in names), None) for key, values in aliases.items()}
    derivable_toes = {
        "left_toe": ("left_big_toe", "left_small_toe"),
        "right_toe": ("right_big_toe", "right_small_toe"),
    }
    required = [key for key in aliases if resolved[key] is None and not (
        key in derivable_toes and all(item in names for item in derivable_toes[key])
    )]
    if required: raise ValueError(f"Canonical motion is missing required landmarks: {required}")
    source_frames = []
    solver_frames = []
    previous_pelvis_quaternion: np.ndarray | None = None
    for t in range(motion.frame_count):
        source={}
        for semantic, original in resolved.items():
            if original is not None:
                source[semantic] = motion.positions[t, index[original]].copy()
                source[original] = source[semantic].copy()
        for semantic, (first, second) in derivable_toes.items():
            if semantic not in source and first in index and second in index:
                source[semantic] = 0.5 * (motion.positions[t, index[first]] + motion.positions[t, index[second]])
        for side in ("left", "right"):
            if f"{side}_wrist" in source:
                source[f"{side}_hand"] = source[f"{side}_wrist"].copy()
        source_frames.append(source)
        # The global/root orientation is always rebuilt from positions.  The
        # adapter may retain validated local orientations for future point
        # tasks, but source file conventions must never decide the base frame.
        pelvis_quaternion = _pelvis_orientation(source, previous_pelvis_quaternion)
        previous_pelvis_quaternion = pelvis_quaternion
        targets = {}
        for semantic, original in resolved.items():
            if semantic not in source:
                continue
            quat = pelvis_quaternion if semantic == "pelvis" else np.array([1., 0., 0., 0.])
            targets[semantic]=(source[semantic],quat)
        solver_frames.append(targets)
    return source_frames, solver_frames, bool(motion.orientation_valid)
