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
    # ``canonical_named_positions`` is the single semantic boundary.  Do not
    # resolve a second set of aliases from the raw adapter names here: that
    # previously made SMPL-X ``left_foot`` disagree with its canonical ankle
    # point and caused contact and solver targets to use different geometry.
    canonical_frames = motion.canonical_named_positions()
    required = (
        "pelvis", "spine3", "left_hip", "right_hip", "left_knee", "right_knee",
        "left_foot", "right_foot", "left_toe", "right_toe", "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    )
    missing = [name for name in required if any(name not in frame for frame in canonical_frames)]
    if missing:
        raise ValueError(f"Canonical motion is missing required landmarks: {missing}")
    source_frames = []
    solver_frames = []
    previous_pelvis_quaternion: np.ndarray | None = None
    for t in range(motion.frame_count):
        source = {
            name: np.asarray(value, dtype=float).copy()
            for name, value in canonical_frames[t].items()
        }
        for side in ("left", "right"):
            source[f"{side}_hand"] = source[f"{side}_wrist"].copy()
        source_frames.append(source)
        # The global/root orientation is always rebuilt from positions.  The
        # adapter may retain validated local orientations for future point
        # tasks, but source file conventions must never decide the base frame.
        pelvis_quaternion = _pelvis_orientation(source, previous_pelvis_quaternion)
        previous_pelvis_quaternion = pelvis_quaternion
        targets = {}
        for semantic in required:
            quat = pelvis_quaternion if semantic == "pelvis" else np.array([1., 0., 0., 0.])
            targets[semantic]=(source[semantic],quat)
        solver_frames.append(targets)
    return source_frames, solver_frames, bool(motion.orientation_valid)
