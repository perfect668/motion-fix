"""Strict HoloSoMo position-mocap loading for the terrain retargeter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation


def load_joint_map(path: str | Path) -> dict:
    mapping = json.loads(Path(path).read_text())
    names = mapping.get("source_joint_names", [])
    if not names or len(names) != len(set(names)):
        raise ValueError("joint map source_joint_names must be nonempty and unique")
    missing = sorted(set(mapping.get("required_source_joints", [])) - set(names))
    if missing:
        raise ValueError(f"joint map required joints absent from source order: {missing}")
    return mapping


def _load_names(npz, mapping: dict, point_count: int) -> list[str]:
    if npz is not None and "joint_names" in npz.files:
        names = [str(item) for item in np.asarray(npz["joint_names"]).tolist()]
    else:
        names = [str(item) for item in mapping["source_joint_names"]]
    if len(names) != point_count:
        raise ValueError(f"Joint-name count {len(names)} does not match motion point count {point_count}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate motion joint names: {duplicates}")
    required = set(mapping["required_source_joints"])
    missing = sorted(required - set(names))
    if missing:
        raise ValueError(f"Motion is missing required HoloSoMo joints: {missing}")
    return names


def load_holosoma_positions(path: str | Path, mapping: dict, default_fps: float = 120.0):
    path = Path(path)
    npz = None
    if path.suffix.lower() == ".npy":
        positions = np.load(path)
        fps = float(default_fps)
    elif path.suffix.lower() == ".npz":
        npz = np.load(path, allow_pickle=False)
        key = "global_joint_positions" if "global_joint_positions" in npz.files else "joint_positions"
        if key not in npz.files:
            raise ValueError(f"{path} has no global_joint_positions or joint_positions")
        positions = npz[key]
        fps = float(np.asarray(npz["fps"]).item()) if "fps" in npz.files else float(default_fps)
    else:
        raise ValueError(f"Unsupported HoloSoMo motion file: {path}")
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"Expected motion shape (T,J,3), got {positions.shape}")
    if not np.isfinite(positions).all() or fps <= 0.0:
        raise ValueError("Motion positions and FPS must be finite; FPS must be positive")
    names = _load_names(npz, mapping, positions.shape[1])
    return positions, names, fps


def resample_positions(positions: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if target_fps <= 0.0:
        raise ValueError("target_fps must be positive")
    if len(positions) <= 1 or np.isclose(source_fps, target_fps):
        return np.asarray(positions, dtype=np.float64).copy()
    count = max(1, int(np.floor((len(positions) - 1) * target_fps / source_fps)) + 1)
    source_time = np.arange(len(positions), dtype=float) / source_fps
    target_time = np.arange(count, dtype=float) / target_fps
    interpolation = interp1d(source_time, positions, axis=0, kind="linear", assume_sorted=True)
    return np.asarray(interpolation(target_time), dtype=np.float64)


def _basis(left: np.ndarray, up: np.ndarray, forward_hint: np.ndarray | None = None) -> np.ndarray:
    y = left / max(float(np.linalg.norm(left)), 1e-12)
    z = up - y * float(y @ up)
    z /= max(float(np.linalg.norm(z)), 1e-12)
    x = np.cross(y, z)
    if forward_hint is not None and float(x @ forward_hint) < 0.0:
        x = -x
        z = -z
    x /= max(float(np.linalg.norm(x)), 1e-12)
    z = np.cross(x, y)
    return np.column_stack((x, y, z))


def _continuous_wxyz(rotations: list[Rotation]) -> np.ndarray:
    quaternions = np.asarray([rotation.as_quat(scalar_first=True) for rotation in rotations])
    for index in range(1, len(quaternions)):
        if float(quaternions[index - 1] @ quaternions[index]) < 0.0:
            quaternions[index] *= -1.0
    return quaternions


def build_solver_frames(positions: np.ndarray, names: list[str], mapping: dict):
    index = {name: i for i, name in enumerate(names)}
    solver_mapping = mapping["solver_joint_mapping"]
    missing = sorted(set(solver_mapping.values()) - set(index))
    if missing:
        raise ValueError(f"Solver mapping references missing joints: {missing}")

    pelvis_rotations, chest_rotations, foot_rotations = [], [], {"left": [], "right": []}
    previous_forward = None
    for frame in positions:
        hips_center = 0.5 * (frame[index["LeftUpLeg"]] + frame[index["RightUpLeg"]])
        left_axis = frame[index["LeftUpLeg"]] - frame[index["RightUpLeg"]]
        up_axis = frame[index["Spine1"]] - hips_center
        pelvis_basis = _basis(left_axis, up_axis, previous_forward)
        previous_forward = pelvis_basis[:, 0]
        pelvis_rotations.append(Rotation.from_matrix(pelvis_basis))

        shoulder_axis = frame[index["LeftArm"]] - frame[index["RightArm"]]
        chest_up = frame[index["Neck"]] - frame[index["Spine1"]]
        chest_rotations.append(Rotation.from_matrix(_basis(shoulder_axis, chest_up, pelvis_basis[:, 0])))
        for side, source_side in (("left", "Left"), ("right", "Right")):
            forward = frame[index[f"{source_side}ToeBase"]] - frame[index[f"{source_side}Foot"]]
            forward /= max(float(np.linalg.norm(forward)), 1e-12)
            left = pelvis_basis[:, 1] if side == "left" else -pelvis_basis[:, 1]
            sole_up = np.cross(forward, left)
            if sole_up[2] < 0.0:
                sole_up = -sole_up
            left_axis_foot = np.cross(sole_up, forward)
            foot_rotations[side].append(Rotation.from_matrix(np.column_stack((forward, left_axis_foot, sole_up))))

    pelvis_quat = _continuous_wxyz(pelvis_rotations)
    chest_quat = _continuous_wxyz(chest_rotations)
    foot_quat = {side: _continuous_wxyz(values) for side, values in foot_rotations.items()}
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    frames = []
    for frame_index, source in enumerate(positions):
        target = {}
        for solver_name, source_name in solver_mapping.items():
            quaternion = identity
            if solver_name == "pelvis":
                quaternion = pelvis_quat[frame_index]
            elif solver_name in {"spine1", "spine3", "neck", "head"}:
                quaternion = chest_quat[frame_index]
            elif solver_name == "left_foot":
                quaternion = foot_quat["left"][frame_index]
            elif solver_name == "right_foot":
                quaternion = foot_quat["right"][frame_index]
            target[solver_name] = (source[index[source_name]].copy(), np.asarray(quaternion).copy())
        frames.append(target)
    orientation_valid = {
        "measured": False,
        "derived": ["pelvis", "spine1", "spine3", "neck", "head", "left_foot", "right_foot"],
    }
    return frames, orientation_valid


def named_source_frames(positions: np.ndarray, names: list[str]) -> list[dict[str, np.ndarray]]:
    return [{name: frame[index].copy() for index, name in enumerate(names)} for frame in positions]
