"""Canonical loading and normalization for Sonic SMPL motion files."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


POSE_KEYS = ("pose_aa", "poses", "original_pose_aa")
TRANS_KEYS = ("transl", "trans", "translation", "root_trans")
FPS_KEYS = ("fps", "mocap_framerate", "mocap_frame_rate", "original_fps")
BETAS_KEYS = ("betas", "beta", "shape")
COORD_TRANSFORMS = {
    "none": None,
    "sonic_yup_to_gmr_zup": np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    ),
}


@dataclass(frozen=True)
class SonicMotion:
    source_path: Path
    poses: np.ndarray
    trans: np.ndarray
    betas: np.ndarray
    fps: float
    pose_key: str
    trans_key: str
    fps_key: str
    adjustments: tuple[str, ...]
    original_num_frames: int | None
    original_fps: float | None


def load_motion_file(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}

    if suffix in {".pkl", ".pickle", ".joblib"}:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError(
                "Reading Sonic pickle files requires joblib. Run this command in "
                "the gear_sonic_train environment or install joblib."
            ) from exc
        try:
            return joblib.load(path)
        except Exception:
            with path.open("rb") as f:
                return pickle.load(f)

    raise ValueError(f"Unsupported input file type: {path}")


def choose_key(data, candidates, preferred="auto", required=True):
    if preferred and preferred != "auto":
        if preferred in data:
            return preferred
        if required:
            raise KeyError(f"Requested key '{preferred}' not found. Available keys: {sorted(data)}")
        return None

    for key in candidates:
        if key in data:
            return key
    if required:
        raise KeyError(f"None of {candidates} found. Available keys: {sorted(data)}")
    return None


def as_numpy(value, name):
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object:
        array = np.asarray(array.tolist())
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got dtype {array.dtype}")
    return array


def _normalize_poses(value, adjustments):
    poses = as_numpy(value, "poses").astype(np.float32, copy=False)
    if poses.ndim == 3 and poses.shape[-1] == 3:
        poses = poses.reshape(poses.shape[0], -1)
        adjustments.append("pose_flattened")
    if poses.ndim != 2 or poses.shape[0] == 0:
        raise ValueError(f"poses must have non-empty shape (T, D) or (T, J, 3), got {poses.shape}")

    width = poses.shape[1]
    if width > 72:
        poses = poses[:, :72]
        adjustments.append(f"pose_truncated_{width}_to_72")
    elif width < 72:
        if width < 66:
            raise ValueError(f"poses must have at least 66 columns, got {poses.shape}")
        padded = np.zeros((poses.shape[0], 72), dtype=np.float32)
        padded[:, :width] = poses
        poses = padded
        adjustments.append(f"pose_padded_{width}_to_72")
    return np.ascontiguousarray(poses, dtype=np.float32)


def _normalize_trans(value, num_frames, resample_mismatched_trans, adjustments):
    trans = as_numpy(value, "trans").astype(np.float32, copy=False)
    if trans.ndim != 2 or trans.shape[0] == 0 or trans.shape[1] != 3:
        raise ValueError(f"trans must have non-empty shape (T, 3), got {trans.shape}")
    if trans.shape[0] == num_frames:
        return np.ascontiguousarray(trans, dtype=np.float32)
    if not resample_mismatched_trans:
        raise ValueError(f"pose/trans frame mismatch: {num_frames} pose frames vs {trans.shape[0]} trans frames")

    source_time = np.linspace(0.0, 1.0, trans.shape[0], dtype=np.float32)
    target_time = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    trans = np.stack(
        [np.interp(target_time, source_time, trans[:, axis]) for axis in range(3)],
        axis=1,
    ).astype(np.float32)
    adjustments.append(f"trans_resampled_{len(source_time)}_to_{num_frames}")
    return np.ascontiguousarray(trans, dtype=np.float32)


def _normalize_betas(data, override, num_betas):
    if override is not None:
        betas = np.asarray(override, dtype=np.float32).reshape(-1)
        if betas.size == 0:
            raise ValueError("betas override must contain at least one value")
        return np.ascontiguousarray(betas)

    key = choose_key(data, BETAS_KEYS, required=False)
    if key:
        betas = as_numpy(data[key], key).astype(np.float32, copy=False)
        if betas.ndim == 2:
            betas = betas[0]
        betas = betas.reshape(-1)
        if betas.size:
            return np.ascontiguousarray(betas)
    return np.zeros(num_betas, dtype=np.float32)


def _read_fps(data, pose_key, preferred, override, default):
    if override is not None:
        return float(override), "override"
    if pose_key == "original_pose_aa" and "original_fps" in data:
        return float(np.asarray(data["original_fps"]).reshape(-1)[0]), "original_fps"
    key = choose_key(data, FPS_KEYS, preferred, required=False)
    if key:
        return float(np.asarray(data[key]).reshape(-1)[0]), key
    if default is None:
        raise KeyError(f"None of {FPS_KEYS} found and no default fps was provided")
    return float(default), "default"


def load_sonic_motion(
    path,
    *,
    pose_key="auto",
    trans_key="auto",
    fps_key="auto",
    fps=None,
    default_fps=None,
    betas=None,
    num_betas=10,
    resample_mismatched_trans=False,
    require_finite=True,
):
    path = Path(path)
    data = load_motion_file(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict in {path}, got {type(data).__name__}")

    adjustments = []
    resolved_pose_key = choose_key(data, POSE_KEYS, pose_key)
    resolved_trans_key = choose_key(data, TRANS_KEYS, trans_key)
    poses = _normalize_poses(data[resolved_pose_key], adjustments)
    trans = _normalize_trans(
        data[resolved_trans_key],
        poses.shape[0],
        resample_mismatched_trans,
        adjustments,
    )
    resolved_fps, resolved_fps_key = _read_fps(data, resolved_pose_key, fps_key, fps, default_fps)
    if not math.isfinite(resolved_fps) or resolved_fps <= 0 or resolved_fps > 240:
        raise ValueError(f"fps must be finite and in (0, 240], got {resolved_fps}")
    if require_finite and (not np.isfinite(poses).all() or not np.isfinite(trans).all()):
        raise ValueError("poses and trans must contain only finite values")

    original_num_frames = None
    original_fps = None
    if "original_pose_aa" in data and "original_fps" in data:
        original_pose = np.asarray(data["original_pose_aa"])
        candidate_fps = float(np.asarray(data["original_fps"]).reshape(-1)[0])
        if original_pose.ndim >= 1 and math.isfinite(candidate_fps) and candidate_fps > 0:
            original_num_frames = int(original_pose.shape[0])
            original_fps = candidate_fps

    return SonicMotion(
        source_path=path,
        poses=poses,
        trans=trans,
        betas=_normalize_betas(data, betas, num_betas),
        fps=resolved_fps,
        pose_key=resolved_pose_key,
        trans_key=resolved_trans_key,
        fps_key=resolved_fps_key,
        adjustments=tuple(adjustments),
        original_num_frames=original_num_frames,
        original_fps=original_fps,
    )


def apply_coord_transform(poses, trans, transform_name):
    transform = COORD_TRANSFORMS[transform_name]
    if transform is None:
        return poses, trans
    root_transform = R.from_matrix(transform.astype(np.float64))
    root_rot = R.from_rotvec(poses[:, :3].astype(np.float64))
    transformed_poses = poses.copy()
    transformed_poses[:, :3] = (root_transform * root_rot).as_rotvec().astype(np.float32)
    transformed_trans = (trans @ transform.T).astype(np.float32)
    return np.ascontiguousarray(transformed_poses), np.ascontiguousarray(transformed_trans)


def to_gmr_smpl(motion, *, coord_transform="sonic_yup_to_gmr_zup", gender="neutral"):
    poses, trans = apply_coord_transform(motion.poses, motion.trans, coord_transform)
    return {
        "poses": poses,
        "betas": motion.betas,
        "trans": trans,
        "mocap_framerate": np.array(motion.fps, dtype=np.float32),
        "gender": np.array(gender),
        "source_file": np.array(str(motion.source_path)),
        "source_pose_key": np.array(motion.pose_key),
        "source_trans_key": np.array(motion.trans_key),
        "source_fps_key": np.array(motion.fps_key),
        "coord_transform": np.array(coord_transform),
        "normalization_adjustments": np.array(",".join(motion.adjustments)),
    }
