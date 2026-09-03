"""Input adapters for the WholeBody retargeting pipelines.

The solver consumes :class:`CanonicalMotion`; dataset-specific details stay
here.  Adapters preserve source coordinates and metadata.  Scene transforms
are deliberately not applied in this module so that the same transform can be
applied to both motion and scene by the selected retargeting entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
import subprocess
import tempfile
from typing import Any

import numpy as np


class MotionFormatError(ValueError):
    """Raised when a motion file matches a format but fails its schema."""


@dataclass
class CanonicalMotion:
    """Dataset-independent motion representation in source-world coordinates."""

    positions: np.ndarray
    joint_names: list[str]
    fps: float
    orientations: np.ndarray | None = None  # quaternion order wxyz
    orientation_valid: bool = False
    orientation_valid_mask: np.ndarray | None = None
    root_name: str | None = None
    scene: dict[str, Any] = field(default_factory=dict)
    contacts: dict[str, Any] = field(default_factory=dict)
    source_format: str = "unknown"
    source_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # Adapter provenance is explicit so solver code never has to infer units
    # or scene ownership from a dataset-specific filename.
    human_height: float | None = None
    source_to_canonical: dict[str, Any] = field(default_factory=dict)
    scene_objects: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float64)
        if self.positions.ndim != 3 or self.positions.shape[-1] != 3:
            raise MotionFormatError(
                f"Canonical positions must have shape (T,J,3), got {self.positions.shape}"
            )
        if not np.isfinite(self.positions).all():
            raise MotionFormatError("Canonical positions contain NaN or Inf")
        self.joint_names = [str(name) for name in self.joint_names]
        if len(self.joint_names) != self.positions.shape[1]:
            raise MotionFormatError("joint_names count does not match motion point count")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise MotionFormatError("Canonical joint_names must be unique")
        self.fps = float(self.fps)
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise MotionFormatError(f"FPS must be finite and positive, got {self.fps}")
        if self.orientations is not None:
            self.orientations = np.asarray(self.orientations, dtype=np.float64)
            expected = (self.positions.shape[0], self.positions.shape[1], 4)
            if self.orientations.shape != expected:
                raise MotionFormatError(
                    f"orientations must have shape {expected}, got {self.orientations.shape}"
                )
            if not np.isfinite(self.orientations).all():
                raise MotionFormatError("Canonical orientations contain NaN or Inf")
        if self.orientation_valid_mask is None:
            self.orientation_valid_mask = np.full(
                (self.positions.shape[0], self.positions.shape[1]),
                bool(self.orientation_valid and self.orientations is not None),
                dtype=bool,
            )
        else:
            self.orientation_valid_mask = np.asarray(self.orientation_valid_mask, dtype=bool)
            expected_mask = self.positions.shape[:2]
            if self.orientation_valid_mask.shape != expected_mask:
                raise MotionFormatError(
                    f"orientation_valid_mask must have shape {expected_mask}, got {self.orientation_valid_mask.shape}"
                )
        if self.human_height is not None:
            self.human_height = float(self.human_height)
            if not np.isfinite(self.human_height) or self.human_height <= 0.0:
                raise MotionFormatError("human_height must be finite and positive")
        if not self.source_to_canonical:
            self.source_to_canonical = {name: name for name in self.joint_names}
        # A read-only alias used by generic scene-aware consumers.  Keep the
        # legacy ``scene`` field for V3/V4 compatibility.
        if not self.scene_objects and self.scene:
            self.scene_objects = [self.scene]

    @property
    def frame_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def joint_count(self) -> int:
        return int(self.positions.shape[1])

    def named_positions(self) -> list[dict[str, np.ndarray]]:
        return [
            {name: frame[index].copy() for index, name in enumerate(self.joint_names)}
            for frame in self.positions
        ]

    def canonical_named_positions(self) -> list[dict[str, np.ndarray]]:
        """Return the small, format-independent semantic skeleton.

        Adapters may retain extra source landmarks for contact inference, but
        this view has stable names and deterministic fallbacks.  The fallback
        points are marked in ``metadata['canonical_provenance']`` and never
        fabricate orientations.
        """
        frames = self.named_positions()
        aliases = {
            "pelvis": ("pelvis", "Hips", "hips", "Pelvis"),
            "spine3": ("spine3", "Spine1", "spine", "Spine"),
            "left_hip": ("left_hip", "LeftUpLeg"), "right_hip": ("right_hip", "RightUpLeg"),
            "left_knee": ("left_knee", "LeftLeg"), "right_knee": ("right_knee", "RightLeg"),
            "left_ankle": ("left_ankle", "LeftFoot"), "right_ankle": ("right_ankle", "RightFoot"),
            "left_foot": ("left_foot", "LeftFoot"), "right_foot": ("right_foot", "RightFoot"),
            "left_toe": ("left_toe", "LeftToeBase", "LeftToe"),
            "right_toe": ("right_toe", "RightToeBase", "RightToe"),
            "left_shoulder": ("left_shoulder", "LeftArm"), "right_shoulder": ("right_shoulder", "RightArm"),
            "left_elbow": ("left_elbow", "LeftForeArm"), "right_elbow": ("right_elbow", "RightForeArm"),
            "left_wrist": ("left_wrist", "LeftHandMiddle3", "LeftHand"),
            "right_wrist": ("right_wrist", "RightHandMiddle3", "RightHand"),
        }
        result = []
        provenance = []
        for frame in frames:
            semantic, frame_provenance = {}, {}
            for canonical, choices in aliases.items():
                value = next((frame[name] for name in choices if name in frame), None)
                if value is None:
                    continue
                semantic[canonical] = value.copy()
                frame_provenance[canonical] = next(name for name in choices if name in frame)
            for side in ("left", "right"):
                foot, toe = semantic.get(f"{side}_foot"), semantic.get(f"{side}_toe")
                ankle = semantic.get(f"{side}_ankle", foot)
                if foot is not None and toe is None:
                    # SMPL-X FK exposes a foot orientation but no ToeBase.
                    # Use its anatomical forward axis when available; the
                    # positional fallback is only used by position-only data.
                    direction = None
                    foot_name = next((name for name in aliases[f"{side}_foot"] if name in frame), None)
                    if self.orientations is not None and foot_name in self.joint_names:
                        from scipy.spatial.transform import Rotation
                        joint_index = self.joint_names.index(foot_name)
                        direction = Rotation.from_quat(
                            self.orientations[len(result), joint_index][[1, 2, 3, 0]]
                        ).apply([1.0, 0.0, 0.0])
                        direction = np.asarray(direction, dtype=float)
                    if direction is None or np.linalg.norm(direction) < 1e-8:
                        direction = foot - semantic.get(f"{side}_knee", foot)
                    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
                    semantic[f"{side}_toe"] = foot + 0.16 * direction
                    frame_provenance[f"{side}_toe"] = "foot_orientation_surface_proxy"
                if foot is not None and toe is not None:
                    semantic[f"{side}_heel"] = foot - 0.28 * (toe - foot)
                    frame_provenance[f"{side}_heel"] = "foot_to_toe_surface_proxy"
                if foot is not None and f"{side}_ankle" not in semantic:
                    semantic[f"{side}_ankle"] = ankle.copy()
                    frame_provenance[f"{side}_ankle"] = "foot_alias"
            result.append(semantic)
            provenance.append(frame_provenance)
        self.metadata["canonical_provenance"] = provenance
        return result


def _as_scalar(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise MotionFormatError(f"{name} must be scalar, got shape {array.shape}")
    return float(array.reshape(-1)[0])


def _continuous_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Remove q/-q representation flips independently for every joint."""
    result = np.asarray(quaternions, dtype=np.float64).copy()
    for frame in range(1, len(result)):
        dots = np.sum(result[frame - 1] * result[frame], axis=-1)
        result[frame][dots < 0.0] *= -1.0
    return result


def _resample_motion(motion: "CanonicalMotion", target_fps: float | None) -> "CanonicalMotion":
    """Resample every adapter to the solver clock (positions + world quats)."""
    if target_fps is None or np.isclose(float(target_fps), motion.fps) or motion.frame_count <= 1:
        return motion
    target_fps = float(target_fps)
    if target_fps <= 0.0:
        raise MotionFormatError(f"target_fps must be positive, got {target_fps}")
    count = max(1, int(np.floor((motion.frame_count - 1) * target_fps / motion.fps)) + 1)
    source_time = np.arange(motion.frame_count, dtype=float) / motion.fps
    target_time = np.arange(count, dtype=float) / target_fps
    from scipy.interpolate import interp1d
    positions = np.asarray(interp1d(source_time, motion.positions, axis=0, kind="linear")(target_time))
    orientations = None
    if motion.orientations is not None:
        from scipy.spatial.transform import Rotation, Slerp
        orientations = np.empty((count, motion.joint_count, 4), dtype=float)
        for joint in range(motion.joint_count):
            rotations = Rotation.from_quat(motion.orientations[:, joint][:, [1, 2, 3, 0]])
            orientations[:, joint] = Slerp(source_time, rotations)(target_time).as_quat(scalar_first=True)
        orientations = _continuous_quaternions(orientations)
    if motion.orientation_valid_mask is not None:
        mask_source = motion.orientation_valid_mask.astype(float)
        mask = interp1d(source_time, mask_source, axis=0, kind="nearest")(target_time) > 0.5
    else:
        mask = None
    motion.positions = positions
    motion.orientations = orientations
    motion.orientation_valid_mask = mask
    motion.fps = target_fps
    motion.metadata["resampled_from_fps"] = float(source_time.size - 1) / max(source_time[-1], 1e-12) if len(source_time) > 1 else float(motion.fps)
    return motion


def _load_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as stream:
            return pickle.load(stream)
    except ModuleNotFoundError as error:
        # Some GRAIL files were serialized with numpy._core module names.
        if "numpy._core" not in str(error):
            raise
        import numpy.core
        import numpy.core.numeric

        import sys

        sys.modules.setdefault("numpy._core", numpy.core)
        sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
        with path.open("rb") as stream:
            return pickle.load(stream)


def detect_motion_format(path: str | Path) -> str:
    """Inspect extension and content, returning a stable adapter name."""

    source = Path(path).expanduser()
    suffix = source.suffix.lower()
    if suffix == ".npy":
        array = np.load(source, mmap_mode="r")
        if array.ndim == 3 and array.shape[-1] == 3:
            return "holosoma_global_positions"
        raise MotionFormatError(f"Unsupported .npy motion shape: {array.shape}")
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as data:
            keys = set(data.files)
            if {"global_joint_positions", "joint_positions"} & keys:
                return "holosoma_global_positions"
            if ("poses" in keys or {"root_orient", "pose_body"} <= keys) and "trans" in keys:
                return "smplx_npz"
        raise MotionFormatError(f"Cannot identify NPZ motion schema; keys={sorted(keys)}")
    if suffix == ".pkl":
        value = _load_pickle(source)
        if isinstance(value, dict) and isinstance(value.get("human_data"), dict):
            return "grail_smplx_recon"
        if isinstance(value, dict) and {"root_pos", "root_rot", "dof_pos"} <= set(value):
            return "robot_motion"
        raise MotionFormatError("PKL is neither a GRAIL reconstruction nor robot motion")
    if suffix == ".bvh":
        return "bvh"
    if suffix == ".fbx":
        # Some legacy datasets store an ASCII BVH hierarchy with an .fbx
        # suffix.  Inspect the header before selecting a parser.
        with source.open("rb") as stream:
            header = stream.read(256)
        if b"HIERARCHY" in header:
            return "bvh"
        return "fbx"
    raise MotionFormatError(f"Unsupported motion extension: {suffix or '<none>'}")


def _load_holosoma(path: Path, joint_map: str | Path, default_fps: float) -> CanonicalMotion:
    from .holosoma_input import load_holosoma_positions, load_joint_map

    mapping = load_joint_map(joint_map)
    positions, names, fps = load_holosoma_positions(path, mapping, default_fps=default_fps)
    source_format = "holosoma_global_positions"
    orientations = None
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "source_format" in data.files:
                source_format = str(np.asarray(data["source_format"]).item())
            if "global_joint_orientations" in data.files:
                orientations = _continuous_quaternions(np.asarray(data["global_joint_orientations"], dtype=float))
    return CanonicalMotion(
        positions=positions,
        joint_names=names,
        fps=fps,
        orientations=orientations,
        orientation_valid=orientations is not None,
        root_name="Hips",
        source_format=source_format,
        source_path=str(path),
        metadata={"joint_map": str(joint_map)},
    )


def _smplx_arrays(human: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    if "poses" in human:
        poses = np.asarray(human["poses"], dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] < 66:
            raise MotionFormatError(f"SMPL-X poses must have shape (T,>=66), got {poses.shape}")
        root_orient, pose_body = poses[:, :3], poses[:, 3:66]
    elif {"root_orient", "pose_body"} <= set(human):
        root_orient = np.asarray(human["root_orient"], dtype=np.float32)
        pose_body = np.asarray(human["pose_body"], dtype=np.float32)
    else:
        raise MotionFormatError("SMPL-X input requires poses or root_orient+pose_body")
    trans = np.asarray(human.get("trans"), dtype=np.float32)
    if trans.ndim != 2 or trans.shape != (len(root_orient), 3):
        raise MotionFormatError(f"SMPL-X trans must have shape {(len(root_orient), 3)}, got {trans.shape}")
    betas = np.asarray(human.get("betas", np.zeros(10)), dtype=np.float32).reshape(-1)[:10]
    if len(betas) == 0:
        betas = np.zeros(10, dtype=np.float32)
    gender = str(human.get("gender", "neutral"))
    fps = _as_scalar(human.get("mocap_frame_rate", 60.0), "mocap_frame_rate")
    return root_orient, pose_body, trans, gender, fps


def _load_smplx(path: Path, human: dict[str, Any], body_models: str | Path, target_fps: float | None, source_format: str, scene: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> CanonicalMotion:
    from .utils.smpl import get_smplx_data_offline_fast, load_smplx_file

    root_orient, pose_body, trans, gender, source_fps = _smplx_arrays(human)
    temporary = path.with_name(f".{path.stem}_canonical_smplx.npz")
    np.savez(
        temporary,
        root_orient=root_orient,
        pose_body=pose_body,
        trans=trans,
        betas=np.asarray(human.get("betas", np.zeros(10)), dtype=np.float32).reshape(-1)[:10],
        gender=np.asarray(gender),
        mocap_frame_rate=np.asarray(source_fps),
    )
    try:
        data, model, output, human_height = load_smplx_file(temporary, Path(body_models))
        frames, fps = get_smplx_data_offline_fast(data, model, output, tgt_fps=target_fps or source_fps)
    finally:
        temporary.unlink(missing_ok=True)
    if not frames:
        raise MotionFormatError(f"SMPL-X input contains no frames: {path}")
    names = list(frames[0].keys())
    positions = np.asarray([[np.asarray(frame[name][0], dtype=float) for name in names] for frame in frames])
    orientations = _continuous_quaternions(np.asarray([[np.asarray(frame[name][1], dtype=float) for name in names] for frame in frames]))
    return CanonicalMotion(
        positions=positions,
        joint_names=names,
        orientations=orientations,
        orientation_valid=True,
        fps=float(fps),
        root_name="pelvis",
        scene=dict(scene or {}),
        source_format=source_format,
        source_path=str(path),
        metadata=dict(metadata or {}),
        human_height=float(human_height),
        source_to_canonical={name: name for name in names},
        scene_objects=list((scene or {}).get("objects", [])) if isinstance(scene, dict) else [],
    )


_BVH_PROFILES = {
    # LAFAN1 and the bundled samples are centimetres, Y-up, right-handed.
    "lafan1": {"unit_scale": 0.01, "axis_matrix": [[1, 0, 0], [0, 0, -1], [0, 1, 0]]},
    "nokov": {"unit_scale": 0.001, "axis_matrix": [[1, 0, 0], [0, 0, -1], [0, 1, 0]]},
    "xsens": {"unit_scale": 0.001, "axis_matrix": [[1, 0, 0], [0, 0, -1], [0, 1, 0]]},
}


def _load_bvh(path: Path, bvh_format: str, fps: float | None) -> CanonicalMotion:
    from scipy.spatial.transform import Rotation as R
    from .utils.lafan_vendor.extract import read_bvh
    from .utils.lafan_vendor import utils

    data = read_bvh(path)
    global_rot, global_pos = utils.quat_fk(data.quats, data.pos, data.parents)
    profile = _BVH_PROFILES.get(str(bvh_format).lower(), _BVH_PROFILES["lafan1"])
    rotation_matrix = np.asarray(profile["axis_matrix"], dtype=float)
    unit_scale = float(profile["unit_scale"])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    names = list(data.bones)
    toe_name = "LeftToe" if "LeftToe" in names else "LeftToeBase" if "LeftToeBase" in names else None
    right_toe_name = "RightToe" if "RightToe" in names else "RightToeBase" if "RightToeBase" in names else None
    if toe_name is None or right_toe_name is None:
        raise MotionFormatError("BVH must contain LeftToe/LeftToeBase and RightToe/RightToeBase")
    frames = []
    for frame_index in range(data.pos.shape[0]):
        result = {}
        for joint_index, bone in enumerate(names):
            orientation = utils.quat_mul(rotation_quat, global_rot[frame_index, joint_index])
            position = global_pos[frame_index, joint_index] @ rotation_matrix.T * unit_scale
            result[bone] = [position, orientation]
        result.setdefault("LeftFootMod", [result["LeftFoot"][0], result[toe_name][1]])
        result.setdefault("RightFootMod", [result["RightFoot"][0], result[right_toe_name][1]])
        frames.append(result)
    if not frames:
        raise MotionFormatError(f"BVH input contains no frames: {path}")
    names = list(frames[0])
    positions = np.asarray([[np.asarray(frame[name][0], dtype=float) for name in names] for frame in frames])
    orientations = _continuous_quaternions(np.asarray([[np.asarray(frame[name][1], dtype=float) for name in names] for frame in frames]))
    return CanonicalMotion(
        positions=positions,
        joint_names=names,
        orientations=orientations,
        orientation_valid=True,
        fps=float(data.fps if fps is None else fps),
        root_name="Hips" if "Hips" in names else names[0],
        source_format=f"bvh_{bvh_format}",
        source_path=str(path),
        metadata={"bvh_format": bvh_format, "unit_scale": unit_scale,
                  "axis_matrix": rotation_matrix.tolist(), "frametime": float(data.frametime)},
    )


def _load_fbx_pickle(path: Path, fps: float) -> CanonicalMotion:
    """Load the legacy OptiTrack FBX-pickle representation used by GMR."""
    value = _load_pickle(path)
    if not isinstance(value, (list, tuple)) or not value or not isinstance(value[0], dict):
        raise MotionFormatError(
            "FBX adapter expects the legacy pickled list of {joint: (position, quaternion)} frames"
        )
    names = list(value[0])
    if any(set(frame) != set(names) for frame in value):
        raise MotionFormatError("FBX frames do not have a stable joint-name set")
    positions = np.asarray([[np.asarray(frame[name][0], dtype=float) for name in names] for frame in value])
    orientations = _continuous_quaternions(np.asarray([[np.asarray(frame[name][1], dtype=float) for name in names] for frame in value]))
    return CanonicalMotion(
        positions=positions,
        joint_names=names,
        orientations=orientations,
        orientation_valid=True,
        fps=fps,
        root_name="Hips" if "Hips" in names else names[0],
        source_format="fbx_optitrack_pickle",
        source_path=str(path),
    )


def load_canonical_motion(
    path: str | Path,
    *,
    joint_map: str | Path | None = None,
    body_models: str | Path = "assets/body_models",
    target_fps: float | None = None,
    default_holosoma_fps: float = 120.0,
    bvh_format: str = "lafan1",
    default_bvh_fps: float = 30.0,
    default_fbx_fps: float = 60.0,
) -> CanonicalMotion:
    """Load a supported motion into :class:`CanonicalMotion`.

    ``target_fps`` only resamples SMPL-X through the existing FK loader.  The
    position-only HoloSoMo path is left untouched here so its caller can use
    the existing interpolation policy explicitly.
    """

    source = Path(path).expanduser().resolve()
    kind = detect_motion_format(source)
    if kind == "holosoma_global_positions":
        if joint_map is None and source.suffix.lower() == ".npz":
            with np.load(source, allow_pickle=False) as data:
                if "joint_names" in data.files:
                    positions = data["global_joint_positions"] if "global_joint_positions" in data.files else data["joint_positions"]
                    names = [str(item) for item in np.asarray(data["joint_names"]).tolist()]
                    fps = _as_scalar(data["fps"], "fps") if "fps" in data.files else default_holosoma_fps
                    source_format = str(np.asarray(data["source_format"]).item()) if "source_format" in data.files else kind
                    orientations = None
                    if "global_joint_orientations" in data.files:
                        orientations = _continuous_quaternions(np.asarray(data["global_joint_orientations"], dtype=float))
                    return _resample_motion(CanonicalMotion(
                        positions, names, fps=fps, orientations=orientations,
                        orientation_valid=orientations is not None,
                        source_format=source_format, source_path=str(source),
                        metadata={"source_unit": str(np.asarray(data["source_unit"]).item()) if "source_unit" in data.files else "unknown"},
                    ), target_fps)
        if joint_map is None:
            raise MotionFormatError("HoloSoMo input requires an explicit joint_map")
        return _resample_motion(_load_holosoma(source, joint_map, default_holosoma_fps), target_fps)
    if kind == "smplx_npz":
        with np.load(source, allow_pickle=False) as data:
            human = {key: data[key] for key in data.files}
        return _load_smplx(source, human, body_models, target_fps, kind)
    if kind == "grail_smplx_recon":
        record = _load_pickle(source)
        human = record["human_data"]
        scene = {
            "object_path": record.get("object_path", ""),
            "obj_data": record.get("obj_data", {}),
            "meta": record.get("meta", {}),
        }
        return _load_smplx(
            source,
            human,
            body_models,
            target_fps,
            kind,
            scene=scene,
            metadata={"record_keys": sorted(record.keys())},
        )
    if kind == "bvh":
        return _resample_motion(_load_bvh(source, bvh_format, None), target_fps)
    if kind == "fbx":
        if source.suffix.lower() == ".fbx":
            # Binary FBX is parsed by Blender, then enters the same canonical
            # loader as every other format.  The adapter remains deterministic
            # and records the exporter metadata in the resulting motion.
            exporter = Path(__file__).resolve().parents[1] / "scripts" / "fbx_to_canonical_npz.py"
            with tempfile.TemporaryDirectory(prefix="gmr_fbx_") as directory:
                output = Path(directory) / "canonical.npz"
                expression = (
                    "import sys; sys.argv=['fbx_to_canonical_npz.py','--input',%r,'--output',%r]; "
                    "exec(compile(open(%r).read(),%r,'exec'))"
                    % (str(source), str(output), str(exporter), str(exporter))
                )
                try:
                    subprocess.run(["blender", "-b", "--python-expr", expression],
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                except (OSError, subprocess.CalledProcessError) as error:
                    raise MotionFormatError(f"Binary FBX conversion failed for {source}: {error}") from error
                with np.load(output, allow_pickle=False) as data:
                    positions = np.asarray(data["global_joint_positions"], dtype=float)
                    orientations = _continuous_quaternions(np.asarray(data["global_joint_orientations"], dtype=float))
                    names = [str(item) for item in np.asarray(data["joint_names"]).tolist()]
                    fps = _as_scalar(data["fps"], "fps")
                    metadata = {"source_unit": str(np.asarray(data["source_unit"]).item()),
                                "coordinate_transform": str(np.asarray(data["coordinate_transform"]).item())}
                return _resample_motion(CanonicalMotion(
                    positions, names, fps=fps, orientations=orientations,
                    orientation_valid=True, root_name="Hips",
                    source_format="fbx_binary_blender_global_positions_meters_zup",
                    source_path=str(source), metadata=metadata), target_fps)
        return _resample_motion(_load_fbx_pickle(source, default_fbx_fps), target_fps)
    raise MotionFormatError(
        f"Detected {kind!r}, but no CanonicalMotion adapter is registered yet. "
        "Use the existing BVH/FBX entry point or add a format adapter explicitly."
    )
