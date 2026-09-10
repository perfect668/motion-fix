"""Explicit, dataset-independent inputs and outputs for WholeBody V5.

These schemas deliberately contain references and metadata rather than loader
implementation details.  A resolver creates one :class:`RetargetTask`; the
solver never needs to inspect a filename or dataset name again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class SceneRelation(str, Enum):
    MOTION_ONLY = "motion_only"
    SHARED_SCENE = "shared_scene"
    SOURCE_TO_TARGET = "source_to_target"
    TARGET_ONLY_WITH_EXPLICIT_BINDING = "target_only_with_explicit_binding"
    # Legacy names remain aliases so V3/V4-facing callers do not break, while
    # V5 manifests and exports use the explicit relation vocabulary above.
    NO_SOURCE_SCENE = "motion_only"
    PAIRED_SOURCE_SCENE = "shared_scene"
    TARGET_REPLACEMENT = "source_to_target"


class FloorPolicy(str, Enum):
    AUTO = "auto"
    EXPLICIT = "explicit"
    DISABLED = "disabled"


@dataclass(frozen=True)
class MotionReference:
    path: Path
    format: str = "auto"
    fps: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())


@dataclass(frozen=True)
class TransformPlan:
    """Named source/canonical/solver frame transform with provenance."""

    source_frame: str
    target_frame: str
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4))
    unit_scale: float = 1.0
    applied: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float).reshape(4, 4)
        if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError("TransformPlan.matrix must be finite homogeneous")
        if not np.isfinite(self.unit_scale) or self.unit_scale <= 0:
            raise ValueError("TransformPlan.unit_scale must be positive")
        object.__setattr__(self, "matrix", matrix)


@dataclass(frozen=True)
class FrameGraph:
    """The named coordinate frames used by one retarget task."""

    source_motion: str = "source_motion"
    source_scene: str = "source_scene"
    canonical: str = "canonical"
    solver_world: str = "solver_world"
    mujoco_world: str = "mujoco_world"
    asset_local: str = "asset_local"
    robot_link: str = "robot_link"


@dataclass(frozen=True)
class RetargetJob:
    """Canonical task description produced by CLI, manifest or Python API."""

    motion: MotionReference
    source_scene: "SceneReference | None" = None
    target_scene: "SceneReference | None" = None
    scene_relation: SceneRelation = SceneRelation.MOTION_ONLY
    robot: str = "ne01"
    transform_policy: dict[str, Any] = field(default_factory=dict)
    morphology_policy: dict[str, Any] = field(default_factory=dict)
    contact_policy: dict[str, Any] = field(default_factory=dict)
    solver_policy: dict[str, Any] = field(default_factory=dict)
    output_policy: dict[str, Any] = field(default_factory=dict)
    time_range: tuple[int | None, int | None] = (None, None)

    def __post_init__(self) -> None:
        if self.robot.lower() != "ne01":
            raise ValueError(f"Unsupported V5 robot profile: {self.robot}")
        start, end = self.time_range
        if start is not None and start < 0:
            raise ValueError("time_range start must be non-negative")
        if start is not None and end is not None and end < start:
            raise ValueError("time_range end precedes start")


@dataclass(frozen=True)
class SceneReference:
    path: Path | None = None
    format: str = "auto"
    asset_id: str = "scene"
    pose: np.ndarray | None = None
    pose_trajectory: "PoseTrajectory | None" = None
    unit_scale: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        if self.pose is not None:
            pose = np.asarray(self.pose, dtype=float).reshape(4, 4)
            if not np.all(np.isfinite(pose)):
                raise ValueError("SceneReference.pose contains NaN or Inf")
            object.__setattr__(self, "pose", pose)


@dataclass(frozen=True)
class PoseTrajectory:
    timestamps: np.ndarray
    translations: np.ndarray
    rotations_wxyz: np.ndarray | None = None

    def __post_init__(self) -> None:
        t = np.asarray(self.timestamps, dtype=float).reshape(-1)
        p = np.asarray(self.translations, dtype=float).reshape((-1, 3))
        if len(t) != len(p) or len(t) == 0 or (len(t) > 1 and np.any(np.diff(t) <= 0)):
            raise ValueError("PoseTrajectory timestamps/translations must be strictly increasing")
        if not np.isfinite(t).all() or not np.isfinite(p).all():
            raise ValueError("PoseTrajectory contains NaN or Inf")
        object.__setattr__(self, "timestamps", t)
        object.__setattr__(self, "translations", p)
        if self.rotations_wxyz is not None:
            q = np.asarray(self.rotations_wxyz, dtype=float).reshape((-1, 4))
            if q.shape[0] != len(t) or not np.isfinite(q).all():
                raise ValueError("PoseTrajectory rotations are inconsistent")
            q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
            object.__setattr__(self, "rotations_wxyz", q)

    @property
    def frame_count(self) -> int:
        return int(len(self.timestamps))


@dataclass(frozen=True)
class SceneAsset:
    asset_id: str
    path: Path
    static: bool = True
    collision_enabled: bool = True
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    pose_trajectory: PoseTrajectory | None = None
    unit_scale: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        pose = np.asarray(self.pose, dtype=float).reshape(4, 4)
        if not np.all(np.isfinite(pose)):
            raise ValueError(f"SceneAsset {self.asset_id} pose contains NaN/Inf")
        object.__setattr__(self, "pose", pose)

    @property
    def asset_space(self) -> str:
        return str(self.metadata.get("asset_space", "asset_local"))

    @property
    def asset_scale_baked(self) -> bool:
        return bool(self.metadata.get("asset_scale_baked", False))

    def pose_at(self, timestamp: float) -> np.ndarray:
        """Return the asset pose on its explicit trajectory, if present."""
        if self.pose_trajectory is None:
            return self.pose.copy()
        trajectory = self.pose_trajectory
        time_value = float(timestamp)
        index = int(np.searchsorted(trajectory.timestamps, time_value, side="left"))
        if index <= 0:
            translation = trajectory.translations[0]
            rotation = trajectory.rotations_wxyz[0] if trajectory.rotations_wxyz is not None else None
        elif index >= trajectory.frame_count:
            translation = trajectory.translations[-1]
            rotation = trajectory.rotations_wxyz[-1] if trajectory.rotations_wxyz is not None else None
        else:
            left, right = index - 1, index
            alpha = (time_value - trajectory.timestamps[left]) / (trajectory.timestamps[right] - trajectory.timestamps[left])
            translation = (1.0 - alpha) * trajectory.translations[left] + alpha * trajectory.translations[right]
            rotation = None
            if trajectory.rotations_wxyz is not None:
                from scipy.spatial.transform import Rotation, Slerp
                rotation = Slerp(trajectory.timestamps[[left, right]], Rotation.from_quat(
                    trajectory.rotations_wxyz[[left, right]][:, [1, 2, 3, 0]]
                ))([time_value]).as_quat(scalar_first=True)[0]
        result = np.eye(4)
        result[:3, 3] = translation
        # A pose trajectory stores translation/orientation samples while the
        # asset's metric scale belongs to the owning SceneAsset pose.  Keep
        # that scale on every interpolated pose; otherwise dynamic query
        # geometry silently becomes unit-scale after frame 0 while MuJoCo
        # collision meshes retain the baked asset scale.
        base_linear = np.asarray(self.pose[:3, :3], dtype=float)
        base_scale = np.linalg.norm(base_linear, axis=0)
        base_scale = np.where(base_scale > 1e-12, base_scale, 1.0)
        if rotation is not None:
            from scipy.spatial.transform import Rotation
            result[:3, :3] = Rotation.from_quat(np.asarray(rotation)[[1, 2, 3, 0]]).as_matrix() @ np.diag(base_scale)
        else:
            result[:3, :3] = self.pose[:3, :3]
        return result


@dataclass(frozen=True)
class SceneModel:
    assets: tuple[SceneAsset, ...] = ()
    floor_policy: FloorPolicy = FloorPolicy.AUTO
    floor_height: float | None = 0.0
    world_up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    transform: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        up = np.asarray(self.world_up, dtype=float).reshape(3)
        up /= max(float(np.linalg.norm(up)), 1e-12)
        if up[2] < 0.99:
            raise ValueError("WholeBody V5 requires a Z-up solver scene")
        object.__setattr__(self, "world_up", up)
        ids = [asset.asset_id for asset in self.assets]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate SceneModel asset ids: {ids}")


@dataclass(frozen=True)
class ResolvedBundle:
    motion: MotionReference
    source_scene: SceneReference | None
    target_scene: SceneReference | None
    scene_relation: SceneRelation
    shared_coordinate_frame: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContactEpisode:
    body_channel: str
    phase: str
    start_frame: int
    end_frame: int
    object_id: str
    surface_id: str
    source_anchor: np.ndarray
    normal: np.ndarray
    triangle_id: int | None = None
    barycentric: np.ndarray | None = None
    asset_local_anchor: np.ndarray | None = None
    confidence: float = 0.0
    source_provenance: str = "unknown"
    target_provenance: str = "unbound"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_anchor", np.asarray(self.source_anchor, dtype=float).reshape(3))
        object.__setattr__(self, "normal", np.asarray(self.normal, dtype=float).reshape(3))
        if self.barycentric is not None:
            barycentric = np.asarray(self.barycentric, dtype=float).reshape(3)
            if not np.isfinite(barycentric).all():
                raise ValueError("ContactEpisode barycentric contains NaN/Inf")
            object.__setattr__(self, "barycentric", barycentric)
        if self.asset_local_anchor is not None:
            object.__setattr__(self, "asset_local_anchor", np.asarray(self.asset_local_anchor, dtype=float).reshape(3))
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("ContactEpisode confidence must be in [0,1]")
        if self.end_frame < self.start_frame:
            raise ValueError("ContactEpisode end_frame precedes start_frame")


@dataclass(frozen=True)
class ContactPlan:
    episodes: tuple[ContactEpisode, ...]
    per_frame_states: tuple[dict[str, dict[str, Any]], ...]
    fps: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.fps)) or float(self.fps) <= 0.0:
            raise ValueError("ContactPlan fps must be finite and positive")
        object.__setattr__(self, "episodes", tuple(self.episodes))
        object.__setattr__(self, "per_frame_states", tuple(self.per_frame_states))


@dataclass(frozen=True)
class MorphologyTargets:
    """Human-to-robot chain mapping kept separate from scene transforms."""

    human_height: float | None
    robot_height: float
    chain_scales: dict[str, float] = field(default_factory=dict)
    mode: str = "scene_preserving"
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneBundle:
    source: SceneModel | None
    target: SceneModel | None
    relation: SceneRelation
    frame_graph: FrameGraph = field(default_factory=FrameGraph)
    transform: TransformPlan | None = None


@dataclass(frozen=True)
class RobotModelSpec:
    name: str
    mjcf_path: Path
    semantic_points: dict[str, dict[str, Any]]
    joint_limits: dict[str, Any] = field(default_factory=dict)
    velocity_limits: dict[str, Any] = field(default_factory=dict)
    nominal_qpos: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mjcf_path", Path(self.mjcf_path).expanduser().resolve())
        if self.nominal_qpos is not None:
            object.__setattr__(self, "nominal_qpos", np.asarray(self.nominal_qpos, dtype=float).reshape(-1))


@dataclass(frozen=True)
class RetargetTask:
    bundle: ResolvedBundle
    scene: SceneModel
    config_path: Path
    robot: RobotModelSpec
    solver_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_path", Path(self.config_path).expanduser().resolve())


@dataclass
class RetargetResult:
    qpos: np.ndarray
    fps: float
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    contact_plan: ContactPlan | None = None
    scene_metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "UNKNOWN"
    failure: str | None = None


# CanonicalMotion is implemented by the adapter package to keep source-format
# loading out of the core schema module. Re-export it lazily here so callers
# have one documented contract location without introducing an import cycle.
def __getattr__(name: str):
    if name == "CanonicalMotion":
        from ..motion_adapters import CanonicalMotion
        return CanonicalMotion
    raise AttributeError(name)
