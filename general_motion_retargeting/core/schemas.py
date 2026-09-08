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
    NO_SOURCE_SCENE = "no_source_scene"
    PAIRED_SOURCE_SCENE = "paired_source_scene"
    TARGET_REPLACEMENT = "target_replacement"


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
        if len(t) != len(p) or len(t) == 0 or np.any(np.diff(t) < 0):
            raise ValueError("PoseTrajectory timestamps/translations are inconsistent")
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_anchor", np.asarray(self.source_anchor, dtype=float).reshape(3))
        object.__setattr__(self, "normal", np.asarray(self.normal, dtype=float).reshape(3))
        if self.end_frame < self.start_frame:
            raise ValueError("ContactEpisode end_frame precedes start_frame")


@dataclass(frozen=True)
class ContactPlan:
    episodes: tuple[ContactEpisode, ...]
    per_frame_states: tuple[dict[str, dict[str, Any]], ...]
    fps: float
    metadata: dict[str, Any] = field(default_factory=dict)


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
