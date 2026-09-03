"""Geometry-agnostic WholeBody Omni V4, preserving the V3 solver core."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import os
import pickle
from time import perf_counter
import mujoco as mj
import mink
import numpy as np
from mink.tasks.task import Task
from scipy.spatial.transform import Rotation
from .wholebody_omni_gmr_v3 import WholeBodyOmniGMRV3
from .scene_limits import AutomaticSceneCollisionLimit
from .terrain_tasks import TerrainPointContactTask
from .motion_adapters import CanonicalMotion


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class RootSE3Task(Task):
    """Full pelvis position/orientation gauge for V4 (V3 remains yaw-only)."""

    def __init__(self, model: mj.MjModel, body_name: str, costs: list[float], gain: float = 0.45) -> None:
        self.model = model
        self.body_id = model.body(body_name).id
        self.target_position = np.zeros(3)
        self.target_rotation = np.eye(3)
        super().__init__(cost=np.asarray(costs, dtype=float), gain=float(gain), lm_damping=1.0)

    def set_target(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> None:
        self.target_position = np.asarray(position, dtype=float).reshape(3)
        quat = np.asarray(quaternion_wxyz, dtype=float).reshape(4)
        quat /= max(float(np.linalg.norm(quat)), 1e-12)
        self.target_rotation = Rotation.from_quat(quat, scalar_first=True).as_matrix()

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        current = configuration.data.xmat[self.body_id].reshape(3, 3)
        rotation_error = Rotation.from_matrix(self.target_rotation.T @ current).as_rotvec()
        return np.r_[configuration.data.xpos[self.body_id] - self.target_position, rotation_error]

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mj.mj_jacBody(self.model, configuration.data, jacp, jacr, self.body_id)
        # The residual is log(R_target.T @ R_current), expressed in the
        # target frame.  mj_jacBody returns world-frame angular velocity;
        # convert it through the SO(3) right-Jacobian inverse so the QP uses
        # the same tangent coordinates as the error.
        phi = Rotation.from_matrix(
            self.target_rotation.T @ configuration.data.xmat[self.body_id].reshape(3, 3)
        ).as_rotvec()
        theta = float(np.linalg.norm(phi))
        hat = np.array([[0.0, -phi[2], phi[1]], [phi[2], 0.0, -phi[0]], [-phi[1], phi[0], 0.0]])
        if theta < 1e-5:
            Jr_inv = np.eye(3) + 0.5 * hat + (hat @ hat) / 12.0
        else:
            half = 0.5 * theta
            coeff = 1.0 / (theta * theta) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
            Jr_inv = np.eye(3) + 0.5 * hat + coeff * (hat @ hat)
        jacr = Jr_inv @ self.target_rotation.T @ jacr
        return np.vstack((jacp, jacr))


class BoneDirectionTask(Task):
    """Soft normalized bone-direction constraints that remove limb mirror modes."""

    def __init__(self, model: mj.MjModel, robot_points: dict, edges: list[tuple[str, str]], cost: float, gain: float = 0.35) -> None:
        self.model = model
        self.robot_points = robot_points
        self.edges = [(a, b) for a, b in edges if a in robot_points and b in robot_points]
        self.sources: dict[str, np.ndarray] = {}
        super().__init__(cost=np.full(3 * len(self.edges), float(cost)), gain=float(gain), lm_damping=1.0)

    def set_source(self, source: dict[str, np.ndarray]) -> None:
        self.sources = source

    @staticmethod
    def _unit(value: np.ndarray) -> tuple[np.ndarray, float]:
        norm = float(np.linalg.norm(value))
        return value / max(norm, 1e-9), norm

    def _edge(self, configuration: mink.Configuration, a: str, b: str):
        pa = self.robot_points[a].point(configuration)
        pb = self.robot_points[b].point(configuration)
        ja = self.robot_points[a].jacobian(configuration)
        jb = self.robot_points[b].jacobian(configuration)
        current, length = self._unit(pb - pa)
        target, _ = self._unit(self.sources[b] - self.sources[a])
        projector = np.eye(3) - np.outer(current, current)
        return current, target, projector @ ((jb - ja) / max(length, 1e-9))

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        values = []
        for a, b in self.edges:
            current, target, _ = self._edge(configuration, a, b)
            values.append(current - target)
        return np.concatenate(values) if values else np.empty(0)

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        values = []
        for a, b in self.edges:
            values.append(self._edge(configuration, a, b)[2])
        return np.vstack(values) if values else np.empty((0, self.model.nv))


class LimbPlaneTask(Task):
    """Weak plane-normal task selecting the anatomical elbow/knee branch."""

    def __init__(self, model: mj.MjModel, robot_points: dict, triples: list[tuple[str, str, str]], cost: float, gain: float = 0.25) -> None:
        self.model = model
        self.robot_points = robot_points
        self.triples = [(a, b, c) for a, b, c in triples if a in robot_points and b in robot_points and c in robot_points]
        self.sources: dict[str, np.ndarray] = {}
        self._previous_normals: dict[tuple[str, str, str], np.ndarray] = {}
        self.confidence: dict[tuple[str, str, str], float] = {}
        super().__init__(cost=np.full(3 * len(self.triples), float(cost)), gain=float(gain), lm_damping=1.0)

    def set_source(self, source: dict[str, np.ndarray]) -> None:
        self.sources = source

    @staticmethod
    def _normal(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, float]:
        cross = np.cross(first, second)
        norm = float(np.linalg.norm(cross))
        return cross / max(norm, 1e-9), norm

    def _triple(self, configuration: mink.Configuration, names: tuple[str, str, str]):
        a, b, c = names
        pa, pb, pc = (self.robot_points[n].point(configuration) for n in names)
        ja, jb, jc = (self.robot_points[n].jacobian(configuration) for n in names)
        u, v = pb - pa, pc - pa
        normal, norm = self._normal(u, v)
        target, target_norm = self._normal(self.sources[b] - self.sources[a], self.sources[c] - self.sources[a])
        scale = max(float(np.linalg.norm(u)) * float(np.linalg.norm(v)), 1e-12)
        sin_angle = norm / scale
        # The plane is undefined for an almost straight limb.  Smoothly
        # fade it out and keep its normal sign continuous across frames.
        confidence = float(np.clip((sin_angle - 0.08) / 0.12, 0.0, 1.0))
        previous = self._previous_normals.get(names)
        if previous is not None and float(normal @ previous) < 0.0:
            normal = -normal
        if np.isfinite(normal).all() and norm > 1e-8:
            self._previous_normals[names] = normal.copy()
        self.confidence[names] = confidence if target_norm > 1e-8 else 0.0
        # d(u x v) = du x v + u x dv, then project through normalized cross.
        cross_j = np.cross(jb - ja, v[:, None], axis=0) + np.cross(u[:, None], jc - ja, axis=0)
        projector = (np.eye(3) - np.outer(normal, normal)) / max(norm, 1e-9)
        return normal, target, confidence * projector @ cross_j

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        values = []
        for triple in self.triples:
            normal, target, _ = self._triple(configuration, triple)
            values.append(self.confidence.get(triple, 0.0) * (normal - target))
        return np.concatenate(values) if values else np.empty(0)

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        return np.vstack([self._triple(configuration, triple)[2] for triple in self.triples]) if self.triples else np.empty((0, self.model.nv))


class WholeBodyOmniGMRV4(WholeBodyOmniGMRV3):
    """V3 with optional MuJoCo scene collision backend.

    The model must be a combined robot+scene model when ``scene_collision``
    backend is ``mujoco`` or ``hybrid`` and scene bodies use the configured
    prefix.  Analytic TerrainField remains active for floor/box regression.
    """
    def __init__(self, config_path: str | Path, terrain, environment_pool: np.ndarray, fps: float = 50.0, solver: str = "daqp") -> None:
        config_path = Path(config_path)
        raw = json.loads(config_path.read_text())
        if raw.get("extends"):
            base_path = config_path.parent / raw["extends"]
            base = json.loads(base_path.read_text())
            base = _deep_merge(base, {k: v for k, v in raw.items() if k != "extends"})
            merged = config_path.with_name(f".{config_path.stem}_merged.json")
            merged.write_text(json.dumps(base, indent=2))
            self._merged_config_path = merged
            config_path = merged
        else:
            self._merged_config_path = None
        super().__init__(config_path, terrain, environment_pool, fps=fps, solver=solver)
        # V4 consumes only canonical semantic names.  The adapters may retain
        # dataset aliases in their source frame for legacy contact code, but
        # no dataset-specific label is allowed to select a V4 target.
        source_aliases = {
            "Hips": "pelvis", "Spine1": "spine3", "Spine": "spine3",
            "LeftUpLeg": "left_hip", "RightUpLeg": "right_hip",
            "LeftLeg": "left_knee", "RightLeg": "right_knee",
            "LeftFoot": "left_foot", "RightFoot": "right_foot",
            "LeftToeBase": "left_toe", "RightToeBase": "right_toe",
            "LeftArm": "left_shoulder", "RightArm": "right_shoulder",
            "LeftForeArm": "left_elbow", "RightForeArm": "right_elbow",
            "LeftHandMiddle3": "left_wrist", "RightHandMiddle3": "right_wrist",
            "LeftHand": "left_wrist", "RightHand": "right_wrist",
        }
        for specification in self.semantic_mapping.values():
            specification["source"] = source_aliases.get(
                specification.get("source"), specification.get("source")
            )
        self.config["global_anchor"]["source"] = "pelvis"
        # V4 adds orientation/direction stabilization after the V3 objects
        # have been constructed; the original V3 class and entry remain intact.
        root_cfg = self.config.get("root_se3", {})
        if bool(root_cfg.get("enabled", True)):
            self.root_task = RootSE3Task(
                self.model,
                self.config["global_anchor"]["robot_body"],
                root_cfg.get("cost", [4.0, 4.0, 1.5, 1.0, 1.0, 1.0]),
                root_cfg.get("gain", 0.35),
            )
        # V4 may add generic body-surface channels without changing the V3
        # contact task.  Rebuild the task from the fully merged configuration
        # so inherited foot/palm/knee channels and costs remain intact.
        contact_cfg = self.config.get("contact_tasks", {})
        generic_points = contact_cfg.get("robot_points", {})
        if generic_points:
            self.contact_task = TerrainPointContactTask(
                self.model,
                generic_points,
                contact_cfg.get("normal_cost", 40.0),
                contact_cfg.get("tangent_cost", 12.0),
                contact_cfg.get("clearance", 0.004),
            )
        direction_cfg = self.config.get("bone_direction", {})
        default_edges = [
            ("pelvis", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_foot"),
            ("left_foot", "left_toe"), ("pelvis", "right_hip"), ("right_hip", "right_knee"),
            ("right_knee", "right_foot"), ("right_foot", "right_toe"),
            ("left_shoulder", "left_elbow"), ("left_elbow", "left_hand"),
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_hand"),
        ]
        edges = [tuple(edge) for edge in direction_cfg.get("edges", default_edges)]
        self.bone_direction_task = BoneDirectionTask(
            self.model, self.interaction_task.robot_points, edges,
            float(direction_cfg.get("cost", 1.5)), float(direction_cfg.get("gain", 0.3)),
        )
        plane_cfg = self.config.get("limb_plane", {})
        default_triples = [
            ("left_hip", "left_knee", "left_foot"),
            ("right_hip", "right_knee", "right_foot"),
            ("left_shoulder", "left_elbow", "left_hand"),
            ("right_shoulder", "right_elbow", "right_hand"),
        ]
        self.limb_plane_task = LimbPlaneTask(
            self.model, self.interaction_task.robot_points,
            [tuple(item) for item in plane_cfg.get("triples", default_triples)],
            float(plane_cfg.get("cost", 0.8)), float(plane_cfg.get("gain", 0.2)),
        )
        cfg = self.config.get("scene_collision", {})
        backend = str(cfg.get("backend", "analytic"))
        self.scene_collision = AutomaticSceneCollisionLimit(self.model, cfg) if backend in {"mujoco", "hybrid"} else None
        self.scene_backend = backend
        if self.scene_collision is not None:
            if not self.scene_collision.scene_geoms:
                raise ValueError(
                    f"V4 scene collision backend '{backend}' requires a combined "
                    "MuJoCo model containing collision-enabled scene bodies; "
                    f"none matched prefix {cfg.get('scene_body_prefix', 'scene_')!r}"
                )
            scene_geoms = set(self.scene_collision.scene_geoms)
            scene_bodies = {
                int(self.model.geom_bodyid[geom_id]) for geom_id in scene_geoms
            }
            # V3 discovers every mesh in its model. In a combined model the
            # scene meshes must not become robot terrain shells or self pairs.
            for name in list(self.terrain_limit.shells):
                if self.terrain_limit.shells[name]["body_id"] in scene_bodies:
                    self.terrain_limit.shells.pop(name)
                    self.terrain_limit.previous.pop(name, None)
                    self.terrain_limit.active.pop(name, None)
                    self.terrain_limit.release_count.pop(name, None)
            self.self_collision_limit.geom_pairs = [
                pair
                for pair in self.self_collision_limit.geom_pairs
                if pair[0] not in scene_geoms and pair[1] not in scene_geoms
            ]

    def __del__(self):
        path = getattr(self, "_merged_config_path", None)
        if path is not None:
            path.unlink(missing_ok=True)

    def retarget(
        self,
        source_frame: dict[str, np.ndarray],
        root_quaternion: np.ndarray,
        contact_frame: dict,
        chest_quaternion: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve one frame with scene collision inside every V4 QP pass."""
        if self.frame_index == 0:
            self._initialize(source_frame, root_quaternion)
        semantics = self._source_semantics(source_frame)
        self.interaction_task.set_source(semantics)
        self.bone_direction_task.set_source(semantics)
        self.limb_plane_task.set_source(semantics)
        root_name = self.config["global_anchor"]["source"]
        self.root_task.set_target(source_frame[root_name], root_quaternion)
        if chest_quaternion is not None:
            self.torso_task.set_target(
                self.configuration, root_quaternion, chest_quaternion
            )
        contacts = contact_frame.get("contacts", {})
        self.contact_task.set_contacts(self.configuration, contacts)
        self.foot_orientation_task.set_contacts(
            contacts, contact_frame.get("flat_foot", {})
        )
        if self.previous_q is not None:
            self.temporal_task.set_target(self.previous_q)

        solver_cfg = self.config["solver"]
        first = self.frame_index == 0
        passes = int(
            solver_cfg["first_frame_iterations"]
            if first
            else solver_cfg["iterations"]
        )
        tasks = [
            self.interaction_task,
            self.bone_direction_task,
            self.limb_plane_task,
            self.contact_task,
            self.foot_orientation_task,
            self.root_task,
            self.torso_task.task,
            self.nominal_task,
        ]
        if self.previous_q is not None:
            tasks.append(self.temporal_task)

        failures = []
        qp_runtime = 0.0
        scene_query_runtime = 0.0
        scene_pair_peak = 0
        for _ in range(passes):
            solve_dt = 1.0 if first else self.dt / passes
            self.terrain_limit.prepare_active_set(self.configuration, solve_dt)
            self.self_collision_limit.prepare_active_set(self.configuration)
            limits = [
                self.configuration_limit,
                self.terrain_limit,
                self.self_collision_limit,
                self.trust_limit,
            ]
            if self.scene_collision is not None:
                self.scene_collision.prepare_active_set(
                    self.configuration, solve_dt
                )
                scene_query_runtime += self.scene_collision.query_runtime_seconds
                scene_pair_peak = max(
                    scene_pair_peak, len(self.scene_collision.active_pairs)
                )
                limits.append(self.scene_collision)
            if not first:
                limits.append(self.velocity_limit)
            try:
                started = perf_counter()
                velocity = mink.solve_ik(
                    self.configuration,
                    tasks,
                    solve_dt,
                    self.solver,
                    self.damping,
                    limits=limits,
                )
                qp_runtime += perf_counter() - started
                self.configuration.integrate_inplace(velocity, solve_dt)
            except Exception as error:
                failures.append(f"{type(error).__name__}: {error}")
                break

        # Every metric below is recomputed after the final integration.
        self.terrain_limit.prepare_active_set(self.configuration, self.dt)
        self.self_collision_limit.prepare_active_set(self.configuration)
        minimum_slack = min(
            (
                item["slack"]
                for item in self.terrain_limit.measurements.values()
            ),
            default=np.inf,
        )
        scene_diagnostics: dict[str, Any] = {
            "active_scene_collision_pairs": 0,
            "minimum_scene_distance": float("inf"),
            "maximum_penetration": 0.0,
        }
        if self.scene_collision is not None:
            self.scene_collision.prepare_active_set(self.configuration, self.dt)
            scene_query_runtime += self.scene_collision.query_runtime_seconds
            scene_diagnostics.update(self.scene_collision.diagnostics())
            if bool(self.config.get("scene_collision", {}).get("debug_collision_jacobian", False)):
                scene_diagnostics["jacobian_finite_difference"] = self.scene_collision.finite_difference_check(self.configuration)
            scene_diagnostics.update(
                self.scene_collision.measure_current_distances(self.configuration)
            )
            scene_diagnostics["peak_active_scene_collision_pairs"] = int(
                scene_pair_peak
            )
            scene_diagnostics["scene_collision_query_runtime_seconds"] = float(
                scene_query_runtime
            )

        output = self.configuration.data.qpos.copy()
        delta = np.zeros(self.model.nv)
        if self.previous_q is not None:
            mj.mj_differentiatePos(
                self.model, delta, self.dt, self.previous_q, output
            )
        self.diagnostics.append(
            {
                "frame": self.frame_index,
                "passes": int(passes),
                "qp_failures": failures,
                "qp_failure": bool(failures),
                "qp_iterations": int(passes - len(failures)),
                "qp_solve_runtime_seconds": float(qp_runtime),
                "qp_solve_time": float(qp_runtime),
                "collision_query_time": float(scene_query_runtime),
                "active_collision_shells": sorted(self.terrain_limit.selected),
                "active_collision_points": int(
                    sum(len(value) for value in self.terrain_limit.selected.values())
                ),
                "active_self_collision_pairs": int(
                    len(self.self_collision_limit.active_pairs)
                ),
                "minimum_self_distance": float(
                    min(
                        (
                            item["distance"]
                            for item in self.self_collision_limit.active_pairs
                        ),
                        default=np.inf,
                    )
                ),
                "minimum_terrain_slack": float(minimum_slack),
                "min_slack_after": float(minimum_slack),
                "active_constraints": int(getattr(self.terrain_limit, "active_count", sum(len(v) for v in getattr(self.terrain_limit, "selected", {}).values())) + len(self.self_collision_limit.active_pairs) + len(getattr(self.scene_collision, "active_pairs", []))),
                "max_velocity": float(np.max(np.abs(delta), initial=0.0)),
                "interaction_error": float(np.linalg.norm(self.interaction_task.compute_error(self.configuration))),
                "torso_pelvis_targets": {
                    key: float(value)
                    for key, value in self.torso_task.targets.items()
                },
                "contact_states": {
                    name: {
                        "score": float(item.get("score", 0.0)),
                        "state": str(item.get("state", "NONE")),
                        "source_state": str(item.get("source_state", item.get("state", "NONE"))),
                        "object_id": str(item.get("object_id", "")),
                        "surface_id": str(item.get("surface_id", "")),
                        "signed_distance": float(item.get("signed_distance", np.inf)),
                        "normal_error": float(item.get("normal_error", 0.0)),
                        "tangent_error": float(item.get("tangent_error", item.get("tangential_speed", 0.0))),
                    }
                    for name, item in contacts.items()
                },
                **scene_diagnostics,
            }
        )
        # Keep a stable diagnostic vocabulary for downstream dataset tools.
        self.diagnostics[-1]["scene_collision_candidate_pairs"] = int(
            len(getattr(self.scene_collision, "robot_geoms", ())) *
            len(getattr(self.scene_collision, "scene_geoms", ()))
            if self.scene_collision is not None else 0
        )
        self.diagnostics[-1]["scene_collision_active_pairs"] = int(
            len(getattr(self.scene_collision, "active_pairs", ()))
        )
        self.previous_q = output.copy()
        self.frame_index += 1
        return output

    def retarget_canonical(
        self,
        motion: CanonicalMotion,
        solver_frames: list[dict[str, tuple[np.ndarray, np.ndarray]]],
        contact_schedule: list[dict],
        source_frames: list[dict[str, np.ndarray]] | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int = 0,
        resume: bool = False,
    ) -> np.ndarray:
        """Retarget canonical motion, optionally with resumable checkpoints."""
        if len(solver_frames) != motion.frame_count or len(contact_schedule) != motion.frame_count:
            raise ValueError("Canonical frame count, solver frames, and contact schedule must match")
        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
        outputs: list[np.ndarray] = []
        start = 0
        if resume and checkpoint is not None and checkpoint.is_file():
            with checkpoint.open("rb") as stream:
                state = pickle.load(stream)
            if state.get("frame_count") != motion.frame_count:
                raise ValueError("Checkpoint frame count does not match canonical motion")
            outputs = [np.asarray(value, dtype=float) for value in state.get("outputs", [])]
            start = int(state.get("next_frame", len(outputs)))
            if start != len(outputs) or start > motion.frame_count:
                raise ValueError("Checkpoint output sequence is inconsistent")
            if outputs:
                self.configuration.update(np.asarray(outputs[-1], dtype=float))
                self.previous_q = np.asarray(state.get("previous_q", outputs[-1]), dtype=float)
                self.frame_index = start
                self.diagnostics = list(state.get("diagnostics", []))
                for attr in ("selected", "previous", "active", "release_count"):
                    value = state.get(f"terrain_{attr}")
                    if value is not None and hasattr(self.terrain_limit, attr):
                        setattr(self.terrain_limit, attr, value)

        def save_checkpoint(next_frame: int) -> None:
            if checkpoint is None:
                return
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "frame_count": motion.frame_count,
                "next_frame": next_frame,
                "outputs": outputs,
                "previous_q": self.previous_q,
                "diagnostics": self.diagnostics,
                "terrain_selected": getattr(self.terrain_limit, "selected", None),
                "terrain_previous": getattr(self.terrain_limit, "previous", None),
                "terrain_active": getattr(self.terrain_limit, "active", None),
                "terrain_release_count": getattr(self.terrain_limit, "release_count", None),
            }
            temporary = checkpoint.with_name(f".{checkpoint.name}.tmp")
            with temporary.open("wb") as stream:
                pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, checkpoint)

        for index in range(start, motion.frame_count):
            frame, contacts = solver_frames[index], contact_schedule[index]
            root = frame.get("pelvis") or frame.get("root")
            chest = frame.get("spine3") or frame.get("chest")
            if root is None:
                raise ValueError("Canonical solver frames must contain pelvis/root")
            outputs.append(
                self.retarget(
                    (source_frames[index] if source_frames is not None else {name: value[0] for name, value in frame.items()}),
                    root[1],
                    contacts,
                    chest_quaternion=None if chest is None else chest[1],
                ).copy()
            )
            if checkpoint_every > 0 and (len(outputs) % checkpoint_every == 0):
                save_checkpoint(index + 1)
        if checkpoint is not None:
            save_checkpoint(motion.frame_count)
        return np.asarray(outputs, dtype=float)
