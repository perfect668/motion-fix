"""Geometry-agnostic WholeBody Omni V4, preserving the V3 solver core."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import json
from time import perf_counter
import mujoco as mj
import mink
import numpy as np
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
                "qp_failures": failures,
                "qp_solve_runtime_seconds": float(qp_runtime),
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
                "max_velocity": float(np.max(np.abs(delta), initial=0.0)),
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
        self.previous_q = output.copy()
        self.frame_index += 1
        return output

    def retarget_canonical(
        self,
        motion: CanonicalMotion,
        solver_frames: list[dict[str, tuple[np.ndarray, np.ndarray]]],
        contact_schedule: list[dict],
        source_frames: list[dict[str, np.ndarray]] | None = None,
    ) -> np.ndarray:
        """Retarget a validated canonical motion without changing V4 IK semantics."""
        if len(solver_frames) != motion.frame_count or len(contact_schedule) != motion.frame_count:
            raise ValueError("Canonical frame count, solver frames, and contact schedule must match")
        outputs = []
        for index, (frame, contacts) in enumerate(zip(solver_frames, contact_schedule)):
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
        return np.asarray(outputs, dtype=float)
