"""Terrain-aware extension of WholeBodyOmniGMRV2.

The flat-ground implementation remains unchanged and is used as a regression
baseline.  This class reuses its GMR FrameTasks, semantic Laplacian, posture
prior, joint/velocity limits, and stabilization while replacing only the
ground-specific contact and non-penetration pieces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mink
import numpy as np

from .terrain_geometry import SceneTransform, TerrainField
from .terrain_limits import TerrainNonPenetrationLimit
from .terrain_tasks import TerrainFootOrientationTask, TerrainFootTemporalTask, TerrainPointContactTask
from .wholebody_omni_gmr_v2 import WholeBodyOmniGMRV2


class WholeBodyTerrainOmniGMRV2(WholeBodyOmniGMRV2):
    def __init__(
        self,
        *args,
        terrain_spec: TerrainField | dict | str | Path,
        joint_map: dict,
        scene_transform_config: SceneTransform | dict,
        terrain_config_path: str | Path,
        **kwargs,
    ) -> None:
        terrain_config_path = Path(terrain_config_path)
        self.terrain_config = json.loads(terrain_config_path.read_text())
        base_config = Path(self.terrain_config["base_config"])
        if not base_config.is_absolute():
            base_config = terrain_config_path.parent / base_config
        kwargs["graph_config_path"] = str(base_config)
        if kwargs.get("tgt_robot") in (None, "ne01"):
            kwargs["tgt_robot"] = "ne01_desktop_assets_wholebody_omni_gmr_v2"
        super().__init__(*args, **kwargs)

        self.joint_map = joint_map
        # HoloSoMo position mocap is already expressed in the robot-style
        # (+X forward, +Y left, +Z up) world frame.  The legacy SMPL-X config
        # contains frame rotations and a root-origin offset calibrated for
        # SMPL-X body axes; applying those to HoloSoMo rotates the complete
        # posture.  Terrain input therefore uses an explicit identity frame
        # offset while preserving the legacy tables for the flat baseline.
        self.robot_root_to_human_root_offset = np.zeros(3, dtype=float)
        self.rot_offsets1 = {name: type(offset).identity() for name, offset in self.rot_offsets1.items()}
        self.rot_offsets2 = {name: type(offset).identity() for name, offset in self.rot_offsets2.items()}
        self.scene_transform = (
            scene_transform_config
            if isinstance(scene_transform_config, SceneTransform)
            else SceneTransform(**scene_transform_config)
        )
        source_terrain = terrain_spec if isinstance(terrain_spec, TerrainField) else (
            TerrainField.from_spec(terrain_spec) if isinstance(terrain_spec, dict) else TerrainField.from_file(terrain_spec)
        )
        source_terrain.support_normal_min_z = float(self.terrain_config["terrain"]["support_normal_min_z"])
        self.source_terrain = source_terrain
        self.terrain = source_terrain.transform(self.scene_transform)

        # SceneTransform owns global scale.  Preserve only relative limb-scale
        # ratios from the established GMR mapping to avoid scaling the root and
        # terrain in different coordinate systems.
        root_scale = float(self.human_scale_table.get(self.human_root_name, 1.0))
        for name in self.human_scale_table:
            self.human_scale_table[name] /= root_scale
        self._motion_floor_z = None

        # Position mocap has no measured rotations.  Restore orientation cost
        # only for the explicitly derived pelvis/chest frames; foot orientation
        # is handled relative to terrain normals below.
        self.set_orientation_valid(False)

        ground_cfg = self.graph_config["ground_nonpenetration"]
        terrain_limit_cfg = self.terrain_config["terrain_nonpenetration"]
        mesh_guards = dict(ground_cfg.get("dynamic_mesh_guards", {}))
        mesh_guards.update(terrain_limit_cfg.get("mesh_shells", {}))
        guard_sites = ground_cfg.get("adaptive_guard_sites", ground_cfg.get("always_active_sites", {}))
        self.terrain_limit = TerrainNonPenetrationLimit(
            self.model,
            self.terrain,
            self.surface_regions,
            guard_sites,
            ground_cfg.get("collision_geoms", {}),
            self.graph_config.get("ground_collision_points", {}),
            mesh_guards,
            terrain_limit_cfg,
        )

        task_cfg = self.terrain_config["terrain_contact_tasks"]
        point_specs = self.terrain_config["robot_contact_points"]
        self.terrain_point_task = TerrainPointContactTask(
            self.model, point_specs,
            task_cfg["normal_cost"], task_cfg["tangent_cost"], task_cfg["clearance"],
        )
        foot_specs = {name: point_specs[name] for name in TerrainFootTemporalTask.CHANNELS}
        self.terrain_foot_orientation_task = TerrainFootOrientationTask(
            self.model,
            {"left": "ANKLE_ROLL_L_LINK", "right": "ANKLE_ROLL_R_LINK"},
            task_cfg["sole_local_normal"], task_cfg["foot_orientation_cost"],
        )
        self.terrain_foot_temporal_task = TerrainFootTemporalTask(
            self.model, foot_specs, self.motion_dt, task_cfg["temporal_tangent_cost"],
        )
        self.terrain_alpha = 0.0
        self.terrain_blend_frames = max(1, int(task_cfg["contact_blend_frames"]))
        self.terrain_inner_iterations = max(1, int(task_cfg["terrain_inner_iterations"]))
        self.terrain_first_frame_iterations = max(1, int(task_cfg["first_frame_iterations"]))
        self.terrain_near_slack = float(terrain_limit_cfg["near_slack_threshold"])
        self.last_terrain_contact_frame: dict[str, Any] = {"contacts": {}, "flat_foot": {}}
        self.terrain_diagnostics: list[dict[str, Any]] = []
        self._last_surface_ids: dict[str, str] = {}
        self._previous_contact_points: dict[str, np.ndarray] = {}

    def _prepare_target_data(self, human_data):
        # Terrain inputs have already undergone the one SceneTransform.  Do not
        # infer or subtract a global floor from the lowest human foot.
        return self._prepare_unshifted_tables(human_data)

    def _prepare_unshifted_tables(self, human_data):
        table1, table2 = super()._prepare_unshifted_tables(human_data)
        for body_name in ("pelvis", "spine3"):
            if body_name not in human_data:
                continue
            quaternion = np.asarray(human_data[body_name][1], dtype=float).copy()
            if body_name in table1:
                table1[body_name][1] = quaternion.copy()
            if body_name in table2:
                table2[body_name][1] = quaternion.copy()
        return table1, table2

    def update_targets(self, human_data, offset_to_ground=False, contact_frame=None):
        super().update_targets(human_data, offset_to_ground, contact_frame=None)
        if self.retarget_call_count == 0:
            # The default free-joint pose may lie inside a terrain primitive.
            # Seed the first linearization at the transformed pelvis target so
            # opposing box-face inequalities do not make the initial QP
            # infeasible before the primary root task can move the robot.
            root_target = self.scaled_human_data[self.human_root_name]
            qpos = self.configuration.data.qpos.copy()
            qpos[:3] = np.asarray(root_target[0], dtype=float)
            qpos[3:7] = np.asarray(root_target[1], dtype=float)
            self.configuration.update(qpos)
        contact_frame = contact_frame or {"contacts": {}, "flat_foot": {}}
        contacts = contact_frame.get("contacts", {})
        non_floor = any(
            item.get("surface_type") != "floor" and float(item.get("score", 0.0)) > 0.15
            for item in contacts.values()
        )
        target_alpha = 1.0 if non_floor else 0.0
        step = 1.0 / self.terrain_blend_frames
        self.terrain_alpha += float(np.clip(target_alpha - self.terrain_alpha, -step, step))
        # Contact schedules already contain their own 7-frame score blend.  A
        # second terrain_alpha only blends mode-level entry/exit, never safety.
        blended = {
            name: {**item, "score": float(item.get("score", 0.0)) * max(self.terrain_alpha, 1.0 if item.get("surface_type") == "floor" else 0.0)}
            for name, item in contacts.items()
        }
        self.terrain_point_task.set_contacts(self.configuration, blended)
        self.terrain_foot_orientation_task.set_contacts(blended, contact_frame.get("flat_foot", {}))
        self.terrain_foot_temporal_task.begin_frame(self.configuration, blended)
        self.last_terrain_contact_frame = {**contact_frame, "contacts": blended}

    def _passes(self, min_slack: float, contacts: dict) -> int:
        if self.retarget_call_count == 0:
            return self.terrain_first_frame_iterations
        active_contact = any(float(item.get("score", 0.0)) > 0.15 for item in contacts.values())
        surface_switch = any(
            name in self._last_surface_ids and self._last_surface_ids[name] != item.get("surface_id")
            for name, item in contacts.items()
            if item.get("state") != "NONE"
        )
        if min_slack < self.terrain_near_slack or active_contact or surface_switch:
            return self.terrain_inner_iterations
        return 1

    def retarget(self, human_data, offset_to_ground=False, contact_frame=None):
        self.update_targets(human_data, offset_to_ground, contact_frame)
        contacts = self.last_terrain_contact_frame.get("contacts", {})
        min_slack_before = self.terrain_limit.min_signed_slack(self.configuration)
        passes = self._passes(min_slack_before, contacts)
        tasks = list(self.primary_tasks) + [
            self.graph_task,
            self.terrain_point_task,
            self.terrain_foot_orientation_task,
            self.terrain_foot_temporal_task,
        ]
        if self._q_prev is not None:
            posture = mink.PostureTask(self.model, cost=0.02, lm_damping=1.0)
            posture.set_target(self._q_prev)
            tasks.append(posture)
        limits = list(self.ik_limits) + [self.terrain_limit]
        sub_dt = self.motion_dt / passes
        qp_failures = 0
        qp_errors = []
        for _ in range(passes):
            self.terrain_limit.prepare_active_set(self.configuration, sub_dt, contacts)
            try:
                velocity = mink.solve_ik(
                    self.configuration, tasks, sub_dt, self.solver, self.damping, limits=limits
                )
            except Exception as error:
                qp_failures += 1
                qp_errors.append(f"{type(error).__name__}: {error}")
                continue
            self.configuration.integrate_inplace(velocity, sub_dt)

        forced = self.terrain_limit.force_activate_violations(self.configuration)
        if forced:
            self.terrain_limit.prepare_active_set(self.configuration, sub_dt, contacts)
            try:
                velocity = mink.solve_ik(
                    self.configuration, tasks, sub_dt, self.solver, self.damping, limits=limits
                )
                self.configuration.integrate_inplace(velocity, sub_dt)
            except Exception as error:
                qp_failures += 1
                qp_errors.append(f"{type(error).__name__}: {error}")

        final_slacks = self.terrain_limit.measure_current_slacks(self.configuration)
        contact_metrics = {}
        for name, contact in contacts.items():
            if name not in self.terrain_point_task.points:
                continue
            point = self.terrain_point_task.points[name].point(self.configuration)
            normal = np.asarray(contact.get("surface_normal_solver", [0.0, 0.0, 1.0]), dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            previous = self._previous_contact_points.get(name)
            velocity = np.zeros(3) if previous is None else (point - previous) / self.motion_dt
            tangent_velocity = velocity - normal * float(normal @ velocity)
            contact_metrics[name] = {
                "score": float(contact.get("score", 0.0)),
                "state": str(contact.get("state", "NONE")),
                "surface_id": str(contact.get("surface_id", "")),
                "robot_point": point.copy(),
                "surface_point": np.asarray(contact.get("surface_point_solver", point), dtype=float).copy(),
                "surface_normal": normal.copy(),
                "normal_distance": float(normal @ (point - np.asarray(contact.get("surface_point_solver", point), dtype=float))),
                "tangential_speed": float(np.linalg.norm(tangent_velocity)),
            }
            self._previous_contact_points[name] = point.copy()
        foot_orientation_errors = {}
        for side, body_id in self.terrain_foot_orientation_task.body_ids.items():
            target = self.terrain_foot_orientation_task.targets[side]
            sole_normal = (
                self.configuration.data.xmat[body_id].reshape(3, 3)
                @ self.terrain_foot_orientation_task.local_normal
            )
            foot_orientation_errors[side] = float(np.arccos(np.clip(
                sole_normal @ target["normal"], -1.0, 1.0
            ))) if target["activation"] > 0.0 else 0.0
        self.terrain_foot_temporal_task.end_frame(self.configuration)
        self._last_surface_ids = {
            name: item["surface_id"] for name, item in contacts.items() if item.get("state") != "NONE"
        }
        self.terrain_diagnostics.append({
            "frame": self.retarget_call_count,
            "passes": passes,
            "active_constraints": self.terrain_limit.active_count,
            "active_candidates": [
                self.terrain_limit.entries[index]["name"]
                for index in self.terrain_limit.active_indices
            ] + [
                f"mesh:{name}" for name, active in self.terrain_limit.mesh_active_mask.items() if active
            ],
            "qp_failures": qp_failures,
            "qp_errors": qp_errors,
            "forced_candidates": forced,
            "min_slack_before": float(min_slack_before),
            "min_slack_after": float(min((item["slack"] for item in final_slacks.values()), default=np.inf)),
            "slacks": final_slacks,
            "contacts": contact_metrics,
            "foot_orientation_error_rad": foot_orientation_errors,
            "terrain_alpha": self.terrain_alpha,
        })
        self.retarget_call_count += 1
        self._q_prev2, self._q_prev = self._q_prev, self.configuration.data.qpos.copy()
        return self.configuration.data.qpos.copy()
