"""Graph-regularized, soft-contact motion retargeting.

This module is intentionally separate from ``motion_retarget.py``.  It keeps
the existing GMR implementation available for regression comparisons while
providing an alternative NE01 pipeline:

* semantic human-body graph regularization through Laplacian coordinates;
* one combined, low-conflict IK solve instead of a world-foot hard lock;
* unilateral soft ground contact that only reacts to sole penetration;
* no post-IK root XY teleport or persistent foot anchor.

The graph task preserves local body relationships.  It is not a Delaunay mesh:
the graph topology is fixed by human semantics so it cannot change between
frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import mink
import mujoco as mj
import numpy as np
from mink.tasks.task import Task

from .motion_retarget import GeneralMotionRetargeting


class SemanticLaplacianTask(Task):
    """Track local Laplacian coordinates of mapped robot body origins."""

    def __init__(
        self,
        model: mj.MjModel,
        configuration: mink.Configuration,
        node_names: list[str],
        frame_names: list[str],
        edges: list[tuple[int, int]],
        cost: float = 1.0,
        gain: float = 0.5,
    ) -> None:
        if len(node_names) != len(frame_names):
            raise ValueError("node_names and frame_names must have equal length")
        self.model = model
        self.configuration = configuration
        self.node_names = list(node_names)
        self.frame_names = list(frame_names)
        self.body_ids = np.asarray(
            [model.body(name).id for name in frame_names], dtype=int
        )
        self.neighbors = [[] for _ in frame_names]
        for i, j in edges:
            if i == j:
                continue
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)
        self.target_laplacian = np.zeros((len(frame_names), 3), dtype=float)
        super().__init__(
            cost=np.full(3 * len(frame_names), float(cost), dtype=float),
            gain=float(gain),
            lm_damping=1.0,
        )

    def set_target_positions(self, positions: np.ndarray) -> None:
        positions = np.asarray(positions, dtype=float)
        if positions.shape != self.target_laplacian.shape:
            raise ValueError(
                f"Expected graph target {self.target_laplacian.shape}, got {positions.shape}"
            )
        for i, neighbors in enumerate(self.neighbors):
            if neighbors:
                self.target_laplacian[i] = positions[i] - np.mean(
                    positions[neighbors], axis=0
                )
            else:
                self.target_laplacian[i] = 0.0

    def _positions_and_jacobians(self):
        data = self.configuration.data
        positions = data.xpos[self.body_ids].copy()
        jacobians = []
        for body_id in self.body_ids:
            jacp = np.zeros((3, self.model.nv), dtype=float)
            jacr = np.zeros((3, self.model.nv), dtype=float)
            mj.mj_jac(
                self.model,
                data,
                jacp,
                jacr,
                data.xpos[int(body_id)],
                int(body_id),
            )
            jacobians.append(jacp)
        return positions, jacobians

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        positions, _ = self._positions_and_jacobians()
        residual = np.zeros_like(positions)
        for i, neighbors in enumerate(self.neighbors):
            if neighbors:
                residual[i] = positions[i] - np.mean(positions[neighbors], axis=0)
        return (residual - self.target_laplacian).reshape(-1)

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        _, jacobians = self._positions_and_jacobians()
        jacobian = np.zeros((3 * len(self.neighbors), self.model.nv), dtype=float)
        for i, neighbors in enumerate(self.neighbors):
            row = slice(3 * i, 3 * i + 3)
            if neighbors:
                jacobian[row] = jacobians[i] - np.mean(
                    [jacobians[j] for j in neighbors], axis=0
                )
        return jacobian


class SoftFootGroundTask(Task):
    """Unilateral sole-ground task with no horizontal foot anchor.

    Above-ground feet contribute zero error.  A foot below the ground plane
    contributes a smooth upward correction, so normal contact is helped while
    swing and release are not pinned to a stale world position.
    """

    def __init__(
        self,
        model: mj.MjModel,
        geom_names: list[str],
        ground_height: float = 0.0,
        clearance: float = 0.002,
        contact_band: float = 0.03,
        cost: float = 8.0,
    ) -> None:
        self.model = model
        self.geom_ids = np.asarray(
            [model.geom(name).id for name in geom_names], dtype=int
        )
        self.body_ids = np.asarray(model.geom_bodyid[self.geom_ids], dtype=int)
        self.ground_height = float(ground_height)
        self.clearance = float(clearance)
        self.contact_band = float(contact_band)
        self.activation = 0.0
        self._active_rows = np.zeros(len(self.geom_ids), dtype=bool)
        super().__init__(
            cost=np.full(len(self.geom_ids), float(cost), dtype=float),
            gain=0.6,
            lm_damping=1.0,
        )

    def set_activation(self, activation: float) -> None:
        self.activation = float(np.clip(activation, 0.0, 1.0))

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        heights = (
            configuration.data.geom_xpos[self.geom_ids, 2]
            - self.model.geom_size[self.geom_ids, 0]
            - self.ground_height
            - self.clearance
        )
        # The penetration barrier is always active.  Contact activation only
        # adds a bounded attraction band around the floor; it must not disable
        # the anti-penetration safety net while a foot is classified as AIR.
        penetration = np.minimum(heights, 0.0)
        contact = np.where(np.abs(heights) < self.contact_band, heights, 0.0)
        self._active_rows = (heights < 0.0) | (
            (self.activation > 0.05) & (np.abs(heights) < self.contact_band)
        )
        return penetration + self.activation * contact

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        jacobian = np.zeros((len(self.geom_ids), self.model.nv), dtype=float)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        for row, (geom_id, body_id) in enumerate(zip(self.geom_ids, self.body_ids)):
            if not self._active_rows[row]:
                continue
            jacp.fill(0.0)
            jacr.fill(0.0)
            mj.mj_jac(
                self.model,
                configuration.data,
                jacp,
                jacr,
                configuration.data.geom_xpos[int(geom_id)],
                int(body_id),
            )
            height = (
                configuration.data.geom_xpos[geom_id, 2]
                - self.model.geom_size[geom_id, 0]
                - self.ground_height
                - self.clearance
            )
            scale = float(height < 0.0) + self.activation * float(
                abs(height) < self.contact_band
            )
            jacobian[row] = scale * jacp[2]
        return jacobian


class SoftFootTangentialTask(Task):
    """Finite, releasable XY support task for a confirmed stance foot."""

    def __init__(self, model: mj.MjModel, geom_ids, cost: float = 20.0) -> None:
        self.model = model
        self.geom_ids = np.asarray(geom_ids, dtype=int)
        self.body_id = int(model.geom_bodyid[self.geom_ids[0]])
        self.target_xy = np.zeros(2, dtype=float)
        self.activation = 0.0
        super().__init__(cost=np.full(2, float(cost)), gain=0.55, lm_damping=1.0)

    def set_target(self, target_xy: np.ndarray, activation: float) -> None:
        self.target_xy = np.asarray(target_xy, dtype=float).reshape(2)
        self.activation = float(np.clip(activation, 0.0, 1.0))

    def _support_point(self, configuration: mink.Configuration) -> np.ndarray:
        return np.mean(configuration.data.geom_xpos[self.geom_ids], axis=0)

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        return self.activation * (self._support_point(configuration)[:2] - self.target_xy)

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mj.mj_jac(
            self.model,
            configuration.data,
            jacp,
            jacr,
            self._support_point(configuration),
            self.body_id,
        )
        return self.activation * jacp[:2]


class LaplacianSoftContactRetargeting(GeneralMotionRetargeting):
    """Alternative retargeter that leaves the legacy GMR class untouched."""

    DEFAULT_GRAPH_EDGES = [
        ("pelvis", "spine3"),
        ("pelvis", "left_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_foot"),
        ("pelvis", "right_hip"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_foot"),
        ("spine3", "left_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("spine3", "right_shoulder"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
    ]

    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float | None = None,
        solver: str = "proxqp",
        damping: float = 5e-1,
        verbose: bool = True,
        use_velocity_limit: bool = True,
        velocity_limit: float = 3 * np.pi,
        motion_fps: float = 50.0,
        graph_config_path: str | Path | None = None,
    ) -> None:
        # The parent initializes the model and regular FrameTasks.  This class
        # deliberately does not call any parent contact or post-processing path.
        super().__init__(
            src_human=src_human,
            tgt_robot=tgt_robot,
            actual_human_height=actual_human_height,
            solver=solver,
            damping=damping,
            verbose=verbose,
            use_velocity_limit=use_velocity_limit,
            velocity_limit=velocity_limit,
            motion_fps=motion_fps,
            legacy_mode=True,
        )
        self.graph_config = self._load_graph_config(graph_config_path)
        self.graph_cost = float(self.graph_config.get("graph_cost", 1.5))
        self.graph_gain = float(self.graph_config.get("graph_gain", 0.45))
        self.soft_contact_cost = float(
            self.graph_config.get("soft_contact_cost", 8.0)
        )
        self.soft_contact_height = float(
            self.graph_config.get("soft_contact_height", 0.04)
        )
        self.soft_contact_speed = float(
            self.graph_config.get("soft_contact_speed", 0.35)
        )
        self.soft_contact_smoothing = float(
            self.graph_config.get("soft_contact_smoothing", 0.25)
        )
        self.soft_contact_vertical_speed = float(
            self.graph_config.get("soft_contact_vertical_speed", 0.35)
        )
        self.soft_contact_stance_on = float(
            self.graph_config.get("soft_contact_stance_on", 0.30)
        )
        self.soft_contact_stance_off = float(
            self.graph_config.get("soft_contact_stance_off", 0.12)
        )
        self.soft_contact_xy_cost = float(
            self.graph_config.get("soft_contact_xy_cost", 35.0)
        )
        self.soft_contact_xy_release_rate = float(
            self.graph_config.get("soft_contact_xy_release_rate", 0.15)
        )
        self.soft_contact_xy_transition_frames = max(
            1,
            int(self.graph_config.get("soft_contact_xy_transition_frames", 5)),
        )
        self.floor_percentile = float(
            self.graph_config.get("floor_percentile", 2.0)
        )
        self.soft_contact_clearance = float(
            self.graph_config.get("soft_contact_clearance", 0.002)
        )
        self.soft_contact_band = float(
            self.graph_config.get("soft_contact_band", 0.03)
        )
        self.soft_root_lift_rate = float(
            self.graph_config.get("soft_root_lift_rate", 0.01)
        )
        self._soft_contact_scores = {"left_foot": 0.0, "right_foot": 0.0}
        self._soft_contact_history = {"left_foot": [], "right_foot": []}
        self._soft_xy_active = {"left_foot": False, "right_foot": False}
        self._soft_xy_anchor = {"left_foot": None, "right_foot": None}
        self._soft_xy_blend = {"left_foot": 0.0, "right_foot": 0.0}
        self._motion_floor_z = None

        self._build_semantic_graph()
        self._build_soft_ground_tasks()
        self._build_primary_tasks()

    @staticmethod
    def _load_graph_config(path: str | Path | None) -> dict:
        if path is None:
            return {}
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("graph config must be a JSON object")
        return data

    def _build_semantic_graph(self) -> None:
        mapping = {}
        for task in self.tasks1:
            frame_name = self.task_frame_names.get(task)
            if frame_name is None:
                continue
            for human_name, candidate in self.human_body_to_task1.items():
                if candidate is task:
                    mapping[human_name] = frame_name
                    break

        configured_nodes = self.graph_config.get("graph_nodes")
        if configured_nodes:
            node_names = [str(name) for name in configured_nodes]
        else:
            node_names = sorted(
                {name for edge in self.DEFAULT_GRAPH_EDGES for name in edge}
            )
        valid_nodes = [name for name in node_names if name in mapping]
        node_index = {name: i for i, name in enumerate(valid_nodes)}
        raw_edges = self.graph_config.get("graph_edges") or self.DEFAULT_GRAPH_EDGES
        edges = []
        for edge in raw_edges:
            if len(edge) != 2 or edge[0] not in node_index or edge[1] not in node_index:
                continue
            edges.append((node_index[edge[0]], node_index[edge[1]]))
        if len(valid_nodes) < 4 or not edges:
            raise ValueError("semantic graph does not contain enough mapped nodes")

        self.graph_node_names = valid_nodes
        self.graph_frame_names = [mapping[name] for name in valid_nodes]
        self.graph_task = SemanticLaplacianTask(
            self.model,
            self.configuration,
            self.graph_node_names,
            self.graph_frame_names,
            edges,
            cost=self.graph_cost,
            gain=self.graph_gain,
        )

    def _build_soft_ground_tasks(self) -> None:
        self.soft_ground_tasks = {}
        self.soft_tangential_tasks = {}
        if self.tgt_robot != "ne01":
            return
        for side in ("left", "right"):
            foot_name = f"{side}_foot"
            geom_names = [
                f"{side}_foot_rear_left_collision",
                f"{side}_foot_rear_right_collision",
                f"{side}_foot_front_left_collision",
                f"{side}_foot_front_right_collision",
            ]
            self.soft_ground_tasks[foot_name] = SoftFootGroundTask(
                self.model,
                geom_names,
                ground_height=float(self.ground[2]),
                clearance=self.soft_contact_clearance,
                contact_band=self.soft_contact_band,
                cost=self.soft_contact_cost,
            )
            if self.soft_contact_xy_cost > 0.0:
                self.soft_tangential_tasks[foot_name] = SoftFootTangentialTask(
                    self.model,
                    [self.model.geom(name).id for name in geom_names],
                    cost=self.soft_contact_xy_cost,
                )

    def _build_primary_tasks(self) -> None:
        # Keep one regular task per robot frame.  This avoids duplicating table1
        # and table2 costs while retaining the strongest existing mapping.
        selected = {}
        for task in self.tasks1 + self.tasks2:
            frame = self.task_frame_names.get(task)
            if frame is None:
                continue
            strength = float(np.sum(task.cost))
            if frame not in selected or strength > selected[frame][0]:
                selected[frame] = (strength, task)
        self.primary_tasks = [item[1] for item in selected.values()]

        # Foot targets should not dominate the graph and torso/root tasks.
        for task in self.primary_tasks:
            frame = self.task_frame_names.get(task, "")
            if "toe_link" in frame or "ANKLE" in frame:
                task.cost[:3] *= 0.30
                task.cost[3:] *= 0.50

    def _prepare_target_data(self, human_data):
        table1, table2 = self._prepare_unshifted_tables(human_data)

        foot_heights = [
            float(table1[name][0][2])
            for name in ("left_foot", "right_foot")
            if name in table1
        ]
        if foot_heights:
            floor_z = (
                float(self._motion_floor_z)
                if self._motion_floor_z is not None
                else float(min(foot_heights))
            )
            shift = floor_z - float(self.ground[2])
            for pos, _ in table1.values():
                pos[2] -= shift

        if foot_heights:
            for pos, _ in table2.values():
                pos[2] -= floor_z - float(self.ground[2])
        return table1, table2

    def _prepare_unshifted_tables(self, human_data):
        raw = {
            name: [np.asarray(value[0], dtype=float).copy(), np.asarray(value[1], dtype=float).copy()]
            for name, value in human_data.items()
        }
        scaled = self.scale_human_data(raw, self.human_root_name, self.human_scale_table)
        base = {k: [v[0].copy(), v[1].copy()] for k, v in scaled.items()}
        table1 = self.offset_human_data(base, self.pos_offsets1, self.rot_offsets1)
        table1 = self.apply_robot_root_to_human_root_offset(table1)
        table1 = self.apply_ground_offset(table1)
        table2 = self.offset_human_data(base, self.pos_offsets2, self.rot_offsets2)
        table2 = self.apply_robot_root_to_human_root_offset(table2)
        table2 = self.apply_ground_offset(table2)
        return table1, table2

    def set_motion_floor(self, motion_frames) -> float:
        """Calibrate one immutable floor height from the complete motion."""
        samples = []
        for frame in motion_frames:
            table1, _ = self._prepare_unshifted_tables(frame)
            values = [
                float(table1[name][0][2])
                for name in ("left_foot", "right_foot")
                if name in table1
            ]
            if values:
                samples.append(min(values))
        if not samples:
            self._motion_floor_z = float(self.ground[2])
        else:
            self._motion_floor_z = float(np.percentile(samples, self.floor_percentile))
        self._human_floor_z = self._motion_floor_z
        return self._motion_floor_z

    def _update_soft_contacts(self, human_data: dict) -> None:
        dt = max(float(self.motion_dt), 1e-6)
        for name, task in self.soft_ground_tasks.items():
            if name not in human_data:
                task.set_activation(0.0)
                continue
            pos = np.asarray(human_data[name][0], dtype=float)
            history = self._soft_contact_history[name]
            history.append(pos.copy())
            if len(history) > 3:
                del history[:-3]
            speed = 0.0
            vertical_speed = 0.0
            if len(history) >= 2:
                speed = float(np.linalg.norm((history[-1] - history[-2])[:2]) / dt)
                vertical_speed = float(abs((history[-1] - history[-2])[2]) / dt)
            height_score = np.clip(
                (self.soft_contact_height - (pos[2] - self.ground[2]))
                / max(self.soft_contact_height, 1e-6),
                0.0,
                1.0,
            )
            vertical_score = np.clip(
                1.0 - vertical_speed / max(self.soft_contact_vertical_speed, 1e-6),
                0.0,
                1.0,
            )
            # Horizontal speed is intentionally not a hard contact gate: the
            # SMPL-X foot joint can move quickly in a real stance window.
            raw_score = float(height_score * vertical_score)
            previous = self._soft_contact_scores[name]
            score = (1.0 - self.soft_contact_smoothing) * previous + self.soft_contact_smoothing * raw_score
            self._soft_contact_scores[name] = score
            task.set_activation(score)
            xy_task = self.soft_tangential_tasks.get(name)
            if xy_task is None:
                continue
            if (
                self._soft_xy_anchor[name] is None
                and score >= self.soft_contact_stance_on
            ):
                self._soft_xy_active[name] = True
                self._soft_xy_anchor[name] = xy_task._support_point(self.configuration)[:2].copy()
            elif self._soft_xy_active[name] and score <= self.soft_contact_stance_off:
                self._soft_xy_active[name] = False
            blend_step = 1.0 / self.soft_contact_xy_transition_frames
            if self._soft_xy_active[name]:
                self._soft_xy_blend[name] = min(
                    1.0, self._soft_xy_blend[name] + blend_step
                )
            else:
                self._soft_xy_blend[name] = max(
                    0.0, self._soft_xy_blend[name] - blend_step
                )
                if self._soft_xy_blend[name] <= 1e-9:
                    self._soft_xy_anchor[name] = None
            if self._soft_xy_active[name]:
                target_xy = np.asarray(human_data[name][0][:2], dtype=float)
                anchor = self._soft_xy_anchor[name]
                delta = target_xy - anchor
                distance = float(np.linalg.norm(delta))
                max_step = self.soft_contact_xy_release_rate * dt
                if distance > max_step and distance > 1e-9:
                    anchor = anchor + delta / distance * max_step
                else:
                    anchor = target_xy.copy()
                self._soft_xy_anchor[name] = anchor
            if self._soft_xy_anchor[name] is not None:
                xy_task.set_target(
                    self._soft_xy_anchor[name], self._soft_xy_blend[name]
                )
            else:
                xy_task.set_target(np.zeros(2), 0.0)

    def update_targets(self, human_data, offset_to_ground=False):
        table1, table2 = self._prepare_target_data(human_data)
        self.scaled_human_data = table1

        for body_name, task in self.human_body_to_task1.items():
            if body_name not in table1:
                continue
            pos, rot = table1[body_name]
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

        for body_name, task in self.human_body_to_task2.items():
            if body_name not in table2:
                continue
            pos, rot = table2[body_name]
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

        graph_positions = np.asarray(
            [table1[name][0] for name in self.graph_node_names], dtype=float
        )
        self.graph_task.set_target_positions(graph_positions)
        self._update_soft_contacts(table1)

    def retarget(self, human_data, offset_to_ground=False):
        self.update_targets(human_data, offset_to_ground)
        # Keep one velocity-limited integration per output frame.  A second
        # integration can accumulate two admissible velocities into a visible
        # joint jump at contact transitions.
        passes = 8 if self.retarget_call_count == 0 else 1
        tasks = list(self.primary_tasks) + [self.graph_task]
        tasks.extend(self.soft_ground_tasks.values())
        tasks.extend(self.soft_tangential_tasks.values())
        if self._q_prev is not None:
            posture = mink.PostureTask(self.model, cost=0.02, lm_damping=1.0)
            posture.set_target(self._q_prev)
            tasks.append(posture)

        for _ in range(passes):
            velocity = mink.solve_ik(
                self.configuration,
                tasks,
                self.motion_dt,
                self.solver,
                self.damping,
                limits=self.ik_limits,
            )
            self.configuration.integrate_inplace(velocity, self.motion_dt)

        self._apply_soft_root_ground_lift()
        self.retarget_call_count += 1
        self._q_prev2 = self._q_prev
        self._q_prev = self.configuration.data.qpos.copy()
        return self.configuration.data.qpos.copy()

    def _apply_soft_root_ground_lift(self) -> None:
        """Rate-limited normal-only lift used after graph IK.

        This is deliberately a scalar base-Z correction.  It never changes
        root XY or foot orientation, so it cannot pin a stance foot or create
        the old double-support root jump.
        """
        if self.tgt_robot != "ne01" or not self.soft_ground_tasks:
            return
        mj.mj_forward(self.model, self.configuration.data)
        lowest = np.inf
        for task in self.soft_ground_tasks.values():
            heights = (
                self.configuration.data.geom_xpos[task.geom_ids, 2]
                - self.model.geom_size[task.geom_ids, 0]
            )
            lowest = min(lowest, float(np.min(heights)))
        if not np.isfinite(lowest):
            return

        correction = max(
            0.0,
            float(self.ground[2]) + self.soft_contact_clearance - lowest,
        )
        correction = min(correction, self.soft_root_lift_rate)
        if correction <= 1e-9:
            return
        qpos = self.configuration.data.qpos.copy()
        qpos[2] += correction
        self.configuration.update(qpos)
