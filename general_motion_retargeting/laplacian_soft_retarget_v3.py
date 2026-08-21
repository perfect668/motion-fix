"""V3 staged NE01 retargeting pipeline.

This module is deliberately separate from the legacy and laplacian-soft
entrances.  It keeps the existing target preparation and semantic graph, but
solves each output frame in two protected stages and commits qpos once.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import mink
import mujoco as mj
import numpy as np
from mink.tasks.task import Task
from mink.limits.limit import Constraint, Limit
from scipy.spatial.transform import Rotation

from . import motion_retarget as motion_retarget_module
from .laplacian_soft_retarget import LaplacianSoftContactRetargeting


class FootContactPhase(str, Enum):
    AIR = "AIR"
    HEEL = "HEEL"
    FLAT = "FLAT"
    TOE = "TOE"


class FootLoadPhase(str, Enum):
    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    SUPPORT = "SUPPORT"
    UNLOADING = "UNLOADING"


@dataclass
class FootContactState:
    contact_phase: FootContactPhase = FootContactPhase.AIR
    load_phase: FootLoadPhase = FootLoadPhase.UNLOADED
    contact_confidence: float = 0.0
    liftoff_probability: float = 0.0
    lock_weight: float = 0.0
    anchor_xy: np.ndarray | None = None
    frames_in_state: int = 0
    last_transition_frame: int = -1
    on_counter: int = 0
    off_counter: int = 0


class QposSubsetTask(Task):
    """Position regularization for hinge/slide qpos coordinates."""

    def __init__(self, model, qpos_indices, target, cost=1.0, gain=0.4):
        self.model = model
        self.qpos_indices = np.asarray(qpos_indices, dtype=int)
        self.dof_indices = np.asarray(
            [self._qpos_to_dof_index(int(index)) for index in self.qpos_indices],
            dtype=int,
        )
        self.target = np.asarray(target, dtype=float).reshape(-1).copy()
        if self.qpos_indices.shape != self.target.shape:
            raise ValueError("qpos_indices and target must have equal length")
        cost_array = np.broadcast_to(
            np.asarray(cost, dtype=float), (len(self.qpos_indices),)
        ).copy()
        super().__init__(
            cost=cost_array,
            gain=float(gain),
            lm_damping=1.0,
        )

    def _qpos_to_dof_index(self, qpos_index):
        for joint_id in range(self.model.njnt):
            qpos_start = int(self.model.jnt_qposadr[joint_id])
            dof_start = int(self.model.jnt_dofadr[joint_id])
            joint_type = self.model.jnt_type[joint_id]
            if joint_type == mj.mjtJoint.mjJNT_FREE:
                # Only the three translational free-joint coordinates have a
                # direct scalar qpos/dof correspondence.
                if qpos_start <= qpos_index < qpos_start + 3:
                    return dof_start + qpos_index - qpos_start
            elif joint_type in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                if qpos_index == qpos_start:
                    return dof_start
        raise ValueError(f"qpos index {qpos_index} has no scalar dof mapping")

    def set_target_qpos(self, target):
        target = np.asarray(target, dtype=float).reshape(-1)
        if target.shape != self.target.shape:
            raise ValueError("invalid qpos target shape")
        self.target = target.copy()

    def compute_error(self, configuration):
        return configuration.data.qpos[self.qpos_indices] - self.target

    def compute_jacobian(self, configuration):
        jac = np.zeros((len(self.qpos_indices), self.model.nv), dtype=float)
        jac[np.arange(len(self.qpos_indices)), self.dof_indices] = 1.0
        return jac


class RootZTask(QposSubsetTask):
    def __init__(self, model, target, cost=1.0):
        super().__init__(model, [2], [target], cost=cost, gain=0.35)


class FootNonPenetrationLimit(Limit):
    """Fixed-row unilateral ground constraint for all sole proxy points."""

    def __init__(self, model, geom_ids, ground_height=0.0, clearance=0.0, gain=1.0):
        self.model = model
        self.geom_ids = np.asarray(geom_ids, dtype=int)
        self.body_ids = np.asarray(model.geom_bodyid[self.geom_ids], dtype=int)
        self.ground_height = float(ground_height)
        self.clearance = float(clearance)
        self.gain = float(gain)

    def compute_qp_inequalities(self, configuration, dt):
        rows = np.zeros((len(self.geom_ids), self.model.nv), dtype=float)
        bounds = np.zeros(len(self.geom_ids), dtype=float)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        for row, (geom_id, body_id) in enumerate(zip(self.geom_ids, self.body_ids)):
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
                configuration.data.geom_xpos[int(geom_id), 2]
                - self.model.geom_size[int(geom_id), 0]
                - self.ground_height
            )
            rows[row] = -jacp[2]
            bounds[row] = self.gain * (height - self.clearance) / max(float(dt), 1e-6)
        return Constraint(G=rows, h=bounds)


class FootNormalTask(Task):
    """Fixed-row unilateral sole task with a continuous support gain."""

    def __init__(self, model, geom_ids, ground_height=0.0, cost=20.0):
        self.model = model
        self.geom_ids = np.asarray(geom_ids, dtype=int)
        self.body_ids = np.asarray(model.geom_bodyid[self.geom_ids], dtype=int)
        self.ground_height = float(ground_height)
        self.activation = 0.0
        self.active_indices = np.arange(len(self.geom_ids), dtype=int)
        self.clearance = 0.002
        super().__init__(
            cost=np.full(len(self.geom_ids), float(cost)),
            gain=0.55,
            lm_damping=1.0,
        )

    def set_state(self, activation, active_indices, clearance=0.002):
        self.activation = float(np.clip(activation, 0.0, 1.0))
        self.active_indices = np.asarray(active_indices, dtype=int)
        self.clearance = float(clearance)

    def _heights(self, configuration):
        return (
            configuration.data.geom_xpos[self.geom_ids, 2]
            - self.model.geom_size[self.geom_ids, 0]
            - self.ground_height
        )

    def compute_error(self, configuration):
        heights = self._heights(configuration)
        error = np.minimum(heights - self.clearance, 0.0)
        if self.activation > 0.0 and len(self.active_indices):
            error[self.active_indices] += self.activation * (
                heights[self.active_indices] - self.clearance
            )
        return error

    def compute_jacobian(self, configuration):
        heights = self._heights(configuration)
        jacobian = np.zeros((len(self.geom_ids), self.model.nv), dtype=float)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        active = set(int(i) for i in self.active_indices.tolist())
        for row, (geom_id, body_id) in enumerate(zip(self.geom_ids, self.body_ids)):
            scale = 1.0 if heights[row] < self.clearance else 0.0
            if row in active:
                scale += self.activation
            if scale <= 0.0:
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
            jacobian[row] = scale * jacp[2]
        return jacobian


class FootAnchorTask(Task):
    """Soft XY support anchor with a radial deadband and continuous gain."""

    def __init__(self, model, geom_ids, cost=10.0, deadband=0.003):
        self.model = model
        self.geom_ids = np.asarray(geom_ids, dtype=int)
        self.body_id = int(model.geom_bodyid[self.geom_ids[0]])
        self.target_xy = np.zeros(2, dtype=float)
        self.activation = 0.0
        self.deadband = float(max(deadband, 0.0))
        super().__init__(cost=np.full(2, float(cost)), gain=0.45, lm_damping=1.0)

    def set_target(self, target_xy, activation):
        self.target_xy = np.asarray(target_xy, dtype=float).reshape(2).copy()
        self.activation = float(np.clip(activation, 0.0, 1.0))

    def _point(self, configuration):
        return np.mean(configuration.data.geom_xpos[self.geom_ids], axis=0)

    def compute_error(self, configuration):
        delta = self._point(configuration)[:2] - self.target_xy
        norm = float(np.linalg.norm(delta))
        if norm <= self.deadband:
            return np.zeros(2, dtype=float)
        return self.activation * delta * (1.0 - self.deadband / norm)

    def compute_jacobian(self, configuration):
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mj.mj_jac(self.model, configuration.data, jacp, jacr, self._point(configuration), self.body_id)
        delta = self._point(configuration)[:2] - self.target_xy
        norm = float(np.linalg.norm(delta))
        if norm <= self.deadband:
            return np.zeros((2, self.model.nv), dtype=float)
        return self.activation * (1.0 - self.deadband / norm) * jacp[:2]


class SwingClearanceTask(FootNormalTask):
    """Unilateral clearance task used only while the foot is unloading/AIR."""

    def compute_error(self, configuration):
        heights = self._heights(configuration)
        return self.activation * np.minimum(heights - self.clearance, 0.0)

    def compute_jacobian(self, configuration):
        old_activation = self.activation
        old_active = self.active_indices.copy()
        self.activation = 0.0
        self.active_indices = np.arange(len(self.geom_ids), dtype=int)
        jac = super().compute_jacobian(configuration)
        self.activation = old_activation
        self.active_indices = old_active
        return self.activation * jac


class ArmPlaneTask(Task):
    """Finite-difference elbow bending-plane task for branch continuity."""

    def __init__(self, model, body_ids, cost=3.0):
        self.model = model
        self.body_ids = tuple(int(i) for i in body_ids)
        self.target_normal = np.array([0.0, 1.0, 0.0], dtype=float)
        self.base_cost = float(cost)
        self.dof_indices = self._path_dofs(self.body_ids[-1])
        super().__init__(cost=np.full(3, float(cost)), gain=0.35, lm_damping=1.0)

    def _path_dofs(self, body_id):
        dofs = []
        body = int(body_id)
        while body > 0:
            jnt_adr = int(self.model.body_jntadr[body])
            jnt_num = int(self.model.body_jntnum[body])
            for jid in range(jnt_adr, jnt_adr + jnt_num):
                width = 6 if self.model.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE else 1
                dofs.extend(range(int(self.model.jnt_dofadr[jid]), int(self.model.jnt_dofadr[jid]) + width))
            body = int(self.model.body_parentid[body])
        return sorted(set(dofs))

    def set_target_normal(self, normal):
        normal = np.asarray(normal, dtype=float).reshape(3)
        norm = np.linalg.norm(normal)
        if norm > 1e-7:
            self.target_normal = normal / norm

    def _normal(self, data):
        p = data.xpos[list(self.body_ids)]
        normal = np.cross(p[1] - p[0], p[2] - p[1])
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 1e-7 else np.zeros(3)

    def compute_error(self, configuration):
        return self._normal(configuration.data) - self.target_normal

    def compute_jacobian(self, configuration):
        q = configuration.data.qpos.copy()
        base = self._normal(configuration.data)
        jac = np.zeros((3, self.model.nv), dtype=float)
        if np.linalg.norm(base) < 1e-7:
            return jac
        for dof in self.dof_indices:
            jid = int(np.searchsorted(self.model.jnt_dofadr, dof, side="right") - 1)
            if jid < 0 or self.model.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE:
                continue
            qadr = int(self.model.jnt_qposadr[jid]) + (dof - int(self.model.jnt_dofadr[jid]))
            q_try = q.copy()
            q_try[qadr] += 1e-4
            configuration.update(q_try)
            plus = self._normal(configuration.data)
            q_try[qadr] -= 2e-4
            configuration.update(q_try)
            minus = self._normal(configuration.data)
            jac[:, dof] = (plus - minus) / 2e-4
            configuration.update(q)
        return jac


@dataclass(frozen=True)
class ContactSnapshot:
    phase: FootContactPhase
    load: FootLoadPhase
    confidence: float
    liftoff: float
    lock_weight: float
    anchor_xy: np.ndarray | None
    active_indices: tuple[int, ...]
    heel_height: float
    toe_height: float
    xy_speed: float
    vertical_speed: float
    frames_in_state: int
    on_counter: int
    off_counter: int


class LaplacianSoftContactRetargetingV3(LaplacianSoftContactRetargeting):
    """NE01 V3: frozen-contact staged IK with one final qpos commit."""

    def __init__(self, *args, config_path=None, **kwargs):
        if config_path is None:
            config_path = Path(__file__).with_name("ik_configs") / "smplx_to_ne01_laplacian_soft_v3.json"
        self.v3_config_path = Path(config_path)
        self.v3_config = self._read_config(self.v3_config_path)
        target_robot = kwargs.get("tgt_robot", args[1] if len(args) > 1 else None)
        original_robot_xml = None
        if target_robot == "ne01":
            original_robot_xml = motion_retarget_module.ROBOT_XML_DICT["ne01"]
            motion_retarget_module.ROBOT_XML_DICT["ne01"] = (
                Path(__file__).resolve().parent.parent / "assets" / "ne01" / "ne01_v3.xml"
            )
        try:
            super().__init__(*args, graph_config_path=str(self.v3_config_path), **kwargs)
        finally:
            if original_robot_xml is not None:
                motion_retarget_module.ROBOT_XML_DICT["ne01"] = original_robot_xml
        self.v3_config = self._read_config(self.v3_config_path)
        self._velocity_limit = float(kwargs.get("velocity_limit", 3 * np.pi))
        self._frame_index = 0
        self._q_prev = None
        self._q_prev2 = None
        self._q_prev3 = None
        self._v_prev = None
        self._a_prev = None
        self._contact_states = {"left_foot": FootContactState(), "right_foot": FootContactState()}
        self._human_foot_history = {"left_foot": [], "right_foot": []}
        self._v3_foot_normal = {}
        self._v3_foot_anchor = {}
        self._v3_swing_clearance = {}
        self._v3_ground_limit = None
        self._build_v3_foot_tasks()
        self._build_v3_arm_tasks()
        self._build_v3_collision_limit()
        self.diagnostics = []
        self._last_snapshot = None
        self._frame_qp_status = "ok"
        self._stage_iterations = {"A": 0, "B": 0, "C": 0}
        self._last_collision_pair = ""
        self._collision_active = False
        self._previous_branch_sign = {"left": 0, "right": 0}
        self.events = []

    @staticmethod
    def _read_config(path):
        with Path(path).open("r", encoding="utf-8") as file:
            value = json.load(file)
        if not isinstance(value, dict):
            raise ValueError("V3 config must be a JSON object")
        return value

    def _build_v3_foot_tasks(self):
        if self.tgt_robot != "ne01":
            return
        all_geom_ids = []
        for side in ("left", "right"):
            name = f"{side}_foot"
            parent_task = self.soft_ground_tasks.get(name)
            if parent_task is None:
                continue
            ids = parent_task.geom_ids
            all_geom_ids.extend(int(value) for value in ids)
            self._v3_foot_normal[name] = FootNormalTask(
                self.model,
                ids,
                ground_height=float(self.ground[2]),
                cost=float(self.v3_config.get("contact", {}).get("normal_weight", 20.0)),
            )
            self._v3_foot_anchor[name] = FootAnchorTask(
                self.model,
                ids,
                cost=float(self.v3_config.get("contact", {}).get("anchor_xy_weight", 10.0)),
                deadband=float(self.v3_config.get("contact", {}).get("anchor_deadband", 0.003)),
            )
            self._v3_swing_clearance[name] = SwingClearanceTask(
                self.model,
                ids,
                ground_height=float(self.ground[2]),
                cost=float(self.v3_config.get("swing", {}).get("clearance_low_weight", 20.0)),
            )
        if all_geom_ids:
            self._v3_ground_limit = FootNonPenetrationLimit(
                self.model,
                all_geom_ids,
                ground_height=float(self.ground[2]),
                clearance=0.0,
            )

    def _build_v3_collision_limit(self):
        cfg = self.v3_config.get("collision", {})
        self.collision_limit = None
        self._collision_groups = []
        self._collision_geom_pairs = []
        if not bool(cfg.get("enabled", True)):
            return
        def group(geom_names):
            result = []
            for geom_name in geom_names:
                try:
                    result.append(int(self.model.geom(geom_name).id))
                except KeyError:
                    continue
            for gid in result:
                self.model.geom_contype[gid] = 1
                self.model.geom_conaffinity[gid] = 1
            return result

        torso = group(("v3_torso_proxy",))
        pelvis = group(("v3_pelvis_proxy",))
        l_fore = group(("v3_left_forearm_proxy",))
        r_fore = group(("v3_right_forearm_proxy",))
        l_hand = group(("v3_left_hand_proxy",))
        r_hand = group(("v3_right_hand_proxy",))
        pairs = [
            (l_fore, torso), (r_fore, torso), (l_hand, torso), (r_hand, torso),
            (l_fore, r_fore), (l_hand, r_hand), (l_hand, pelvis), (r_hand, pelvis),
        ]
        pairs = [(a, b) for a, b in pairs if a and b]
        if pairs:
            self._collision_groups = pairs
            try:
                self.collision_limit = mink.CollisionAvoidanceLimit(
                    self.model,
                    pairs,
                    gain=float(cfg.get("gain", 0.5)),
                    minimum_distance_from_collisions=float(cfg.get("minimum_distance", 0.01)),
                    collision_detection_distance=float(cfg.get("activation_distance", 0.05)),
                    bound_relaxation=float(cfg.get("bound_relaxation", 0.0)),
                )
                if not self.collision_limit.geom_id_pairs:
                    self.collision_limit = None
                    print("[V3] collision limit disabled: no valid geom pairs")
                else:
                    self._collision_geom_pairs = list(self.collision_limit.geom_id_pairs)
            except Exception as exc:
                print(f"[V3] collision limit disabled: {type(exc).__name__}: {exc}")

    def _build_v3_arm_tasks(self):
        self._arm_plane_tasks = {}
        for side in ("left", "right"):
            frames = [
                f"SHOULDER_YAW_{side[0].upper()}_LINK",
                f"ELBOW_PITCH_{side[0].upper()}_LINK",
                f"HAND_YAW_{side[0].upper()}_LINK",
            ]
            try:
                body_ids = [self.model.body(frame).id for frame in frames]
            except Exception:
                continue
            self._arm_plane_tasks[side] = ArmPlaneTask(
                self.model,
                body_ids,
                cost=float(self.v3_config.get("arms", {}).get("pole_vector_weight", 4.0)),
            )

    def update_targets(self, human_data, offset_to_ground=False):
        table1, table2 = self._prepare_target_data(human_data)
        self.scaled_human_data = table1
        for body_name, task in self.human_body_to_task1.items():
            if body_name in table1:
                pos, rot = table1[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
        for body_name, task in self.human_body_to_task2.items():
            if body_name in table2:
                pos, rot = table2[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
        graph_positions = np.asarray([table1[name][0] for name in self.graph_node_names], dtype=float)
        self.graph_task.set_target_positions(graph_positions)

    def _foot_feature(self, human_data, side):
        key = f"{side}_foot"
        pos, quat = human_data[key]
        rot = Rotation.from_quat(np.asarray(quat, dtype=float), scalar_first=True)
        forward = rot.apply(np.array([1.0, 0.0, 0.0]))
        forward[2] = 0.0
        norm = np.linalg.norm(forward[:2])
        forward = forward / norm if norm > 1e-7 else np.array([1.0, 0.0, 0.0])
        pos = np.asarray(pos, dtype=float)
        heel = pos - 0.08 * forward
        toe = pos + 0.12 * forward
        return heel, toe, pos

    def _predict_liftoff(self, side, current, future_frames):
        heights = []
        centers = []
        knee_flexion = []
        for frame in future_frames[: int(self.v3_config.get("contact", {}).get("future_preview_frames", 8))]:
            if f"{side}_foot" in frame:
                heel, toe, center = self._foot_feature(frame, side)
                heights.append(min(heel[2], toe[2]))
                centers.append(center)
                try:
                    hip = np.asarray(frame[f"{side}_hip"][0], dtype=float)
                    knee = np.asarray(frame[f"{side}_knee"][0], dtype=float)
                    upper = hip - knee
                    lower = center - knee
                    cosine = np.dot(upper, lower) / max(np.linalg.norm(upper) * np.linalg.norm(lower), 1e-8)
                    knee_flexion.append(np.pi - np.arccos(np.clip(cosine, -1.0, 1.0)))
                except KeyError:
                    pass
        if not heights:
            return 0.0
        rise = max(0.0, float(heights[-1] - heights[0]))
        positive = float(np.mean(np.diff(heights) > 0.001)) if len(heights) > 1 else 0.0
        xy_motion = 0.0
        if len(centers) > 1:
            xy_motion = float(max(np.linalg.norm(np.diff(np.asarray(centers)[:, :2], axis=0), axis=1)) / max(self.motion_dt, 1e-6))
        knee_change = max(0.0, float(knee_flexion[-1] - knee_flexion[0])) if len(knee_flexion) > 1 else 0.0
        opposite = "right_foot" if side == "left" else "left_foot"
        opposite_support = self._contact_states[opposite].load_phase in (
            FootLoadPhase.LOADING,
            FootLoadPhase.SUPPORT,
        )
        score = (
            6.0 * rise
            + 0.45 * positive
            + 0.25 * np.clip(xy_motion / 0.5, 0.0, 1.0)
            + 0.20 * np.clip(knee_change / 0.35, 0.0, 1.0)
            + 0.10 * float(opposite_support)
        )
        return float(np.clip(score, 0.0, 1.0))

    def _freeze_contact_snapshot(self, human_data, future_frames, q_start):
        contact_cfg = self.v3_config.get("contact", {})
        result = {}
        for side in ("left", "right"):
            name = f"{side}_foot"
            if name not in human_data or name not in self._v3_foot_normal:
                continue
            heel, toe, center = self._foot_feature(human_data, side)
            history = self._human_foot_history[name]
            dt = max(float(self.motion_dt), 1e-6)
            xy_speed = 0.0
            vz = 0.0
            if history:
                previous = history[-1]
                delta = center - previous
                xy_speed = float(np.linalg.norm(delta[:2]) / dt)
                vz = float(delta[2] / dt)
            ground = float(self.ground[2])
            heel_h = float(heel[2] - ground)
            toe_h = float(toe[2] - ground)
            min_h = min(heel_h, toe_h)
            on = float(contact_cfg.get("height_on", 0.018))
            off = float(contact_cfg.get("height_off", 0.035))
            vz_on = float(contact_cfg.get("vz_on", 0.08))
            vz_off = float(contact_cfg.get("vz_off", 0.16))
            previous_state = self._contact_states[name]
            near = float(np.clip((off - min_h) / max(off, 1e-6), 0.0, 1.0))
            slow = float(np.clip(1.0 - abs(vz) / max(vz_off, 1e-6), 0.0, 1.0))
            xy_slow = float(np.clip(1.0 - xy_speed / max(float(contact_cfg.get("xy_speed_off", 0.15)), 1e-6), 0.0, 1.0))
            confidence_obs = near * slow * xy_slow
            tau = float(contact_cfg.get("filter_tau", 0.08))
            alpha = 1.0 - np.exp(-dt / max(tau, 1e-6))
            confidence = alpha * confidence_obs + (1.0 - alpha) * previous_state.contact_confidence
            liftoff_observation = self._predict_liftoff(side, center, future_frames)
            liftoff_tau = float(contact_cfg.get("liftoff_filter_tau", 0.06))
            liftoff_alpha = 1.0 - np.exp(-dt / max(liftoff_tau, 1e-6))
            liftoff = (
                liftoff_alpha * liftoff_observation
                + (1.0 - liftoff_alpha) * previous_state.liftoff_probability
            )

            if min_h > off or vz > vz_off:
                observed_phase = FootContactPhase.AIR
            elif heel_h + 0.012 < toe_h:
                observed_phase = FootContactPhase.HEEL
            elif toe_h + 0.012 < heel_h:
                observed_phase = FootContactPhase.TOE
            else:
                observed_phase = FootContactPhase.FLAT
            if min_h < on and abs(vz) < vz_on and confidence > 0.25:
                observed_load = FootLoadPhase.SUPPORT if liftoff < 0.35 else FootLoadPhase.UNLOADING
            elif liftoff > 0.45 or min_h > off:
                observed_load = FootLoadPhase.UNLOADING
            elif min_h < off:
                observed_load = FootLoadPhase.LOADING
            else:
                observed_load = FootLoadPhase.UNLOADED

            near_contact = observed_phase != FootContactPhase.AIR
            on_counter = previous_state.on_counter + 1 if near_contact else 0
            off_counter = previous_state.off_counter + 1 if not near_contact else 0
            phase = observed_phase
            min_on_frames = int(contact_cfg.get("min_on_frames", 3))
            min_off_frames = int(contact_cfg.get("min_off_frames", 3))
            min_state_frames = int(contact_cfg.get("min_state_frames", 3))
            if previous_state.contact_phase == FootContactPhase.AIR and near_contact and on_counter < min_on_frames:
                phase = FootContactPhase.AIR
            elif previous_state.contact_phase != FootContactPhase.AIR and not near_contact and off_counter < min_off_frames:
                phase = previous_state.contact_phase
            elif (
                previous_state.contact_phase != FootContactPhase.AIR
                and observed_phase not in (FootContactPhase.AIR, previous_state.contact_phase)
                and previous_state.frames_in_state < min_state_frames
            ):
                phase = previous_state.contact_phase

            if phase == FootContactPhase.AIR:
                load = (
                    FootLoadPhase.UNLOADING
                    if previous_state.lock_weight > 0.03 or liftoff > 0.35
                    else FootLoadPhase.UNLOADED
                )
            else:
                load = observed_load
            if (
                previous_state.load_phase == FootLoadPhase.SUPPORT
                and load == FootLoadPhase.UNLOADING
                and liftoff < 0.45
                and near_contact
            ):
                load = FootLoadPhase.SUPPORT
            elif (
                previous_state.load_phase == FootLoadPhase.UNLOADING
                and load == FootLoadPhase.SUPPORT
                and liftoff > 0.30
            ):
                load = FootLoadPhase.UNLOADING
            if (
                load != previous_state.load_phase
                and previous_state.frames_in_state < min_state_frames
                and previous_state.frames_in_state > 0
            ):
                load = previous_state.load_phase
            desired_lock = float(np.clip(confidence * (1.0 - liftoff), 0.0, 1.0))
            anchor = previous_state.anchor_xy
            anchor_strain = 0.0
            if anchor is not None:
                anchor_strain = float(
                    np.linalg.norm(
                        self._v3_foot_anchor[name]._point(self.configuration)[:2]
                        - np.asarray(anchor, dtype=float)
                    )
                )
                sigma = float(contact_cfg.get("feasibility_sigma", 0.05))
                desired_lock *= float(np.exp(-((anchor_strain / max(sigma, 1e-6)) ** 2)))
                forced_threshold = float(contact_cfg.get("forced_release_anchor_error", 0.08))
                opposite = "right_foot" if name == "left_foot" else "left_foot"
                opposite_support = self._contact_states[opposite].load_phase in (
                    FootLoadPhase.LOADING,
                    FootLoadPhase.SUPPORT,
                )
                if anchor_strain > forced_threshold and (liftoff > 0.2 or opposite_support):
                    load = FootLoadPhase.UNLOADING
                    desired_lock = 0.0
            if load == FootLoadPhase.SUPPORT:
                desired_lock = max(desired_lock, 0.45)
            frames = previous_state.frames_in_state + 1
            if phase != previous_state.contact_phase or load != previous_state.load_phase:
                frames = 1
            smooth_frames = int(contact_cfg.get("loading_frames", 5) if desired_lock > previous_state.lock_weight else contact_cfg.get("unloading_frames", 7))
            blend = min(1.0, 1.0 / max(smooth_frames, 1))
            lock_weight = previous_state.lock_weight + blend * (desired_lock - previous_state.lock_weight)
            if anchor is None and lock_weight > 0.35:
                anchor = self._v3_foot_anchor[name]._point(self.configuration)[:2].copy()
            if lock_weight < 0.03:
                anchor = None
            if phase == FootContactPhase.HEEL:
                active = (0, 1)
            elif phase == FootContactPhase.TOE:
                active = (2, 3)
            else:
                active = tuple(range(len(self._v3_foot_normal[name].geom_ids)))
            result[name] = ContactSnapshot(
                phase=phase,
                load=load,
                confidence=float(confidence),
                liftoff=float(liftoff),
                lock_weight=float(lock_weight),
                anchor_xy=None if anchor is None else np.asarray(anchor, dtype=float).copy(),
                active_indices=tuple(active),
                heel_height=heel_h,
                toe_height=toe_h,
                xy_speed=xy_speed,
                vertical_speed=vz,
                frames_in_state=frames,
                on_counter=on_counter,
                off_counter=off_counter,
            )
        return result

    def _configure_contact_tasks(self, snapshot):
        for name, state in snapshot.items():
            support = state.load in (FootLoadPhase.LOADING, FootLoadPhase.SUPPORT)
            self._v3_foot_normal[name].set_state(
                state.lock_weight if support else 0.0,
                state.active_indices if support else tuple(range(len(self._v3_foot_normal[name].geom_ids))),
            )
            self._v3_foot_anchor[name].set_target(
                state.anchor_xy if state.anchor_xy is not None else np.zeros(2),
                state.lock_weight if support else 0.0,
            )
            clearance_cfg = self.v3_config.get("swing", {})
            clearance = float(np.clip(
                clearance_cfg.get("clearance_min", 0.025) + 0.5 * state.liftoff * clearance_cfg.get("clearance_max", 0.10),
                clearance_cfg.get("clearance_min", 0.025),
                clearance_cfg.get("clearance_max", 0.10),
            ))
            self._v3_swing_clearance[name].set_state(
                1.0 if not support else 0.0,
                tuple(range(len(self._v3_swing_clearance[name].geom_ids))),
                clearance=clearance,
            )

    def _set_arm_targets(self, human_data):
        for side, task in self._arm_plane_tasks.items():
            try:
                s = np.asarray(human_data[f"{side}_shoulder"][0], dtype=float)
                e = np.asarray(human_data[f"{side}_elbow"][0], dtype=float)
                w = np.asarray(human_data[f"{side}_wrist"][0], dtype=float)
                normal = np.cross(e - s, w - e)
                norm = np.linalg.norm(normal)
                scale = max(np.linalg.norm(e - s) * np.linalg.norm(w - e), 1e-8)
                bend_sine = float(np.clip(norm / scale, 0.0, 1.0))
                threshold = float(self.v3_config.get("arms", {}).get("singularity_threshold", 0.12))
                activation = np.clip(
                    (bend_sine - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0
                )
                task.cost[:] = task.base_cost * max(float(activation), 0.05)
                if bend_sine > threshold:
                    target = normal / norm
                    if np.dot(target, task.target_normal) < 0.0:
                        target = -target
                    task.set_target_normal(target)
            except KeyError:
                continue

    def _temporal_tasks(self, q_start):
        tasks = []
        if self._q_prev is None:
            return tasks
        actuated = []
        for jid in range(self.model.njnt):
            jtype = self.model.jnt_type[jid]
            if jtype not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                continue
            actuated.append(int(self.model.jnt_qposadr[jid]))
        if not actuated:
            return tasks
        q_prev = self._q_prev[actuated]
        dt = max(float(self.motion_dt), 1e-6)
        velocity_target = q_prev
        acceleration_target = q_prev.copy()
        jerk_target = q_prev.copy()
        if self._v_prev is not None:
            acceleration_target += self._v_prev[actuated] * dt
            jerk_target = acceleration_target.copy()
        if self._a_prev is not None:
            jerk_target += self._a_prev[actuated] * dt * dt
        temporal = self.v3_config.get("temporal", {})
        group_scale = temporal.get("group_scale", {})
        cost_scale = np.ones(len(actuated), dtype=float)
        for row, qpos_index in enumerate(actuated):
            joint_id = int(np.searchsorted(self.model.jnt_qposadr, qpos_index, side="right") - 1)
            joint_name = (mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, joint_id) or "").upper()
            if "TORSO" in joint_name or "WAIST" in joint_name:
                group_name = "torso"
            elif "SHOULDER" in joint_name:
                group_name = "shoulder"
            elif "ELBOW" in joint_name:
                group_name = "elbow"
            elif "HAND" in joint_name or "WRIST" in joint_name:
                group_name = "wrist"
            else:
                group_name = "leg"
            cost_scale[row] = float(group_scale.get(group_name, 1.0))
        if float(temporal.get("velocity_weight", 0.02)) > 0:
            tasks.append(QposSubsetTask(self.model, actuated, velocity_target, float(temporal.get("velocity_weight", 0.02)) * cost_scale, gain=0.25))
        if float(temporal.get("acceleration_weight", 0.15)) > 0:
            tasks.append(QposSubsetTask(self.model, actuated, acceleration_target, float(temporal.get("acceleration_weight", 0.15)) * cost_scale, gain=0.3))
        if self._a_prev is not None and float(temporal.get("jerk_weight", 0.03)) > 0:
            tasks.append(QposSubsetTask(self.model, actuated, jerk_target, float(temporal.get("jerk_weight", 0.03)) * cost_scale, gain=0.2))
        root_cfg = self.v3_config.get("root", {})
        root_prediction = float(self._q_prev[2])
        if self._v_prev is not None and len(self._v_prev) > 2:
            root_prediction += float(self._v_prev[2] * dt)
        tasks.append(RootZTask(self.model, root_prediction, cost=float(root_cfg.get("regularization", 5.0))))
        return tasks

    def _stage_tasks(self, stage, snapshot, q_reference=None):
        frame_names = {
            str(self.task_frame_names.get(task, "")): task
            for task in self.primary_tasks
        }
        support = []
        detail = []
        for frame, task in frame_names.items():
            upper = frame.upper()
            if frame in ("base_link", "TORSO_LINK") or any(token in upper for token in ("HIP", "KNEE", "ANKLE", "TOE")):
                support.append(task)
            else:
                detail.append(task)
        tasks = support if stage == "A" else list(frame_names.values()) + [self.graph_task]
        if stage == "B":
            preserve_indices = [2]
            for jid in range(self.model.njnt):
                jtype = self.model.jnt_type[jid]
                if jtype in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                    name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, jid) or ""
                    if any(t in name.upper() for t in ("HIP", "KNEE", "ANKLE", "WAIST", "TORSO")):
                        preserve_indices.append(int(self.model.jnt_qposadr[jid]))
            if q_reference is not None and preserve_indices:
                tasks.append(QposSubsetTask(self.model, sorted(set(preserve_indices)), q_reference[sorted(set(preserve_indices))], cost=2.0, gain=0.35))
        tasks.extend(self._temporal_tasks(self.configuration.data.qpos.copy()))
        for name, state in snapshot.items():
            tasks.extend((self._v3_foot_normal[name], self._v3_foot_anchor[name], self._v3_swing_clearance[name]))
        if stage == "B":
            tasks.extend(self._arm_plane_tasks.values())
        return tasks

    def _stage_limits(self, q_start, q_current, trust_radius=None):
        limits = [mink.ConfigurationLimit(self.model)]
        mapping = {}
        for jid in range(self.model.njnt):
            jtype = self.model.jnt_type[jid]
            if jtype != mj.mjtJoint.mjJNT_HINGE:
                continue
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, jid)
            if name:
                qadr = int(self.model.jnt_qposadr[jid])
                used = abs(float(q_current[qadr] - q_start[qadr]))
                remaining = max(self._velocity_limit * self.motion_dt - used, 1e-9)
                if trust_radius is not None:
                    remaining = min(remaining, float(trust_radius))
                mapping[name] = np.array([remaining / max(self.motion_dt, 1e-6)])
        if mapping:
            limits.append(mink.VelocityLimit(self.model, mapping))
        if self.collision_limit is not None:
            limits.append(self.collision_limit)
        if self._v3_ground_limit is not None:
            limits.append(self._v3_ground_limit)
        return limits

    def _clip_total_budget(self, q_start, q_candidate):
        q = np.asarray(q_candidate, dtype=float).copy()
        max_delta = self._velocity_limit * self.motion_dt
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] != mj.mjtJoint.mjJNT_HINGE:
                continue
            qa = int(self.model.jnt_qposadr[jid])
            delta = float(q[qa] - q_start[qa])
            q[qa] = q_start[qa] + np.clip(delta, -max_delta, max_delta)
        root_cfg = self.v3_config.get("root", {})
        root_xy = float(root_cfg.get("xy_velocity_limit", 1.0)) * self.motion_dt
        root_z_velocity = float(root_cfg.get("z_velocity_limit", 0.25))
        root_z = root_z_velocity * self.motion_dt
        q[:2] = q_start[:2] + np.clip(q[:2] - q_start[:2], -root_xy, root_xy)
        dz = float(np.clip(q[2] - q_start[2], -root_z, root_z))
        if self._v_prev is not None:
            acceleration_limit = float(root_cfg.get("z_acceleration_limit", 2.5))
            previous_vz = float(self._v_prev[2])
            candidate_vz = dz / max(self.motion_dt, 1e-6)
            candidate_vz = np.clip(
                candidate_vz,
                previous_vz - acceleration_limit * self.motion_dt,
                previous_vz + acceleration_limit * self.motion_dt,
            )
            candidate_vz = float(np.clip(candidate_vz, -root_z_velocity, root_z_velocity))
            dz = candidate_vz * self.motion_dt
        q[2] = q_start[2] + dz
        start_quat = np.asarray(q_start[3:7], dtype=float)
        candidate_quat = np.asarray(q[3:7], dtype=float)
        start_quat /= max(np.linalg.norm(start_quat), 1e-12)
        candidate_quat /= max(np.linalg.norm(candidate_quat), 1e-12)
        if np.dot(candidate_quat, start_quat) < 0.0:
            candidate_quat *= -1.0
        start_rotation = Rotation.from_quat(start_quat, scalar_first=True)
        candidate_rotation = Rotation.from_quat(candidate_quat, scalar_first=True)
        relative_rotvec = (candidate_rotation * start_rotation.inv()).as_rotvec()
        relative_angle = float(np.linalg.norm(relative_rotvec))
        max_root_angle = float(root_cfg.get("angular_velocity_limit", 3.0 * np.pi)) * self.motion_dt
        if relative_angle > max_root_angle and relative_angle > 1e-12:
            relative_rotvec *= max_root_angle / relative_angle
            candidate_rotation = Rotation.from_rotvec(relative_rotvec) * start_rotation
        q[3:7] = candidate_rotation.as_quat(scalar_first=True)
        return q

    def _interpolate_configuration(
        self, q_start, q_candidate, alpha, preserve_root_translation=False
    ):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        q = np.asarray(q_start, dtype=float) + alpha * (
            np.asarray(q_candidate, dtype=float) - np.asarray(q_start, dtype=float)
        )
        if preserve_root_translation:
            q[:3] = q_candidate[:3]
        start_rotation = Rotation.from_quat(q_start[3:7], scalar_first=True)
        candidate_rotation = Rotation.from_quat(q_candidate[3:7], scalar_first=True)
        relative = (candidate_rotation * start_rotation.inv()).as_rotvec()
        q[3:7] = (Rotation.from_rotvec(alpha * relative) * start_rotation).as_quat(
            scalar_first=True
        )
        return q

    def _configuration_is_safe(self, q):
        self.configuration.update(q)
        mj.mj_forward(self.model, self.configuration.data)
        min_foot_height = min(
            (
                float(np.min(task._heights(self.configuration)))
                for task in self._v3_foot_normal.values()
            ),
            default=float("inf"),
        )
        min_collision = self._min_collision_distance(q)
        collision_threshold = float(
            self.v3_config.get("collision", {}).get("minimum_distance", 0.01)
        )
        return min_foot_height >= -1e-5 and min_collision >= collision_threshold - 1e-5

    def _safety_backtrack(self, q_start, q_candidate):
        if self._configuration_is_safe(q_candidate):
            return np.asarray(q_candidate, dtype=float).copy(), False
        if not self._configuration_is_safe(q_start):
            return np.asarray(q_candidate, dtype=float).copy(), False
        # Self-collision does not depend on global root translation, and a
        # lowered root can become foot-safe once the leg interpolation has
        # progressed. Search the interval before allowing root-Z to be scaled.
        samples = np.linspace(0.0, 1.0, 17)
        safe_samples = []
        for alpha in samples:
            q_sample = self._interpolate_configuration(
                q_start, q_candidate, alpha, preserve_root_translation=True
            )
            if self._configuration_is_safe(q_sample):
                safe_samples.append(float(alpha))
        if safe_samples:
            low = max(safe_samples)
            higher = samples[samples > low]
            high = float(higher[0]) if len(higher) else 1.0
            for _ in range(12):
                middle = 0.5 * (low + high)
                q_middle = self._interpolate_configuration(
                    q_start, q_candidate, middle, preserve_root_translation=True
                )
                if self._configuration_is_safe(q_middle):
                    low = middle
                else:
                    high = middle
            return self._interpolate_configuration(
                q_start, q_candidate, low, preserve_root_translation=True
            ), True

        preserve_root_translation = False
        low, high = 0.0, 1.0
        for _ in range(14):
            middle = 0.5 * (low + high)
            q_middle = self._interpolate_configuration(
                q_start,
                q_candidate,
                middle,
                preserve_root_translation=preserve_root_translation,
            )
            if self._configuration_is_safe(q_middle):
                low = middle
            else:
                high = middle
        return self._interpolate_configuration(
            q_start,
            q_candidate,
            low,
            preserve_root_translation=preserve_root_translation,
        ), True

    def _task_error_norm(self, tasks, q):
        self.configuration.update(q)
        values = []
        for task in tasks:
            try:
                error = np.asarray(task.compute_error(self.configuration), dtype=float)
            except Exception:
                continue
            if error.size:
                values.append(float(np.linalg.norm(error) / np.sqrt(error.size)))
        return float(max(values, default=0.0))

    def _protect_stage_a(self, q_start, q_a, q_b, stage_a_tasks):
        baseline = self._task_error_norm(stage_a_tasks, q_a)
        candidate = self._task_error_norm(stage_a_tasks, q_b)
        pipeline = self.v3_config.get("solver_pipeline", {})
        tolerance = float(pipeline.get("stage_a_preserve_tolerance", 0.002))
        reject = bool(pipeline.get("reject_stage_b_on_high_priority_degradation", True))
        rejected = False
        if reject and candidate > baseline + tolerance:
            for blend in (0.5, 0.25, 0.125):
                q_try = self._clip_total_budget(q_start, q_a + blend * (q_b - q_a))
                error = self._task_error_norm(stage_a_tasks, q_try)
                if error <= baseline + tolerance:
                    q_b = q_try
                    candidate = error
                    rejected = True
                    break
            else:
                q_b = q_a.copy()
                candidate = baseline
                rejected = True
        self.configuration.update(q_b)
        return q_b, baseline, candidate, rejected

    def _solve_stage(self, q_initial, q_start, tasks, max_iterations, trust_radius=None, stage_name=""):
        q = np.asarray(q_initial, dtype=float).copy()
        for iteration in range(max(1, int(max_iterations))):
            if stage_name:
                self._stage_iterations[stage_name] = iteration + 1
            self.configuration.update(q)
            limits = self._stage_limits(q_start, q, trust_radius=trust_radius)
            try:
                velocity = mink.solve_ik(
                    self.configuration,
                    tasks,
                    self.motion_dt,
                    self.solver,
                    self.damping,
                    limits=limits,
                )
                self.configuration.integrate_inplace(velocity, self.motion_dt)
                q_next = self.configuration.data.qpos.copy()
                q_next = self._clip_total_budget(q_start, q_next)
                if np.linalg.norm(q_next - q) < 1e-6:
                    q = q_next
                    break
                q = q_next
            except Exception as exc:
                prefix = f"Stage {stage_name}: " if stage_name else ""
                self._frame_qp_status = f"{prefix}{type(exc).__name__}: {exc}"
                break
        self.configuration.update(q)
        return q

    def _min_collision_distance(self, q):
        if not self._collision_geom_pairs:
            self._last_collision_pair = ""
            return float("inf")
        self.configuration.update(q)
        mj.mj_forward(self.model, self.configuration.data)
        best = float("inf")
        best_pair = ""
        fromto = np.zeros(6, dtype=float)
        for geom_a, geom_b in self._collision_geom_pairs:
            distance = mj.mj_geomDistance(
                self.model,
                self.configuration.data,
                int(geom_a),
                int(geom_b),
                0.2,
                fromto,
            )
            if float(distance) < best:
                best = float(distance)
                name_a = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, int(geom_a)) or str(geom_a)
                name_b = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, int(geom_b)) or str(geom_b)
                best_pair = f"{name_a}:{name_b}"
        self._last_collision_pair = best_pair
        return best

    def _robot_foot_heights(self, name):
        task = self._v3_foot_normal.get(name)
        if task is None:
            return float("nan"), float("nan")
        heights = task._heights(self.configuration)
        return float(np.mean(heights[:2])), float(np.mean(heights[2:]))

    def _arm_branch_sign(self, side):
        task = self._arm_plane_tasks.get(side)
        if task is None:
            return 0
        normal = task._normal(self.configuration.data)
        try:
            torso = self.configuration.data.xmat[self.model.body("TORSO_LINK").id].reshape(3, 3)
            reference = torso[:, 1]
        except Exception:
            reference = np.array([0.0, 1.0, 0.0])
        value = float(np.dot(normal, reference))
        return int(np.sign(value)) if abs(value) > 1e-6 else 0

    def _record_diagnostics(
        self,
        q_start,
        q_final,
        snapshot,
        stage_a,
        stage_b,
        stage_c,
        stage_a_error,
        stage_b_error,
        stage_b_detail_error,
        stage_b_rejected,
        safety_backtracked,
    ):
        dt = max(float(self.motion_dt), 1e-6)
        root_dz = float(q_final[2] - q_start[2])
        root_ddz = 0.0
        if self._v_prev is not None:
            root_ddz = float(root_dz / dt - self._v_prev[2]) / dt
        root_angle = float(
            (
                Rotation.from_quat(q_final[3:7], scalar_first=True)
                * Rotation.from_quat(q_start[3:7], scalar_first=True).inv()
            ).magnitude()
        )
        min_collision_distance = self._min_collision_distance(q_final)
        record = {
            "frame_idx": self._frame_index,
            "root_z": float(q_final[2]),
            "root_dz": root_dz,
            "root_ddz": root_ddz,
            "max_joint_delta": float(np.max(np.abs(q_final[7:] - q_start[7:])) if len(q_final) > 7 else 0.0),
            "max_joint_velocity_ratio": float(np.max(np.abs(q_final[7:] - q_start[7:])) / max(self._velocity_limit * dt, 1e-9)) if len(q_final) > 7 else 0.0,
            "min_self_collision_distance": min_collision_distance,
            "active_collision_pair": self._last_collision_pair,
            "stage_c_triggered": int(stage_c is not None),
            "safety_backtracked": int(safety_backtracked),
            "frame_velocity_budget_usage_max": float(np.max(np.abs(q_final[7:] - q_start[7:])) / max(self._velocity_limit * self.motion_dt, 1e-9)) if len(q_final) > 7 else 0.0,
            "frame_root_z_budget_usage": abs(root_dz) / max(float(self.v3_config.get("root", {}).get("z_velocity_limit", 0.25)) * dt, 1e-9),
            "frame_root_angular_budget_usage": root_angle / max(float(self.v3_config.get("root", {}).get("angular_velocity_limit", 3.0 * np.pi)) * dt, 1e-9),
            "left_elbow_branch_sign": self._arm_branch_sign("left"),
            "right_elbow_branch_sign": self._arm_branch_sign("right"),
            "solver_mode": "staged_v3",
            "stage_a_high_priority_error": stage_a_error,
            "stage_b_high_priority_error": stage_b_error,
            "stage_b_rejected": int(stage_b_rejected),
            "stage_a_iterations": self._stage_iterations["A"],
            "stage_b_iterations": self._stage_iterations["B"],
            "stage_c_iterations": self._stage_iterations["C"],
            "stage_b_detail_error": stage_b_detail_error,
            "qp_status": self._frame_qp_status,
            "qp_resolve_count": 0,
        }
        for name, value in snapshot.items():
            prefix = "left" if name.startswith("left") else "right"
            robot_heel, robot_toe = self._robot_foot_heights(name)
            anchor_error = 0.0
            if value.anchor_xy is not None:
                anchor_error = float(np.linalg.norm(self._v3_foot_anchor[name]._point(self.configuration)[:2] - value.anchor_xy))
            clearance = float(self._v3_swing_clearance[name].clearance)
            record[f"{prefix}_contact_phase"] = value.phase.value
            record[f"{prefix}_load_phase"] = value.load.value
            record[f"{prefix}_contact_confidence"] = value.confidence
            record[f"{prefix}_liftoff_probability"] = value.liftoff
            record[f"{prefix}_lock_weight"] = value.lock_weight
            record[f"{prefix}_heel_height"] = robot_heel
            record[f"{prefix}_toe_height"] = robot_toe
            record[f"{prefix}_foot_vxy"] = value.xy_speed
            record[f"{prefix}_anchor_error_xy"] = anchor_error
            record[f"{prefix}_swing_clearance_error"] = max(0.0, clearance - min(robot_heel, robot_toe))
            record[f"{prefix}_contact_slack"] = max(0.0, -min(robot_heel, robot_toe))
        self.diagnostics.append(record)
        self._record_events(record, snapshot)

    def _record_events(self, record, snapshot):
        for name, value in snapshot.items():
            old = self._contact_states[name]
            side = "left" if name.startswith("left") else "right"
            if value.phase != old.contact_phase:
                event = "liftoff" if value.phase == FootContactPhase.AIR else "touchdown"
                self.events.append({"frame_idx": self._frame_index, "event": event, "side": side, "value": value.phase.value})
            if value.load != old.load_phase:
                self.events.append({"frame_idx": self._frame_index, "event": "load_transition", "side": side, "value": f"{old.load_phase.value}->{value.load.value}"})
        for side in ("left", "right"):
            sign = int(record[f"{side}_elbow_branch_sign"])
            previous = self._previous_branch_sign[side]
            if previous and sign and sign != previous:
                self.events.append({"frame_idx": self._frame_index, "event": "arm_branch_flip", "side": side, "value": sign})
            if sign:
                self._previous_branch_sign[side] = sign
        collision_now = float(record["min_self_collision_distance"]) < float(self.v3_config.get("collision", {}).get("activation_distance", 0.05))
        if collision_now and not self._collision_active:
            self.events.append({"frame_idx": self._frame_index, "event": "collision_constraint_activation", "side": "", "value": record["active_collision_pair"]})
        self._collision_active = collision_now
        if float(record["max_joint_velocity_ratio"]) >= 0.999:
            self.events.append({"frame_idx": self._frame_index, "event": "velocity_saturation", "side": "", "value": record["max_joint_velocity_ratio"]})
        if record["qp_status"] != "ok":
            self.events.append({"frame_idx": self._frame_index, "event": "qp_infeasible", "side": "", "value": record["qp_status"]})

    def retarget(self, human_data, future_frames=None, offset_to_ground=False):
        self._frame_qp_status = "ok"
        self._stage_iterations = {"A": 0, "B": 0, "C": 0}
        future_frames = list(future_frames or [human_data])
        self.update_targets(human_data, offset_to_ground)
        aligned_human_data = self.scaled_human_data
        aligned_future_frames = [
            self._prepare_target_data(frame)[0] for frame in future_frames
        ]
        q_start = self.configuration.data.qpos.copy()
        snapshot = self._freeze_contact_snapshot(
            aligned_human_data, aligned_future_frames, q_start
        )
        self._last_snapshot = snapshot
        self._configure_contact_tasks(snapshot)
        self._set_arm_targets(aligned_human_data)

        stage_a_tasks = self._stage_tasks("A", snapshot)
        q_a = self._solve_stage(
            q_start,
            q_start,
            stage_a_tasks,
            self.v3_config.get("solver_pipeline", {}).get("stage_a_max_iterations", 2),
            stage_name="A",
        )
        stage_b_tasks = self._stage_tasks("B", snapshot, q_reference=q_a)
        q_b = self._solve_stage(
            q_a,
            q_start,
            stage_b_tasks,
            self.v3_config.get("solver_pipeline", {}).get("stage_b_max_iterations", 2),
            trust_radius=self.v3_config.get("solver_pipeline", {}).get("stage_b_trust_region_rad", 0.12),
            stage_name="B",
        )
        q_b, stage_a_error, stage_b_error, stage_b_rejected = self._protect_stage_a(
            q_start, q_a, q_b, stage_a_tasks
        )
        stage_b_detail_error = self._task_error_norm(stage_b_tasks, q_b)
        stage_c = None
        safety_cfg = self.v3_config.get("solver_pipeline", {})
        min_distance = self._min_collision_distance(q_b)
        minimum_foot_height = min(
            (float(np.min(task._heights(self.configuration))) for task in self._v3_foot_normal.values()),
            default=float("inf"),
        )
        requires_projection = (
            min_distance < float(self.v3_config.get("collision", {}).get("minimum_distance", 0.01))
            or minimum_foot_height < -0.002
        )
        if bool(safety_cfg.get("enable_safety_projection", True)) and requires_projection:
            stage_c_tasks = [mink.PostureTask(self.model, cost=40.0)]
            stage_c_tasks[0].set_target(q_b)
            stage_c_tasks.extend(self._v3_foot_normal.values())
            stage_c_tasks.extend(self._v3_foot_anchor.values())
            stage_c = self._solve_stage(
                q_b,
                q_start,
                stage_c_tasks,
                safety_cfg.get("stage_c_max_iterations", 1),
                trust_radius=0.05,
                stage_name="C",
            )
            q_final = stage_c
        else:
            q_final = q_b
        q_final = self._clip_total_budget(q_start, q_final)
        q_final, safety_backtracked = self._safety_backtrack(q_start, q_final)
        self.configuration.update(q_final)
        mj.mj_forward(self.model, self.configuration.data)

        self._record_diagnostics(
            q_start,
            q_final,
            snapshot,
            q_a,
            q_b,
            stage_c,
            stage_a_error,
            stage_b_error,
            stage_b_detail_error,
            stage_b_rejected,
            safety_backtracked,
        )
        self._q_prev3 = self._q_prev2
        self._q_prev2 = self._q_prev
        self._q_prev = q_final.copy()
        if self._q_prev2 is not None:
            self._v_prev = (self._q_prev - self._q_prev2) / max(self.motion_dt, 1e-6)
        if self._q_prev3 is not None and self._q_prev2 is not None:
            self._a_prev = (self._q_prev - 2.0 * self._q_prev2 + self._q_prev3) / max(self.motion_dt * self.motion_dt, 1e-6)
        for side in ("left", "right"):
            name = f"{side}_foot"
            if name in aligned_human_data:
                self._human_foot_history[name].append(self._foot_feature(aligned_human_data, side)[2].copy())
                self._human_foot_history[name] = self._human_foot_history[name][-3:]
            if name in snapshot:
                old = self._contact_states[name]
                value = snapshot[name]
                self._contact_states[name] = FootContactState(
                    contact_phase=value.phase,
                    load_phase=value.load,
                    contact_confidence=value.confidence,
                    liftoff_probability=value.liftoff,
                    lock_weight=value.lock_weight,
                    anchor_xy=value.anchor_xy,
                    frames_in_state=value.frames_in_state,
                    last_transition_frame=self._frame_index if value.phase != old.contact_phase else old.last_transition_frame,
                    on_counter=value.on_counter,
                    off_counter=value.off_counter,
                )
        self._frame_index += 1
        return q_final.copy()

    def save_diagnostics(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.diagnostics:
            return
        fields = sorted({key for row in self.diagnostics for key in row})
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.diagnostics)
        if self.events and bool(self.v3_config.get("diagnostics", {}).get("save_event_log", True)):
            if path.name.endswith(".diagnostics.csv"):
                event_path = path.with_name(path.name[: -len(".diagnostics.csv")] + ".events.csv")
            else:
                event_path = path.with_suffix(".events.csv")
            event_fields = ("frame_idx", "event", "side", "value")
            with event_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=event_fields)
                writer.writeheader()
                writer.writerows(self.events)
