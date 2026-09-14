"""WholeBody V5 solver.

V5 owns its orchestration and QP construction.  It does not subclass or call
the historical V3/V4 retargeters.  The only deliberately shared pieces are
the canonical motion adapters, terrain geometry and the NE01 configuration
format.  This keeps the old pipelines available as regression baselines while
making the task-level data flow explicit.
"""

from __future__ import annotations

import json
import copy
import time
from pathlib import Path
from typing import Any

import mink
import mujoco as mj
import numpy as np
from mink.limits.limit import Constraint, Limit
from mink.tasks.task import Task
from scipy.spatial.transform import Rotation

from .core.schemas import ContactPlan, RetargetResult, RetargetTask
from .robot_profile import RobotProfile, is_dof_joint_type
from .task_builder import TaskBuilder
from .terrain_geometry import TerrainField
from .v5_terrain_limit import TerrainNonPenetrationLimit


def _named_id(model: mj.MjModel, object_type: Any, name: str) -> int:
    """MuJoCo-version independent named-object lookup."""
    value = mj.mj_name2id(model, object_type, str(name))
    if value < 0:
        raise KeyError(f"MuJoCo object does not exist: {name}")
    return int(value)


def _joint_id(model: mj.MjModel, name: str) -> int:
    return _named_id(model, mj.mjtObj.mjOBJ_JOINT, name)


def _joint_qposadr(model: mj.MjModel, name: str) -> int:
    return int(np.asarray(model.jnt_qposadr[_joint_id(model, name)]).reshape(-1)[0])


def _joint_dofadr(model: mj.MjModel, name: str) -> int:
    return int(np.asarray(model.jnt_dofadr[_joint_id(model, name)]).reshape(-1)[0])


def _interpolate_qpos(
    model: mj.MjModel,
    start_qpos: np.ndarray,
    end_qpos: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Interpolate configurations on MuJoCo's joint manifold.

    Raw array interpolation is invalid for free/ball-joint quaternions.  The
    difference/integration pair follows the same manifold convention as Mink
    and is therefore suitable for collision line searches.
    """
    start = np.asarray(start_qpos, dtype=float).reshape(model.nq)
    end = np.asarray(end_qpos, dtype=float).reshape(model.nq)
    value = float(np.clip(fraction, 0.0, 1.0))
    tangent = np.zeros(model.nv, dtype=float)
    mj.mj_differentiatePos(model, tangent, 1.0, start, end)
    result = start.copy()
    mj.mj_integratePos(model, result, tangent, value)
    return result


def _load_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if not raw.get("extends"):
        return raw
    parent = _load_config((path.parent / raw["extends"]).resolve())
    result = dict(parent)
    for key, value in raw.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            nested = _deep_merge(result[key], value); result[key] = nested
        else:
            result[key] = value
    return result


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class _Point:
    def __init__(self, model: mj.MjModel, spec: dict[str, Any]):
        self.model = model
        self.spec = dict(spec)
        self.sites = tuple(str(x) for x in self.spec.get("sites", ()))
        self.body_name = self.spec.get("robot_body", self.spec.get("body"))
        raw_offset = self.spec.get("robot_offset", self.spec.get("offset", self.spec.get("offsets", [[0, 0, 0]])))
        raw_offset = np.asarray(raw_offset, dtype=float)
        self.offset = raw_offset.reshape((-1, 3)).mean(axis=0)
        if self.sites:
            self.site_ids = tuple(int(model.site(name).id) for name in self.sites)
        elif self.body_name is not None:
            self.body_id = int(model.body(str(self.body_name)).id)
        else:
            raise ValueError("Robot point requires sites or robot_body")

    def value(self, configuration: mink.Configuration) -> np.ndarray:
        configuration.update()
        if self.sites:
            return np.mean([configuration.data.site_xpos[i] for i in self.site_ids], axis=0)
        rotation = configuration.data.xmat[self.body_id].reshape(3, 3)
        return configuration.data.xpos[self.body_id] + rotation @ self.offset

    def jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        configuration.update()
        if self.sites:
            jac = np.zeros((3, self.model.nv), dtype=float)
            for site_id in self.site_ids:
                jp = np.zeros_like(jac); jr = np.zeros_like(jac)
                mj.mj_jacSite(self.model, configuration.data, jp, jr, site_id)
                jac += jp
            return jac / len(self.site_ids)
        point = self.value(configuration)
        jp = np.zeros((3, self.model.nv), dtype=float); jr = np.zeros_like(jp)
        mj.mj_jac(self.model, configuration.data, jp, jr, point, self.body_id)
        return jp


class _InteractionTask(Task):
    """Fixed-topology semantic interaction graph with asset points as anchors."""
    def __init__(self, model, specs, environment_pool, config):
        self.model = model
        self.points = {name: _Point(model, spec["robot"] if "robot" in spec else spec)
                       for name, spec in specs.items()}
        self.names = list(self.points)
        pool = np.asarray(environment_pool, dtype=float).reshape((-1, 3))
        count = int(config.get("environment_points", min(64, len(pool))))
        self.environment_count = min(max(4, count), len(pool)) if len(pool) else 0
        self.pool = pool
        self.environment = pool[:self.environment_count].copy()
        self._topology_ready = False
        self.laplacian = np.zeros((len(self.names) + self.environment_count,) * 2)
        self.target = np.zeros((len(self.names) + self.environment_count, 3))
        cost = float(config.get("semantic_cost", 1.0))
        env_cost = float(config.get("environment_cost", 0.12))
        super().__init__(cost=np.repeat(np.r_[np.full(len(self.names), cost), np.full(self.environment_count, env_cost)], 3), gain=float(config.get("gain", .35)), lm_damping=1.0)

    @staticmethod
    def _graph(vertices: np.ndarray, k: int = 4) -> np.ndarray:
        n = len(vertices); graph = np.zeros((n, n), dtype=float)
        if n < 2: return graph
        distances = np.linalg.norm(vertices[:, None] - vertices[None, :], axis=-1)
        for i in range(n):
            neighbors = np.argsort(distances[i], kind="stable")[1:min(k + 1, n)]
            graph[i, i] = 1.0
            graph[i, neighbors] = -1.0 / len(neighbors)
        return graph

    def set_target(self, source: dict[str, np.ndarray]) -> None:
        human = np.asarray([source[name] for name in self.names], dtype=float)
        if self.environment_count and not self._topology_ready:
            # Select scene anchors exactly once.  Re-selecting nearest points
            # every frame (or every QP pass) changes the Laplacian topology and
            # creates artificial impulses in the pelvis and legs.
            distances = np.linalg.norm(self.pool[:, None] - human[None, :], axis=-1).min(axis=1)
            selected = np.argsort(distances, kind="stable")[:self.environment_count]
            self.environment = self.pool[selected].copy()
        vertices = np.vstack((human, self.environment))
        if not self._topology_ready:
            self.laplacian = self._graph(vertices)
            self._topology_ready = True
        self.target = self.laplacian @ vertices

    def _current(self, configuration):
        robot = np.asarray([self.points[name].value(configuration) for name in self.names])
        jac = [self.points[name].jacobian(configuration) for name in self.names]
        return np.vstack((robot, self.environment)), jac

    def compute_error(self, configuration):
        vertices, _ = self._current(configuration)
        return (self.laplacian @ vertices - self.target).reshape(-1)

    def compute_jacobian(self, configuration):
        _, jac = self._current(configuration)
        vertex_jac = np.zeros((3 * len(self.laplacian), self.model.nv))
        for i, value in enumerate(jac): vertex_jac[3*i:3*i+3] = value
        return np.kron(self.laplacian, np.eye(3)) @ vertex_jac


class _BoneDirectionTask(Task):
    """Preserve directed limb vectors and remove mirrored IK branches."""

    def __init__(self, model, robot_points, edges, cost=1.5, gain=.3):
        self.model = model
        self.robot_points = robot_points
        self.edges = [tuple(edge) for edge in edges
                      if edge[0] in robot_points and edge[1] in robot_points]
        self.sources = {}
        super().__init__(cost=np.full(3 * len(self.edges), float(cost)),
                         gain=float(gain), lm_damping=1.0)

    def set_source(self, source):
        self.sources = source

    @staticmethod
    def _unit(value):
        length = float(np.linalg.norm(value))
        return value / max(length, 1e-9), length

    def _edge(self, configuration, first, second):
        pa = self.robot_points[first].value(configuration)
        pb = self.robot_points[second].value(configuration)
        ja = self.robot_points[first].jacobian(configuration)
        jb = self.robot_points[second].jacobian(configuration)
        current, length = self._unit(pb - pa)
        target, _ = self._unit(
            np.asarray(self.sources[second]) - np.asarray(self.sources[first])
        )
        projector = np.eye(3) - np.outer(current, current)
        jacobian = projector @ ((jb - ja) / max(length, 1e-9))
        return current, target, jacobian

    def compute_error(self, configuration):
        if not self.edges:
            return np.empty(0)
        return np.concatenate([
            current - target
            for edge in self.edges
            for current, target, _ in (self._edge(configuration, *edge),)
        ])

    def compute_jacobian(self, configuration):
        if not self.edges:
            return np.empty((0, self.model.nv))
        return np.vstack([
            self._edge(configuration, *edge)[2] for edge in self.edges
        ])


class _LimbPlaneTask(Task):
    """Weak anatomical plane constraint selecting the natural knee/elbow side."""

    def __init__(self, model, robot_points, triples, cost=.6, gain=.2):
        self.model = model
        self.robot_points = robot_points
        self.triples = [tuple(triple) for triple in triples
                        if all(name in robot_points for name in triple)]
        self.sources = {}
        self.previous_targets = {}
        super().__init__(cost=np.full(3 * len(self.triples), float(cost)),
                         gain=float(gain), lm_damping=1.0)

    def set_source(self, source):
        self.sources = source

    @staticmethod
    def _normal(first, second):
        cross = np.cross(first, second)
        length = float(np.linalg.norm(cross))
        return cross / max(length, 1e-9), length

    def _triple(self, configuration, triple):
        a, b, c = triple
        pa, pb, pc = (self.robot_points[name].value(configuration)
                      for name in triple)
        ja, jb, jc = (self.robot_points[name].jacobian(configuration)
                      for name in triple)
        robot_normal, robot_norm = self._normal(pb - pa, pc - pa)
        target_normal, target_norm = self._normal(
            np.asarray(self.sources[b]) - np.asarray(self.sources[a]),
            np.asarray(self.sources[c]) - np.asarray(self.sources[a]),
        )
        previous = self.previous_targets.get(triple)
        if previous is not None and float(target_normal @ previous) < 0.0:
            target_normal = -target_normal
        if target_norm > 1e-8 and np.isfinite(target_normal).all():
            self.previous_targets[triple] = target_normal.copy()
        # Fade out the plane when either source or robot limb is nearly
        # straight, where a normal is ill-conditioned.
        source_u = np.asarray(self.sources[b]) - np.asarray(self.sources[a])
        source_v = np.asarray(self.sources[c]) - np.asarray(self.sources[a])
        source_scale = max(float(np.linalg.norm(source_u) * np.linalg.norm(source_v)), 1e-12)
        robot_scale = max(float(np.linalg.norm(pb - pa) * np.linalg.norm(pc - pa)), 1e-12)
        confidence = min(
            np.clip((robot_norm / robot_scale - .08) / .12, 0., 1.),
            np.clip((target_norm / source_scale - .08) / .12, 0., 1.),
        )
        cross_j = (
            np.cross(jb - ja, (pc - pa)[:, None], axis=0)
            + np.cross((pb - pa)[:, None], jc - ja, axis=0)
        )
        projector = (np.eye(3) - np.outer(robot_normal, robot_normal)) / max(robot_norm, 1e-9)
        return robot_normal, target_normal, float(confidence), projector @ cross_j

    def compute_error(self, configuration):
        if not self.triples:
            return np.empty(0)
        return np.concatenate([
            confidence * (robot_normal - target_normal)
            for triple in self.triples
            for robot_normal, target_normal, confidence, _ in
            (self._triple(configuration, triple),)
        ])

    def compute_jacobian(self, configuration):
        if not self.triples:
            return np.empty((0, self.model.nv))
        return np.vstack([
            confidence * jacobian
            for triple in self.triples
            for _, _, confidence, jacobian in
            (self._triple(configuration, triple),)
        ])


class _RootTask(Task):
    def __init__(self, model, body_name, cost):
        self.model = model; self.body_id = int(model.body(body_name).id)
        self.target_position = np.zeros(3)
        self.target_rotation = np.eye(3)
        raw_cost = np.asarray(cost, dtype=float).reshape(-1)
        if raw_cost.size == 4:
            # Preserve the historical [x, y, z, yaw] config shape while
            # giving the floating base explicit roll/pitch stabilization.
            raw_cost = np.r_[raw_cost[:3], np.repeat(raw_cost[3], 3)]
        elif raw_cost.size == 3:
            raw_cost = np.r_[raw_cost, np.full(3, 1.0)]
        if raw_cost.size != 6:
            raise ValueError("V5 global_anchor.cost must contain 3 position and 3 orientation costs")
        super().__init__(cost=raw_cost, gain=.45, lm_damping=1.0)

    def set_target(self, position, quaternion):
        self.target_position = np.asarray(position, dtype=float)
        quat = np.asarray(quaternion, dtype=float).reshape(4)
        quat /= max(float(np.linalg.norm(quat)), 1e-12)
        self.target_rotation = Rotation.from_quat(quat, scalar_first=True).as_matrix()

    def compute_error(self, configuration):
        p = configuration.data.xpos[self.body_id]
        current = configuration.data.xmat[self.body_id].reshape(3, 3)
        # Express orientation error in the target frame.  This closes the
        # otherwise unbounded floating-base pitch/roll null mode while still
        # allowing the articulated waist to express torso motion.
        error = Rotation.from_matrix(self.target_rotation.T @ current).as_rotvec()
        return np.r_[p - self.target_position, error]

    def compute_jacobian(self, configuration):
        jp = np.zeros((3, self.model.nv)); jr = np.zeros_like(jp)
        mj.mj_jacBody(self.model, configuration.data, jp, jr, self.body_id)
        current = configuration.data.xmat[self.body_id].reshape(3, 3)
        phi = Rotation.from_matrix(self.target_rotation.T @ current).as_rotvec()
        theta = float(np.linalg.norm(phi))
        hat = np.array([[0.0, -phi[2], phi[1]],
                        [phi[2], 0.0, -phi[0]],
                        [-phi[1], phi[0], 0.0]])
        if theta < 1e-5:
            right_jacobian_inv = np.eye(3) + 0.5 * hat + (hat @ hat) / 12.0
        else:
            half = 0.5 * theta
            denom = max(2.0 * theta * np.sin(theta), 1e-9)
            coefficient = 1.0 / (theta * theta) - (1.0 + np.cos(theta)) / denom
            right_jacobian_inv = np.eye(3) + 0.5 * hat + coefficient * (hat @ hat)
        return np.vstack((jp, right_jacobian_inv @ self.target_rotation.T @ jr))


class _TorsoCoherenceTask:
    """Keep the two NE01 waist DoFs consistent with the source torso frame.

    The Omni Laplacian should express limb intent, not spend the free waist
    yaw on satisfying a distant wrist/foot edge.  A small posture task on the
    actual waist joints removes this otherwise weakly constrained null mode.
    """

    def __init__(self, model: mj.MjModel, config: dict[str, Any]) -> None:
        self.model = model
        self.alpha = 2.0 / (max(1, int(config.get("blend_frames", 7))) + 1.0)
        self.previous = {"yaw": None, "roll": None}
        self.joints = {
            "yaw": str(config.get("waist_yaw_joint", "WAIST_YAW_JOINT")),
            "roll": str(config.get("torso_roll_joint", "TORSO_ROLL_JOINT")),
        }
        costs = np.zeros(model.nv)
        for key, name in self.joints.items():
            if name in [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]:
                costs[_joint_dofadr(model, name)] = float(config.get(f"{key}_cost", config.get("cost", 0.25)))
        self.task = mink.PostureTask(
            model, costs, gain=float(config.get("gain", .4)), lm_damping=1.0
        )
        self.task.set_target(model.qpos0)

    @staticmethod
    def _wrap(value: float) -> float:
        return float((value + np.pi) % (2.0 * np.pi) - np.pi)

    def set_source(self, source: dict[str, np.ndarray], pelvis_quaternion: np.ndarray, qpos: np.ndarray) -> None:
        target = qpos.copy()
        if not {"left_shoulder", "right_shoulder", "pelvis"} <= set(source):
            self.task.set_target(target)
            return
        lateral = source["left_shoulder"] - source["right_shoulder"]
        up = .5 * (source["left_shoulder"] + source["right_shoulder"]) - source["pelvis"]
        lateral /= max(float(np.linalg.norm(lateral)), 1e-12)
        up /= max(float(np.linalg.norm(up)), 1e-12)
        forward = np.cross(lateral, up)
        if np.linalg.norm(forward) < 1e-8:
            self.task.set_target(target)
            return
        forward /= np.linalg.norm(forward)
        up = np.cross(forward, lateral)
        chest = Rotation.from_matrix(np.column_stack((forward, lateral, up)))
        pelvis = Rotation.from_quat(pelvis_quaternion, scalar_first=True)
        yaw, _, roll = (pelvis.inv() * chest).as_euler("zyx")
        for key, value in (("yaw", float(yaw)), ("roll", float(roll))):
            name = self.joints[key]
            if name not in [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]:
                continue
            joint_index = _joint_id(self.model, name)
            lower, upper = self.model.jnt_range[joint_index]
            value = float(np.clip(self._wrap(value), lower, upper))
            old = self.previous[key]
            if old is not None:
                value = float(np.clip(old + self.alpha * self._wrap(value - old), lower, upper))
            self.previous[key] = value
            target[_joint_qposadr(self.model, name)] = value
        self.task.set_target(target)


class _ContactTask(Task):
    def __init__(self, model, specs, config):
        self.points = {name: _Point(model, spec) for name, spec in specs.items()}
        self.contacts: dict[str, dict[str, Any]] = {}
        self.model = model
        self.clearance = float(config.get("clearance", .004))
        names = list(self.points); self.names = names
        self.normal_cost = float(config.get("normal_cost", 40.))
        self.tangent_cost = float(config.get("tangent_cost", 12.))
        self.static_transition_cost = float(
            config.get("static_transition_cost", self.tangent_cost)
        )
        # A sliding contact follows the source tangential trajectory with a
        # deliberately softer objective.  Leaving all tangential rows at
        # zero lets a foot drift off a stair tread while its normal distance
        # still looks valid; treating it as STATIC, on the other hand, locks
        # the foot to one world anchor.  This intermediate cost preserves the
        # intended surface point without over-constraining swing/transfer.
        self.sliding_tangent_ratio = float(config.get("sliding_tangent_ratio", 0.35))
        self.sliding_tangent_cost = float(config.get(
            "sliding_tangent_cost", self.tangent_cost * self.sliding_tangent_ratio
        ))
        self.normal_only = False
        # Support polish may temporarily give a source-confirmed heel/toe a
        # small minimum task weight while the detector's multi-frame ramp is
        # still entering. Ordinary IK leaves this at zero and uses the
        # detector activation unchanged.
        self.activation_floor = 0.0
        self.task_activation_floor = float(
            config.get("task_activation_floor", 0.20)
        )
        # Support validation uses the actual foot collision spheres rather
        # than the visual/contact site centres.  Keep this mapping inferred
        # from model names so another robot profile can provide its own
        # ``*_support_geoms`` configuration without changing the validator.
        configured_support = config.get("support_geoms", {})
        self.support_geom_ids = {}
        for side in ("left", "right"):
            for channel, region in ((f"{side}_heel", "rear"), (f"{side}_toe", "front")):
                geom_names = configured_support.get(channel)
                if geom_names is None:
                    geom_names = [
                        f"{side}_foot_{region}_left_collision",
                        f"{side}_foot_{region}_right_collision",
                    ]
                ids = []
                for name in geom_names:
                    geom_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, str(name)))
                    if geom_id >= 0:
                        ids.append(geom_id)
                self.support_geom_ids[channel] = tuple(ids)
        # Every channel has one normal and two tangent rows.  Keeping the row
        # layout fixed is important: changing task dimensions as contacts
        # enter/leave would make the QP warm-start and damping discontinuous.
        super().__init__(cost=np.zeros(3 * len(self.names)), gain=.5, lm_damping=1.0)

    def set_contacts(self, contacts): self.contacts = contacts or {}

    def support_value(self, configuration, channel: str, normal: np.ndarray) -> np.ndarray:
        """Return the lowest physical collision-surface point for a foot."""
        point, _ = self._support_point_and_geom(configuration, channel, normal)
        return point

    def tangent_value(self, configuration, channel: str) -> np.ndarray:
        """Return the stable physical feature used for tangential sticking.

        Normal support is measured at the lowest collision-sphere surface,
        which changes as the sole normal changes and as the minimum switches
        between the inner and outer sphere.  That is intentionally *not* a
        valid temporal anchor.  The configured contact proxy is fixed in the
        foot body frame and therefore owns all tangent anchors and drift
        measurements.
        """
        return self.points[channel].value(configuration)

    def tangent_jacobian(self, configuration, channel: str) -> np.ndarray:
        """Return the Jacobian of :meth:`tangent_value`."""
        return self.points[channel].jacobian(configuration)

    def _support_point_and_geom(self, configuration, channel: str, normal: np.ndarray):
        """Return the selected physical foot support point and its geom id.

        Foot contact tasks must use the same sphere surface that terrain
        non-penetration and final support replay measure.  The XML guard
        sites are useful for broad coverage, but they are not the physical
        shoe surface and can be several millimetres above/below it.
        """
        configuration.update()
        ids = self.support_geom_ids.get(str(channel), ())
        if not ids:
            return self.points[channel].value(configuration), None
        normal = np.asarray(normal, dtype=float).reshape(3)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        candidates = []
        for geom_id in ids:
            # Foot support geoms are spheres.  geom_rbound is a conservative
            # fallback for a profile that uses a capsule/cylinder proxy.
            radius = float(self.model.geom_size[geom_id, 0])
            if int(self.model.geom_type[geom_id]) != int(mj.mjtGeom.mjGEOM_SPHERE):
                radius = float(self.model.geom_rbound[geom_id])
            point = configuration.data.geom_xpos[geom_id] - radius * normal
            candidates.append((float(normal @ point), point.copy(), int(geom_id)))
        _, point, geom_id = min(candidates, key=lambda item: item[0])
        return point, geom_id

    def support_jacobian(self, configuration, channel: str, normal: np.ndarray):
        """Return the Jacobian of the selected physical support point."""
        point, geom_id = self._support_point_and_geom(configuration, channel, normal)
        if geom_id is None:
            return point, self.points[channel].jacobian(configuration)
        body_id = int(self.model.geom_bodyid[geom_id])
        position = np.zeros((3, self.model.nv))
        rotation = np.zeros_like(position)
        mj.mj_jac(
            self.model,
            configuration.data,
            position,
            rotation,
            point,
            body_id,
        )
        return point, position

    @staticmethod
    def _tangent_basis(normal: np.ndarray) -> np.ndarray:
        normal = np.asarray(normal, dtype=float)
        axis = np.array([0., 0., 1.])
        if abs(float(normal @ axis)) > .92:
            axis = np.array([1., 0., 0.])
        tangent_1 = axis - normal * float(axis @ normal)
        tangent_1 /= max(float(np.linalg.norm(tangent_1)), 1e-12)
        tangent_2 = np.cross(normal, tangent_1)
        tangent_2 /= max(float(np.linalg.norm(tangent_2)), 1e-12)
        return np.vstack((tangent_1, tangent_2))

    def _rows(self, configuration):
        errors = []
        jacobians = []
        weights = []
        for name in self.names:
            item = self.contacts.get(name, {})
            state = str(item.get("state", "NONE"))
            # STATIC episodes own one surface frame.  Reusing the per-frame
            # triangle hit here makes a tessellated chair seat move the
            # target under the robot and is the main source of seated drift.
            # SLIDING contacts intentionally follow the current surface hit.
            normal_key = "anchor_normal_solver" if state == "STATIC" and "anchor_normal_solver" in item else "surface_normal_solver"
            normal = np.asarray(item.get(normal_key, [0., 0., 1.]), dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            if name in self.support_geom_ids:
                point, jacobian = self.support_jacobian(configuration, name, normal)
                tangent_point = self.tangent_value(configuration, name)
                tangent_jacobian = self.tangent_jacobian(configuration, name)
            else:
                point = self.points[name].value(configuration)
                jacobian = self.points[name].jacobian(configuration)
                tangent_point = point
                tangent_jacobian = jacobian
            # A realized STATIC anchor is only a tangential sticking target.
            # Keep the normal target on the actual source surface so a
            # floating robot point cannot validate against itself.
            surface = np.asarray(item.get("surface_point_solver", [0., 0., 0.]), dtype=float)
            # ``state`` is the discrete diagnostic label; ``activation`` is
            # the frame-smoothed task weight supplied by the source contact
            # detector.  Keeping these separate avoids a hard QP objective
            # discontinuity when a heel or butt contact enters/leaves.
            confidence = float(item.get("activation", item.get("score", 0.0)))
            # Detector confidence is intentionally smoothed for hysteresis,
            # but it is not permission to keep attracting a released contact
            # to the surface.  NONE keeps only the independent terrain
            # non-penetration limit; all desired-contact rows are disabled.
            active = 0.0 if state == "NONE" else confidence
            if state != "NONE":
                active = max(active, self.task_activation_floor)
            if state != "NONE" and self.activation_floor > 0.0:
                active = max(active, self.activation_floor)
            tangent = self._tangent_basis(normal)
            # STATIC contacts stick to the episode anchor.  SLIDING contacts
            # retain normal contact but follow the currently observed tangent
            # point, so they cannot be accidentally locked to world XY.
            anchor = np.asarray(
                item.get("tangent_anchor_solver", item.get("surface_point_solver", surface)),
                dtype=float,
            )
            target = anchor if state == "STATIC" else surface
            # Contact targets and collision use the same physical clearance
            # convention.  The source surface lies at d=0; the configured
            # robot proxy is a centre/representative point and must stay one
            # clearance outside it in the free-space normal direction.
            # Omitting this term asked desired contact to enter the very hull
            # that AutomaticSceneCollisionLimit was simultaneously keeping
            # four millimetres away from.
            contact_target = surface + normal * self.clearance
            errors.extend((normal @ (point - contact_target),
                           *(tangent @ (tangent_point - target))))
            jacobians.extend((normal @ jacobian,
                              tangent[0] @ tangent_jacobian,
                              tangent[1] @ tangent_jacobian))
            tangent_weight = 0.0
            if not self.normal_only:
                if state == "STATIC":
                    if bool(item.get("robot_static_anchor_bound", True)):
                        transition = int(item.get("robot_static_anchor_age", 1)) <= 0
                        tangent_weight = (
                            self.static_transition_cost if transition else self.tangent_cost
                        ) * active
                elif state == "SLIDING":
                    tangent_weight = self.sliding_tangent_cost * active
            weights.extend((self.normal_cost * active, tangent_weight, tangent_weight))
        return np.asarray(errors), np.asarray(jacobians), np.asarray(weights)

    def compute_error(self, configuration):
        error, _, weights = self._rows(configuration)
        self.cost = weights
        return error

    def compute_jacobian(self, configuration):
        _, jacobian, weights = self._rows(configuration)
        self.cost = weights
        return np.asarray(jacobian)


class _FootNormalTask(Task):
    """Align a supporting sole normal with its shared terrain surface.

    The task has a fixed six-row topology (three rows per foot); inactive
    feet simply receive zero cost.  It only activates when that foot's heel
    and toe are both reliable contacts on the same surface, so a single-point
    toe or heel strike does not force an artificial flat-foot pose.
    """

    def __init__(self, model, config):
        self.model = model
        self.cost_value = float(config.get("cost", 0.25))
        self.activation_floor = float(config.get("activation_floor", 0.3))
        self.channels = (("left_heel", "left_toe", "ANKLE_ROLL_L_LINK"),
                         ("right_heel", "right_toe", "ANKLE_ROLL_R_LINK"))
        self.contacts = {}
        self.body_ids = []
        self.local_axes = []
        qpos = np.asarray(model.qpos0, dtype=float).copy()
        data = mj.MjData(model)
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        for _, _, body_name in self.channels:
            body_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name))
            if body_id < 0:
                self.body_ids.append(-1)
                self.local_axes.append(np.array([0., 0., 1.]))
                continue
            self.body_ids.append(body_id)
            world_axis = np.array([0., 0., 1.])
            self.local_axes.append(data.xmat[body_id].reshape(3, 3).T @ world_axis)
        super().__init__(cost=np.zeros(6), gain=float(config.get("gain", .35)), lm_damping=1.0)

    def set_contacts(self, contacts):
        self.contacts = contacts or {}

    @staticmethod
    def _normal(item):
        key = "anchor_normal_solver" if item.get("state") == "STATIC" else "surface_normal_solver"
        value = np.asarray(item.get(key, [0., 0., 1.]), dtype=float)
        return value / max(float(np.linalg.norm(value)), 1e-12)

    def _active_target(self, heel_name, toe_name):
        heel = self.contacts.get(heel_name, {})
        toe = self.contacts.get(toe_name, {})
        if heel.get("state", "NONE") == "NONE" or toe.get("state", "NONE") == "NONE":
            return None, 0.0
        if float(heel.get("activation", heel.get("score", 0.0))) < self.activation_floor:
            return None, 0.0
        if float(toe.get("activation", toe.get("score", 0.0))) < self.activation_floor:
            return None, 0.0
        if str(heel.get("surface_id", "")) != str(toe.get("surface_id", "")):
            return None, 0.0
        normal = self._normal(heel) + self._normal(toe)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        return normal, min(float(heel.get("activation", 0.0)), float(toe.get("activation", 0.0)))

    def _rows(self, configuration):
        configuration.update()
        errors, jacobians, costs = [], [], []
        for index, (heel, toe, _) in enumerate(self.channels):
            target, activation = self._active_target(heel, toe)
            body_id = self.body_ids[index]
            if body_id < 0:
                errors.extend((0., 0., 0.)); jacobians.extend((np.zeros(self.model.nv),) * 3); costs.extend((0., 0., 0.)); continue
            rotation = configuration.data.xmat[body_id].reshape(3, 3)
            current = rotation @ self.local_axes[index]
            angular = np.zeros((3, self.model.nv)); position = np.zeros_like(angular)
            mj.mj_jacBody(self.model, configuration.data, position, angular, body_id)
            if target is None:
                errors.extend((0., 0., 0.)); jacobians.extend((np.zeros(self.model.nv),) * 3); costs.extend((0., 0., 0.)); continue
            # d(current) = omega x current = -skew(current) omega.
            skew_current = np.array([[0., -current[2], current[1]], [current[2], 0., -current[0]], [-current[1], current[0], 0.]])
            skew_target = np.array([[0., -target[2], target[1]], [target[2], 0., -target[0]], [-target[1], target[0], 0.]])
            errors.extend(np.cross(current, target))
            # e = current x target, d(current) = omega x current.
            # Therefore de = [target]x [current]x omega.
            jacobian = skew_target @ skew_current @ angular
            jacobians.extend(tuple(jacobian[row] for row in range(3)))
            costs.extend((self.cost_value * activation,) * 3)
        return np.asarray(errors), np.asarray(jacobians), np.asarray(costs)

    def compute_error(self, configuration):
        error, _, costs = self._rows(configuration); self.cost = costs; return error

    def compute_jacobian(self, configuration):
        _, jacobian, costs = self._rows(configuration); self.cost = costs; return jacobian


class _FootMotionTask(Task):
    """Track source foot motion without flattening heel/toe roll phases.

    The forward axis is active whenever a valid foot-to-toe vector exists.
    The source sole normal is active only in flight; desired-contact normal
    alignment remains exclusively owned by ``_FootNormalTask`` when heel and
    toe are both on the same surface.
    """

    def __init__(self, model, config):
        self.model = model
        self.channels = (
            ("left", "ANKLE_ROLL_L_LINK"),
            ("right", "ANKLE_ROLL_R_LINK"),
        )
        self.body_ids = []
        self.local_forward = []
        self.local_normal = []
        self.targets: dict[str, dict[str, Any]] = {}
        self.previous_forward: dict[str, np.ndarray] = {}
        self.previous_normal: dict[str, np.ndarray] = {}
        self.swing_forward_cost = float(config.get("swing_forward_cost", 0.10))
        self.support_forward_cost = float(config.get("support_forward_cost", 0.08))
        self.swing_normal_cost = float(config.get("swing_normal_cost", 0.06))
        data = mj.MjData(model)
        data.qpos[:] = model.qpos0
        mj.mj_forward(model, data)
        for _, body_name in self.channels:
            body_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name))
            if body_id < 0:
                raise KeyError(f"V5 foot motion body does not exist: {body_name}")
            rotation = data.xmat[body_id].reshape(3, 3)
            self.body_ids.append(body_id)
            self.local_forward.append(rotation.T @ np.array([1.0, 0.0, 0.0]))
            self.local_normal.append(rotation.T @ np.array([0.0, 0.0, 1.0]))
        super().__init__(
            cost=np.zeros(12),
            gain=float(config.get("gain", 0.30)),
            lm_damping=1.0,
        )

    @staticmethod
    def _unit(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=float).reshape(3)
        length = float(np.linalg.norm(value))
        return value / length if length > 1e-8 else np.asarray(fallback, dtype=float)

    def set_target(self, source, contacts):
        lateral = self._unit(
            np.asarray(source["left_hip"]) - np.asarray(source["right_hip"]),
            np.array([0.0, 1.0, 0.0]),
        )
        for side, _ in self.channels:
            forward = self._unit(
                np.asarray(source[f"{side}_toe"])
                - np.asarray(source[f"{side}_foot"]),
                self.previous_forward.get(side, np.array([1.0, 0.0, 0.0])),
            )
            if side in self.previous_forward and float(
                forward @ self.previous_forward[side]
            ) < -0.25:
                forward = self.previous_forward[side].copy()
            normal = self._unit(
                np.cross(forward, lateral),
                self.previous_normal.get(side, np.array([0.0, 0.0, 1.0])),
            )
            if normal[2] < 0.0:
                normal *= -1.0
            if side in self.previous_normal and float(
                normal @ self.previous_normal[side]
            ) < 0.0:
                normal *= -1.0
            # Re-orthogonalize forward so its residual cannot ask the ankle
            # to satisfy an internally inconsistent source frame.
            forward = self._unit(
                forward - normal * float(forward @ normal),
                self.previous_forward.get(side, np.array([1.0, 0.0, 0.0])),
            )
            self.previous_forward[side] = forward.copy()
            self.previous_normal[side] = normal.copy()
            heel_state = str(contacts.get(f"{side}_heel", {}).get("state", "NONE"))
            toe_state = str(contacts.get(f"{side}_toe", {}).get("state", "NONE"))
            flight = heel_state == "NONE" and toe_state == "NONE"
            self.targets[side] = {
                "forward": forward,
                "normal": normal,
                "forward_cost": (
                    self.swing_forward_cost if flight else self.support_forward_cost
                ),
                "normal_cost": self.swing_normal_cost if flight else 0.0,
                "mode": (
                    "FLIGHT" if flight
                    else "HEEL" if heel_state != "NONE" and toe_state == "NONE"
                    else "TOE" if heel_state == "NONE" and toe_state != "NONE"
                    else "DOUBLE"
                ),
            }

    @staticmethod
    def _skew(value):
        x, y, z = np.asarray(value, dtype=float)
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

    def _rows(self, configuration):
        configuration.update()
        errors, jacobians, costs = [], [], []
        for index, (side, _) in enumerate(self.channels):
            target = self.targets.get(side)
            body_id = self.body_ids[index]
            position = np.zeros((3, self.model.nv))
            angular = np.zeros_like(position)
            mj.mj_jacBody(self.model, configuration.data, position, angular, body_id)
            rotation = configuration.data.xmat[body_id].reshape(3, 3)
            current_normal = rotation @ self.local_normal[index]
            current_forward = rotation @ self.local_forward[index]
            if target is None:
                errors.extend(np.zeros(6))
                jacobians.extend((np.zeros(self.model.nv),) * 6)
                costs.extend(np.zeros(6))
                continue
            target_normal = target["normal"]
            target_forward = target["forward"]
            errors.extend(np.cross(current_normal, target_normal))
            errors.extend(np.cross(current_forward, target_forward))
            normal_jacobian = (
                self._skew(target_normal) @ self._skew(current_normal) @ angular
            )
            forward_jacobian = (
                self._skew(target_forward) @ self._skew(current_forward) @ angular
            )
            jacobians.extend(tuple(normal_jacobian[row] for row in range(3)))
            jacobians.extend(tuple(forward_jacobian[row] for row in range(3)))
            costs.extend((float(target["normal_cost"]),) * 3)
            costs.extend((float(target["forward_cost"]),) * 3)
        return np.asarray(errors), np.asarray(jacobians), np.asarray(costs)

    def compute_error(self, configuration):
        error, _, costs = self._rows(configuration)
        self.cost = costs
        return error

    def compute_jacobian(self, configuration):
        _, jacobian, costs = self._rows(configuration)
        self.cost = costs
        return jacobian


class _FootTemporalTask(Task):
    """Suppress STATIC tangent velocity between exported 50 Hz frames."""

    CHANNELS = ("left_heel", "left_toe", "right_heel", "right_toe")

    def __init__(self, model, contact: _ContactTask, dt: float, config):
        self.model = model
        self.contact = contact
        self.dt = max(float(dt), 1e-9)
        self.cost_value = float(config.get("cost", 0.08))
        self.contacts: dict[str, dict[str, Any]] = {}
        self.previous_output: dict[str, np.ndarray] = {}
        self.frame_previous: dict[str, np.ndarray] = {}
        self.previous_static: dict[str, bool] = {}
        self.frame_active: dict[str, bool] = {}
        super().__init__(
            cost=np.zeros(8),
            gain=float(config.get("gain", 0.35)),
            lm_damping=1.0,
        )

    def begin_frame(self, configuration, contacts):
        self.contacts = contacts or {}
        self.frame_active = {}
        self.frame_previous = {
            name: self.previous_output.get(
                name, self.contact.tangent_value(configuration, name)
            ).copy()
            for name in self.CHANNELS
            if name in self.contact.points
        }
        for name in self.CHANNELS:
            item = self.contacts.get(name, {})
            self.frame_active[name] = bool(
                self.previous_static.get(name, False)
                and str(item.get("state", "NONE")) == "STATIC"
                and bool(item.get("robot_static_anchor_bound", False))
            )

    def end_frame(self, configuration):
        self.previous_output = {
            name: self.contact.tangent_value(configuration, name).copy()
            for name in self.CHANNELS
            if name in self.contact.points
        }
        self.previous_static = {
            name: bool(
                str(self.contacts.get(name, {}).get("state", "NONE")) == "STATIC"
                and self.contacts.get(name, {}).get("robot_static_anchor_bound", False)
            )
            for name in self.CHANNELS
        }

    def _rows(self, configuration):
        errors, jacobians, costs = [], [], []
        for name in self.CHANNELS:
            item = self.contacts.get(name, {})
            state = str(item.get("state", "NONE"))
            active = (
                self.frame_active.get(name, False)
                and name in self.frame_previous
            )
            normal = np.asarray(
                item.get("anchor_normal_solver", [0.0, 0.0, 1.0]),
                dtype=float,
            )
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            tangent = self.contact._tangent_basis(normal)
            if active:
                point = self.contact.tangent_value(configuration, name)
                velocity = (point - self.frame_previous[name]) / self.dt
                jacobian = self.contact.tangent_jacobian(configuration, name) / self.dt
                activation = float(item.get("activation", item.get("score", 0.0)))
            else:
                velocity = np.zeros(3)
                jacobian = np.zeros((3, self.model.nv))
                activation = 0.0
            errors.extend(tangent @ velocity)
            jacobians.extend((tangent[0] @ jacobian, tangent[1] @ jacobian))
            costs.extend((self.cost_value * activation,) * 2)
        return np.asarray(errors), np.asarray(jacobians), np.asarray(costs)

    def compute_error(self, configuration):
        error, _, costs = self._rows(configuration)
        self.cost = costs
        return error

    def compute_jacobian(self, configuration):
        _, jacobian, costs = self._rows(configuration)
        self.cost = costs
        return jacobian


class _SceneCollisionLimit(Limit):
    """Independent MuJoCo robot-scene active-set limit for V5."""
    def __init__(self, model, config):
        self.model = model
        self.config = dict(config)
        self.enabled = bool(config.get("enabled", True))
        self.activate = float(config.get("activate_distance", .06))
        self.margin = float(config.get("margin", .004))
        self.hard_band = max(0.0, float(config.get("hard_activation_band", 0.0)))
        self.pair_hysteresis = max(
            0.0, float(config.get("pair_hysteresis", 0.015))
        )
        prefix = str(config.get("scene_body_prefix", "scene_"))
        self.scene_geoms = [g for g in range(model.ngeom) if model.geom_contype[g] and (mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or "").startswith(prefix)]
        self.robot_geoms = [g for g in range(model.ngeom) if model.geom_contype[g] and g not in self.scene_geoms and int(model.geom_bodyid[g]) != 0]
        self.active_pairs = []
        self.previous_selection = {}
        self.minimum_distance = np.inf
        self.maximum_penetration = 0.
        self.exact_query_pairs = 0
        self.broadphase_culled_pairs = 0

    def prepare_active_set(self, configuration, dt=0.):
        del dt; configuration.update(); pairs=[]; all_distances=[]
        if not self.enabled:
            self.active_pairs = []; self.minimum_distance = np.inf; self.maximum_penetration = 0.0
            self.exact_query_pairs = 0; self.broadphase_culled_pairs = 0
            return
        self.exact_query_pairs = 0
        self.broadphase_culled_pairs = 0
        continuity_candidates = []
        for robot_geom in self.robot_geoms:
            for scene_geom in self.scene_geoms:
                # A geom's bounding sphere gives a conservative lower bound
                # on the pair distance.  If that lower bound is already past
                # the activation band, the exact MuJoCo query cannot produce
                # an active constraint and is safely skipped.  This is only a
                # broad-phase optimization; every surviving pair still uses
                # mj_geomDistance and the final validator replays all pairs.
                center_distance = float(np.linalg.norm(
                    configuration.data.geom_xpos[robot_geom]
                    - configuration.data.geom_xpos[scene_geom]
                ))
                broadphase_distance = center_distance - float(
                    self.model.geom_rbound[robot_geom] + self.model.geom_rbound[scene_geom]
                )
                if broadphase_distance > self.activate:
                    all_distances.append(broadphase_distance)
                    self.broadphase_culled_pairs += 1
                    continue
                fromto=np.zeros(6)
                self.exact_query_pairs += 1
                distance=float(mj.mj_geomDistance(self.model,configuration.data,robot_geom,scene_geom,self.activate,fromto))
                all_distances.append(distance)
                if distance > self.activate: continue
                vector=fromto[:3]-fromto[3:]; norm=float(np.linalg.norm(vector))
                if norm < 1e-10: continue
                # MuJoCo returns the closest points in geom1/geom2 order.
                # For a separated pair p1-p2 points toward free space; for a
                # penetrating pair the signed distance reverses that local
                # direction.  Apply the sign so n always points from the
                # scene into robot free space, which is the direction required
                # by -n^T J dq <= d-margin.
                sign = np.sign(distance) if abs(distance) > 1e-12 else 1.0
                item = {"robot_geom":robot_geom,"scene_geom":scene_geom,"distance":distance,"normal":None,"robot_point":fromto[:3].copy()}
                # Positive-distance pairs farther than the hard safety band
                # are useful diagnostics but needlessly constrain the motion
                # objective.  Any penetration (distance < 0) and every pair
                # inside margin remain unconditionally active.
                if distance <= self.activate:
                    continuity_candidates.append(item)
                if distance > self.margin + self.hard_band:
                    continue
                item["normal"] = sign * vector/norm
                pairs.append(item)
        # A convex decomposition can return many overlapping/near-identical
        # pieces for one robot link.  Enforcing every pair at once creates
        # contradictory normals at a penetration seam (this is especially
        # common when an ankle mesh has six or more CoACD pieces touching the
        # same stair edge).  Keep the closest deterministic pair per
        # (robot-body, scene-piece) by default.  The next SQP substep re-
        # queries after the link has moved away.  ``pair_grouping=geom`` is
        # retained as an explicit diagnostic option for callers that need the
        # old per-geom active set.
        selected = {}
        grouping = str(self.config.get("pair_grouping", "body_scene"))
        def group_key(item):
            if grouping == "geom":
                return int(item["robot_geom"])
            body_id = int(self.model.geom_bodyid[item["robot_geom"]])
            return (body_id, int(item["scene_geom"]))

        active_by_group = {}
        for item in pairs:
            active_by_group.setdefault(group_key(item), []).append(item)
        all_by_pair = {
            (int(item["robot_geom"]), int(item["scene_geom"])): item
            for item in continuity_candidates
        }
        # A previous pair can remain in the active set slightly beyond the
        # hard activation band.  This keeps the collision normal continuous
        # while a link crosses a CoACD seam; it is still bounded by the
        # ordinary ``activate`` distance and is re-queried every substep.
        group_keys = set(active_by_group)
        for key, previous in self.previous_selection.items():
            current = all_by_pair.get(
                (int(previous["robot_geom"]), int(previous["scene_geom"]))
            )
            if current is not None and current["distance"] <= self.margin + self.hard_band + self.pair_hysteresis:
                group_keys.add(key)
        for key in sorted(group_keys, key=str):
            candidates = active_by_group.get(key, [])
            previous = self.previous_selection.get(key)
            # Hard safety always wins over continuity: if any currently
            # queried pair for this link is already penetrating, select the
            # deepest one immediately.  Hysteresis is only for the positive
            # distance band where keeping a stable normal is safe.
            penetrating = [
                item for item in candidates
                if float(item["distance"]) < self.margin
            ]
            if penetrating:
                selected[key] = min(
                    penetrating,
                    key=lambda item: (float(item["distance"]), int(item["scene_geom"]), int(item["robot_geom"])),
                )
                continue
            if previous is not None:
                retained = all_by_pair.get(
                    (int(previous["robot_geom"]), int(previous["scene_geom"]))
                )
                if retained is not None and retained["distance"] <= self.margin + self.hard_band + self.pair_hysteresis:
                    if retained["normal"] is None:
                        # The pair was outside the hard band, so its normal
                        # was not needed above.  Re-querying the exact pair
                        # here avoids reusing a stale normal across motion.
                        closest = np.zeros(6)
                        distance = float(mj.mj_geomDistance(
                            self.model, configuration.data,
                            int(retained["robot_geom"]), int(retained["scene_geom"]),
                            self.activate, closest,
                        ))
                        vector = closest[:3] - closest[3:]
                        norm = float(np.linalg.norm(vector))
                        if norm < 1e-10:
                            retained = None
                        else:
                            sign = np.sign(distance) if abs(distance) > 1e-12 else 1.0
                            retained["normal"] = sign * vector / norm
                            retained["robot_point"] = closest[:3].copy()
                            retained["distance"] = distance
                    if retained is not None:
                        selected[key] = retained
                        continue
            if candidates:
                selected[key] = min(
                    candidates,
                    key=lambda item: (float(item["distance"]), int(item["scene_geom"]), int(item["robot_geom"])),
                )
        self.active_pairs = list(selected.values())
        self.previous_selection = {
            key: dict(value) for key, value in selected.items()
        }
        self.minimum_distance=min(all_distances,default=np.inf); self.maximum_penetration=max(0.,-self.minimum_distance)

    def all_distances(self, configuration) -> np.ndarray:
        """Replay every robot/scene geom pair for final validation."""
        configuration.update()
        values = []
        for robot_geom in self.robot_geoms:
            for scene_geom in self.scene_geoms:
                values.append(float(mj.mj_geomDistance(
                    self.model, configuration.data, robot_geom, scene_geom,
                    1e6, np.zeros(6))))
        return np.asarray(values, dtype=float)

    def compute_qp_inequalities(self, configuration, dt):
        del dt; rows=[]; bounds=[]
        for item in self.active_pairs:
            body=int(self.model.geom_bodyid[item["robot_geom"]]); jp=np.zeros((3,self.model.nv)); jr=np.zeros_like(jp); mj.mj_jac(self.model,configuration.data,jp,jr,item["robot_point"],body)
            rows.append(-item["normal"]@jp); bounds.append(item["distance"]-self.margin)
        return Constraint(G=np.asarray(rows),h=np.asarray(bounds)) if rows else Constraint()


class _TrustLimit(Limit):
    def __init__(self, model, radius): self.model=model; self.radius=float(radius)
    def compute_qp_inequalities(self, configuration, dt):
        del configuration, dt; identity=np.eye(self.model.nv); return Constraint(np.vstack((identity,-identity)), np.full(2*self.model.nv,self.radius))


class _FrameDisplacementLimit(Limit):
    """Bound total motion from the previous exported 50 Hz configuration.

    Applying a velocity clamp after QP integration can invalidate collision
    inequalities that the QP just satisfied.  This limit expresses the same
    frame budget directly in Mink's ``delta-q`` variable.  An L1 ball is used
    for free-base translation/rotation because its eight linear half-spaces
    conservatively imply the configured Euclidean speed limit.
    """

    _SIGNS_3D = np.asarray([
        [sx, sy, sz]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ])

    def __init__(self, model: mj.MjModel, config: dict[str, Any]):
        self.model = model
        self.linear_limit = float(config.get("base_linear_velocity_limit", np.inf))
        self.angular_limit = float(config.get("base_angular_velocity_limit", np.inf))
        self.joint_limit = float(config.get("joint_velocity_limit", np.inf))
        self.anchor_qpos: np.ndarray | None = None
        self.frame_dt = 0.0
        self.enabled = False

    def set_frame(self, anchor_qpos: np.ndarray, frame_dt: float, *, enabled: bool) -> None:
        self.anchor_qpos = np.asarray(anchor_qpos, dtype=float).reshape(self.model.nq).copy()
        self.frame_dt = float(frame_dt)
        self.enabled = bool(enabled)

    def compute_qp_inequalities(self, configuration, dt):
        del dt
        if not self.enabled or self.anchor_qpos is None or self.frame_dt <= 0.0:
            return Constraint()
        current_delta = np.zeros(self.model.nv, dtype=float)
        mj.mj_differentiatePos(
            self.model,
            current_delta,
            1.0,
            self.anchor_qpos,
            configuration.data.qpos,
        )
        rows: list[np.ndarray] = []
        bounds: list[float] = []
        for start, speed in ((0, self.linear_limit), (3, self.angular_limit)):
            if not np.isfinite(speed) or speed <= 0.0:
                continue
            for signs in self._SIGNS_3D:
                row = np.zeros(self.model.nv, dtype=float)
                row[start:start + 3] = signs
                rows.append(row)
                bounds.append(
                    speed * self.frame_dt
                    - float(signs @ current_delta[start:start + 3])
                )
        if np.isfinite(self.joint_limit) and self.joint_limit > 0.0:
            maximum = self.joint_limit * self.frame_dt
            for joint_id in range(self.model.njnt):
                if not is_dof_joint_type(self.model.jnt_type[joint_id]):
                    continue
                dof = int(self.model.jnt_dofadr[joint_id])
                row = np.zeros(self.model.nv, dtype=float)
                row[dof] = 1.0
                rows.extend((row, -row))
                bounds.extend((
                    maximum - float(current_delta[dof]),
                    maximum + float(current_delta[dof]),
                ))
        return (
            Constraint(G=np.asarray(rows), h=np.asarray(bounds))
            if rows else Constraint()
        )


class WholeBodyRetargetSolver:
    """Independent task/QP pipeline operating on one resolved V5 task."""
    def __init__(self, task: RetargetTask, terrain: TerrainField, environment_pool: np.ndarray, fps: float = 50.0, solver: str = "daqp", scene_model=None):
        self.task=task; self.config=_load_config(task.config_path); self.terrain=terrain; self.scene_model=scene_model; self.fps=float(fps); self.dt=1./self.fps; self.solver=solver
        # The resolved RobotModelSpec owns the actual MJCF path.  Do not infer
        # it from the temporary config location (which may live outside the
        # repository when a user chooses an arbitrary output directory).
        self.robot_xml = task.robot.mjcf_path.resolve()
        self.model=mj.MjModel.from_xml_path(str(self.robot_xml)); self.configuration=mink.Configuration(self.model)
        self.robot_profile = RobotProfile.from_model(
            task.robot.name, self.model, task.robot.semantic_points
        )
        self.robot_profile.validate(self.model)
        morphology_cfg = self.config.get("morphology", {})
        scene_cfg = self.config.get("scene", {})
        self.root_policy = str(morphology_cfg.get("root_policy", "support_aware"))
        self.robot_height = float(scene_cfg.get("robot_height", 1.316))
        self.human_height = float(
            morphology_cfg.get("human_height", scene_cfg.get("default_human_height", 1.78))
        )
        # Filled from the complete source timeline before solving.  Root Z is
        # anchored to the robot's measured support height and only follows
        # source vertical *changes* thereafter; an adult pelvis world height
        # must never be copied directly into the shorter robot.
        self._root_reference_pelvis_z: float | None = None
        self._root_reference_surface_z: float | None = None
        self._robot_root_support_height: float | None = None
        self._root_vertical_scale = self.robot_height / max(self.human_height, 1e-6)
        self.reference_metadata: dict[str, Any] = {}
        for name,bounds in self.config.get("joint_position_limits",{}).items():
            if name in [mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_JOINT,i) for i in range(self.model.njnt)]:
                jid=_joint_id(self.model, name)
                xml_lower, xml_upper = self.model.jnt_range[jid]
                lower, upper = max(float(xml_lower), float(bounds[0])), min(float(xml_upper), float(bounds[1]))
                if not lower < upper:
                    raise ValueError(f"Invalid V5 joint range for {name}: {bounds}")
                self.model.jnt_range[jid] = (lower, upper); self.model.jnt_limited[jid]=1
        self.interaction=_InteractionTask(self.model,self.config["semantic_points"],environment_pool,self.config.get("interaction_graph",{}))
        direction_cfg = self.config.get("bone_direction", {})
        direction_edges = direction_cfg.get("edges", [
            ["pelvis", "left_hip"], ["left_hip", "left_knee"],
            ["left_knee", "left_foot"], ["left_foot", "left_toe"],
            ["pelvis", "right_hip"], ["right_hip", "right_knee"],
            ["right_knee", "right_foot"], ["right_foot", "right_toe"],
            ["left_shoulder", "left_elbow"], ["left_elbow", "left_hand"],
            ["right_shoulder", "right_elbow"], ["right_elbow", "right_hand"],
        ])
        self.bone_direction = _BoneDirectionTask(
            self.model, self.interaction.points, direction_edges,
            cost=float(direction_cfg.get("cost", 1.5)),
            gain=float(direction_cfg.get("gain", .3)),
        )
        plane_cfg = self.config.get("limb_plane", {})
        plane_triples = plane_cfg.get("triples", [
            ["left_hip", "left_knee", "left_foot"],
            ["right_hip", "right_knee", "right_foot"],
            ["left_shoulder", "left_elbow", "left_hand"],
            ["right_shoulder", "right_elbow", "right_hand"],
        ])
        self.limb_plane = _LimbPlaneTask(
            self.model, self.interaction.points, plane_triples,
            cost=float(plane_cfg.get("cost", .6)),
            gain=float(plane_cfg.get("gain", .2)),
        )
        root=self.config["global_anchor"]; self.root=_RootTask(self.model,root["robot_body"],root["cost"])
        self.torso = _TorsoCoherenceTask(self.model, self.config.get("torso_pelvis_coherence", {}))
        self.contact=_ContactTask(self.model,self.config.get("contact_tasks",{}).get("robot_points",{}),self.config.get("contact_tasks",{}))
        self.foot_normal = _FootNormalTask(
            self.model, self.config.get("foot_orientation", {})
        )
        self.foot_motion = _FootMotionTask(
            self.model, self.config.get("foot_motion", {})
        )
        self.foot_temporal = _FootTemporalTask(
            self.model, self.contact, self.dt,
            self.config.get("foot_temporal", {}),
        )
        self.terrain_limit=TerrainNonPenetrationLimit(self.model,terrain,self.config.get("terrain_nonpenetration",{})); self.trust=_TrustLimit(self.model,self.config.get("solver",{}).get("trust_region",.12)); self.scene_collision=_SceneCollisionLimit(self.model,self.config.get("scene_collision",{}))
        self.frame_displacement = _FrameDisplacementLimit(
            self.model, self.config.get("solver", {})
        )
        self.config_limit=mink.ConfigurationLimit(self.model); self.velocity=mink.VelocityLimit(self.model,{mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_JOINT,i):float(self.config.get("solver",{}).get("joint_velocity_limit",10.)) for i in range(self.model.njnt) if is_dof_joint_type(self.model.jnt_type[i])})
        posture=self.config.get("posture",{})
        costs=np.full(self.model.nv,float(posture.get("nominal_cost",.01)))
        # A free-base's xyz/roll/pitch/yaw are not ordinary joint posture.
        # Root and anatomical-frame tasks own those six DoFs; anchoring them
        # to qpos0 here creates a competing objective and can twist the torso
        # when a foot or butt contact is active.
        if self.model.nv >= 6:
            costs[:6] = float(posture.get("floating_base_cost", 0.0))
        self.nominal=mink.PostureTask(self.model,costs,gain=.3,lm_damping=1.); self.nominal.set_target(self.model.qpos0)
        self._temporal_costs = np.full(self.model.nv, float(posture.get("temporal_cost", .05)))
        if self.model.nv >= 6:
            self._temporal_costs[:6] = float(posture.get("floating_base_temporal_cost", 0.0))
        self.previous=None; self.frame_index=0; self.diagnostics=[]
        self._robot_static_anchors: dict[str, dict[str, Any]] = {}
        self.task_builder = TaskBuilder(self)
        self.retry_damping_factors = tuple(
            float(value) for value in self.config.get("solver", {}).get(
                "retry_damping_factors", [1.0, 2.5, 6.0]
            )
        )
        self.retry_trust_factors = tuple(
            float(value) for value in self.config.get("solver", {}).get(
                "retry_trust_factors", [1.0, 1.5, 2.5]
            )
        )
        if not self.retry_damping_factors or len(self.retry_damping_factors) != len(self.retry_trust_factors):
            raise ValueError(
                "V5 solver retry_damping_factors and retry_trust_factors "
                "must be non-empty arrays of equal length"
            )

    def _hard_constraint_violations(self, solve_dt: float) -> tuple[bool, bool]:
        """Re-query nonlinear terrain and scene constraints at current qpos."""
        self.terrain_limit.prepare_active_set(self.configuration, solve_dt)
        self.scene_collision.prepare_active_set(self.configuration, solve_dt)
        tolerance = float(
            self.config.get("solver", {}).get(
                "nonlinear_backtracking_tolerance", 1e-6
            )
        )
        terrain_violation = any(
            float(item["slack"]) < -tolerance
            for item in self.terrain_limit.all_measurements
        )
        collision_violation = any(
            float(item["distance"]) < self.scene_collision.margin - tolerance
            for item in self.scene_collision.active_pairs
        )
        return terrain_violation, collision_violation

    def _backtrack_to_safe_configuration(
        self,
        start_qpos: np.ndarray,
        proposed_qpos: np.ndarray,
        solve_dt: float,
    ) -> tuple[float, bool]:
        """Keep the largest collision-free prefix of a nonlinear IK step.

        A local QP inequality can cross a curved mesh between linearization
        points.  Restoring the complete previous frame is safe but can freeze
        a trajectory indefinitely at a contact boundary.  This bounded line
        search keeps the hard margin unchanged and retains forward progress
        on the MuJoCo configuration manifold.
        """
        start = np.asarray(start_qpos, dtype=float).reshape(self.model.nq)
        proposed = np.asarray(proposed_qpos, dtype=float).reshape(self.model.nq)
        self.configuration.update(start)
        if any(self._hard_constraint_violations(solve_dt)):
            self.configuration.update(proposed)
            self._hard_constraint_violations(solve_dt)
            return 0.0, False

        low, high = 0.0, 1.0
        iterations = max(
            1,
            int(
                self.config.get("solver", {}).get(
                    "nonlinear_backtracking_iterations", 10
                )
            ),
        )
        for _ in range(iterations):
            middle = 0.5 * (low + high)
            self.configuration.update(
                _interpolate_qpos(self.model, start, proposed, middle)
            )
            if any(self._hard_constraint_violations(solve_dt)):
                high = middle
            else:
                low = middle
        self.configuration.update(
            _interpolate_qpos(self.model, start, proposed, low)
        )
        safe = not any(self._hard_constraint_violations(solve_dt))
        return float(low), bool(safe)

    def root_target(self, source, frame, contact_frame):
        """Return the pelvis target from the unified robot reference motion."""
        del source, contact_frame
        pelvis_target, pelvis_quaternion = frame.get("pelvis", frame.get("root"))
        return (
            np.asarray(pelvis_target, dtype=float).copy(),
            np.asarray(pelvis_quaternion, dtype=float).copy(),
        )

    def _configure_root_reference(self, source_frames, contact_frames) -> np.ndarray:
        """Measure one source/robot support baseline before the IK loop.

        Contact detection remains source-only.  This method only converts the
        source support-phase pelvis height into a robot root baseline; it does
        not inspect robot qpos or feed robot state back into contact labels.
        """
        self.configuration.update(self.model.qpos0)
        root_z = float(self.configuration.data.xpos[self.root.body_id, 2])
        foot_heights = []
        for name in ("left_heel", "left_toe", "right_heel", "right_toe"):
            if name in self.contact.support_geom_ids:
                foot_heights.append(float(self.contact.support_value(
                    self.configuration, name, np.array([0.0, 0.0, 1.0])
                )[2]))
            elif name in self.contact.points:
                foot_heights.append(float(
                    self.contact.points[name].value(self.configuration)[2]
                ))
        if not foot_heights:
            return np.zeros(len(source_frames), dtype=float)
        self._robot_root_support_height = root_z - float(np.median(foot_heights))

        pelvis_values = np.asarray([
            float(np.asarray(source["pelvis"], dtype=float)[2])
            for source in source_frames
        ])
        observed_surfaces = np.full(len(source_frames), np.nan, dtype=float)
        clearances = []
        for index, (source, frame) in enumerate(zip(source_frames, contact_frames)):
            pelvis = source.get("pelvis")
            if pelvis is None:
                continue
            contacts = frame.get("contacts", {}) if isinstance(frame, dict) else {}
            support_state = str(frame.get("support_state", "")) if isinstance(frame, dict) else ""
            candidates = []
            for channel in ("left_heel", "left_toe", "right_heel", "right_toe"):
                item = contacts.get(channel, {})
                if str(item.get("state", "NONE")) == "NONE":
                    continue
                if float(item.get("activation", item.get("score", 0.0))) <= 1e-3:
                    continue
                surface = item.get("surface_point_solver")
                normal = np.asarray(item.get("surface_normal_solver", [0., 0., 1.]), dtype=float)
                if surface is not None and np.isfinite(np.asarray(surface, dtype=float)).all() and normal[2] > 0.6:
                    candidates.append(float(np.asarray(surface, dtype=float).reshape(3)[2]))
            if candidates and (support_state in {"SUPPORTED", ""} or contacts):
                observed_surfaces[index] = float(np.median(candidates))
                clearances.append(pelvis_values[index] - observed_surfaces[index])

        known = np.flatnonzero(np.isfinite(observed_surfaces))
        floor = getattr(self.terrain, "floor_z", None)
        fallback_surface = float(floor) if floor is not None else 0.0
        if len(known):
            surface_track = np.interp(
                np.arange(len(source_frames), dtype=float),
                known.astype(float),
                observed_surfaces[known],
            )
            baseline_clearance = float(np.median(clearances))
        else:
            surface_track = np.full(len(source_frames), fallback_surface)
            foot_clearances = []
            for source in source_frames:
                feet = [
                    float(np.asarray(source[name], dtype=float)[2])
                    for name in ("left_heel", "left_toe", "right_heel", "right_toe",
                                 "left_foot", "right_foot")
                    if name in source
                ]
                if feet:
                    foot_clearances.append(
                        float(np.asarray(source["pelvis"], dtype=float)[2])
                        - min(feet)
                    )
            baseline_clearance = float(np.median(foot_clearances)) if foot_clearances else float(
                np.median(pelvis_values - fallback_surface)
            )
        self._root_reference_pelvis_z = baseline_clearance
        self._root_reference_surface_z = float(np.median(surface_track))
        target_pelvis_z = (
            surface_track
            + float(self._robot_root_support_height)
            + (pelvis_values - surface_track - baseline_clearance)
            * self._root_vertical_scale
        )
        offsets = target_pelvis_z - pelvis_values
        self.reference_metadata = {
            "policy": "terrain_relative_robot_reference",
            "robot_root_support_height": float(self._robot_root_support_height),
            "source_support_clearance": baseline_clearance,
            "vertical_scale": float(self._root_vertical_scale),
            "observed_support_frames": int(len(known)),
            "minimum_vertical_offset": float(np.min(offsets, initial=0.0)),
            "maximum_vertical_offset": float(np.max(offsets, initial=0.0)),
        }
        return offsets

    def _prepare_robot_reference_motion(
        self, source_frames, solver_frames, contact_frames
    ):
        """Apply one shared height correction to every robot target point."""
        references = [
            {name: np.asarray(value, dtype=float).copy() for name, value in frame.items()}
            for frame in source_frames
        ]
        targets = [
            {
                name: (
                    np.asarray(value[0], dtype=float).copy(),
                    np.asarray(value[1], dtype=float).copy(),
                )
                for name, value in frame.items()
            }
            for frame in solver_frames
        ]
        if self.root_policy in {"source", "disabled"}:
            self.reference_metadata = {"policy": self.root_policy}
            return references, targets
        offsets = self._configure_root_reference(references, contact_frames)
        for index, offset in enumerate(offsets):
            translation = np.array([0.0, 0.0, float(offset)])
            for name in references[index]:
                if references[index][name].shape == (3,):
                    references[index][name] += translation
            for name, (position, quaternion) in targets[index].items():
                targets[index][name] = (position + translation, quaternion)
        return references, targets

    def _update_scene_time(self, timestamp: float) -> None:
        """Update dynamic query geometry and MuJoCo mocap bodies together."""
        if hasattr(self.terrain, "set_time"):
            self.terrain.set_time(timestamp)
        if self.scene_model is None:
            return
        for asset in self.scene_model.assets:
            if asset.pose_trajectory is None:
                continue
            body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, f"scene_{asset.asset_id}")
            if body_id < 0:
                body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, f"scene_{str(asset.asset_id).replace('-', '_')}")
            if body_id < 0:
                continue
            mocap_id = int(self.model.body_mocapid[body_id])
            if mocap_id < 0:
                continue
            pose = asset.pose_at(timestamp)
            self.configuration.data.mocap_pos[mocap_id] = pose[:3, 3]
            self.configuration.data.mocap_quat[mocap_id] = Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
        self.configuration.update()

    def _project_limited_qpos(self) -> None:
        """Project only tiny numerical limit drift after integration.

        Mink's velocity/configuration limits constrain the QP step, but a
        floating-point integration can leave a hinge a few ulps outside its
        XML range.  This is a numerical projection, not a pose correction:
        it never expands a limit or changes the free base.
        """
        qpos = self.configuration.data.qpos
        for joint_id in range(self.model.njnt):
            if not self.model.jnt_limited[joint_id]:
                continue
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type not in (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)):
                continue
            address = int(self.model.jnt_qposadr[joint_id])
            lower, upper = self.model.jnt_range[joint_id]
            qpos[address] = np.clip(qpos[address], float(lower), float(upper))
        self.configuration.update()

    def _project_frame_velocity(self, frame_anchor_qpos: np.ndarray, dt: float) -> None:
        """Bound cumulative frame motion after each QP integration.

        Mink's joint velocity limit intentionally covers articulated DoFs;
        MuJoCo's free joint is not represented by that limit.  During a stair
        contact transition an otherwise feasible QP could therefore move the
        root by several trust-region radii in one frame, producing the leg
        twist/jump seen in the V5 replay.  Contact/collision polish steps are
        also integrations, so the bound is measured from the frame anchor,
        not from the immediately preceding substep.  This prevents several
        individually-valid substeps from exceeding the exported 50 Hz
        velocity contract.
        """
        frame_anchor_qpos = np.asarray(frame_anchor_qpos, dtype=float).reshape(-1)
        if frame_anchor_qpos.size < 7 or not np.isfinite(dt) or dt <= 0.0:
            return
        qpos = self.configuration.data.qpos
        linear_limit = float(self.config.get("solver", {}).get("base_linear_velocity_limit", 2.5))
        angular_limit = float(self.config.get("solver", {}).get("base_angular_velocity_limit", 6.0))
        if np.isfinite(linear_limit) and linear_limit > 0.0:
            delta = np.asarray(qpos[:3] - frame_anchor_qpos[:3], dtype=float)
            max_step = linear_limit * float(dt)
            length = float(np.linalg.norm(delta))
            if np.isfinite(length) and length > max_step:
                qpos[:3] = frame_anchor_qpos[:3] + delta * (max_step / max(length, 1e-12))
        if np.isfinite(angular_limit) and angular_limit > 0.0:
            anchor = Rotation.from_quat(frame_anchor_qpos[3:7], scalar_first=True)
            current = Rotation.from_quat(qpos[3:7], scalar_first=True)
            delta_rotation = anchor.inv() * current
            angle = float(np.linalg.norm(delta_rotation.as_rotvec()))
            max_angle = angular_limit * float(dt)
            if np.isfinite(angle) and angle > max_angle:
                current = anchor * Rotation.from_rotvec(
                    delta_rotation.as_rotvec() * (max_angle / max(angle, 1e-12))
                )
                qpos[3:7] = current.as_quat(scalar_first=True)

        # The articulated velocity limit is expressed in scalar MuJoCo DOFs.
        # Apply it to cumulative frame displacement as well, including the
        # extra contact/collision polish integrations.
        joint_limit = float(self.config.get("solver", {}).get("joint_velocity_limit", np.inf))
        if np.isfinite(joint_limit) and joint_limit > 0.0:
            max_joint_step = joint_limit * float(dt)
            for joint_id in range(self.model.njnt):
                if not is_dof_joint_type(self.model.jnt_type[joint_id]):
                    continue
                address = int(self.model.jnt_qposadr[joint_id])
                delta = float(qpos[address] - frame_anchor_qpos[address])
                qpos[address] = frame_anchor_qpos[address] + float(
                    np.clip(delta, -max_joint_step, max_joint_step)
                )
        self.configuration.update()

    def _contact_normal_residual(self, contact_frame: dict[str, Any]) -> float:
        """Return the largest reliable source-contact normal residual.

        This is deliberately evaluated on the post-integration configuration.
        The ordinary IK passes can trade a few millimetres of desired contact
        for the primary interaction objective; a bounded contact polish below
        recovers that residual without changing the task topology or touching
        unconstrained/low-confidence channels.
        """
        return self._contact_residuals(contact_frame)[0]

    def _contact_residuals(
        self,
        contact_frame: dict[str, Any],
        channels: set[str] | None = None,
        activation_threshold: float | None = None,
    ) -> tuple[float, float]:
        """Return normal and STATIC tangential residuals for contact polish."""
        threshold = float(
            self.config.get("contact_tasks", {}).get(
                "polish_activation", 0.25
            )
            if activation_threshold is None else activation_threshold
        )
        maximum = 0.0
        tangent_maximum = 0.0
        for channel, item in contact_frame.get("contacts", {}).items():
            if channels is not None and channel not in channels:
                continue
            if channel not in self.contact.points:
                continue
            if str(item.get("state", "NONE")) == "NONE":
                continue
            if float(item.get("activation", item.get("score", 0.0))) < threshold:
                continue
            state = str(item.get("state", "NONE"))
            normal_key = (
                "anchor_normal_solver"
                if state == "STATIC" and "anchor_normal_solver" in item
                else "surface_normal_solver"
            )
            normal = np.asarray(item.get(normal_key, [0., 0., 1.]), dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            if channel in self.contact.support_geom_ids:
                point = self.contact.support_value(self.configuration, channel, normal)
                tangent_point = self.contact.tangent_value(
                    self.configuration, channel
                )
            else:
                point = self.contact.points[channel].value(self.configuration)
                tangent_point = point
            surface = np.asarray(item.get("surface_point_solver", point), dtype=float)
            residual = abs(float(normal @ (point - surface) - self.contact.clearance))
            maximum = max(maximum, residual)
            if state == "STATIC":
                tangent = self.contact._tangent_basis(normal)
                anchor = np.asarray(item.get("tangent_anchor_solver", surface), dtype=float)
                tangent_maximum = max(
                    tangent_maximum,
                    float(np.linalg.norm(tangent @ (tangent_point - anchor))),
                )
        return maximum, tangent_maximum

    def _contact_channel_residuals(
        self,
        contact_frame: dict[str, Any],
        channels: set[str] | None = None,
        activation_threshold: float = 0.0,
    ) -> dict[str, float]:
        """Return normal residuals per contact channel at the current qpos.

        Support polish used to solve all heel/toe channels in one QP.  That
        makes a single unreachable channel (typically a foot crossing a stair
        edge) reject corrections that are feasible for the other foot.  The
        per-channel map keeps the correction policy local and gives the
        diagnostics enough information to explain which contact was actually
        unreachable.
        """
        residuals: dict[str, float] = {}
        for channel, item in contact_frame.get("contacts", {}).items():
            if channels is not None and channel not in channels:
                continue
            if channel not in self.contact.points:
                continue
            if str(item.get("state", "NONE")) == "NONE":
                continue
            if float(item.get("activation", item.get("score", 0.0))) < activation_threshold:
                continue
            state = str(item.get("state", "NONE"))
            normal_key = (
                "anchor_normal_solver"
                if state == "STATIC" and "anchor_normal_solver" in item
                else "surface_normal_solver"
            )
            normal = np.asarray(item.get(normal_key, [0., 0., 1.]), dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            if channel in self.contact.support_geom_ids:
                point = self.contact.support_value(self.configuration, channel, normal)
            else:
                point = self.contact.points[channel].value(self.configuration)
            surface = np.asarray(item.get("surface_point_solver", point), dtype=float)
            residuals[channel] = abs(
                float(normal @ (point - surface) - self.contact.clearance)
            )
        return residuals

    def _constraint_snapshot(self, solve_dt: float) -> dict[str, Any]:
        """Return concrete nonlinear terrain/scene violations for diagnostics."""
        self.terrain_limit.prepare_active_set(self.configuration, solve_dt)
        self.scene_collision.prepare_active_set(self.configuration, solve_dt)
        tolerance = float(
            self.config.get("solver", {}).get(
                "nonlinear_backtracking_tolerance", 1e-6
            )
        )
        terrain_rows = []
        for item in self.terrain_limit.violating_measurements(
            self.configuration, tolerance=tolerance
        ):
            hit = item["hit"]
            terrain_rows.append({
                "body_id": int(item["body_id"]),
                "index": int(item["index"]),
                "signed_distance": float(item["signed_distance"]),
                "margin": float(item["margin"]),
                "slack": float(item["slack"]),
                "surface_id": str(hit.surface_id),
                "normal": np.asarray(hit.normal, dtype=float).copy(),
                "point": np.asarray(item["point"], dtype=float).copy(),
            })
        scene_rows = []
        for pair in self.scene_collision.active_pairs:
            distance = float(pair["distance"])
            if distance < self.scene_collision.margin - tolerance:
                scene_rows.append({
                    "robot_geom": int(pair["robot_geom"]),
                    "scene_geom": int(pair["scene_geom"]),
                    "distance": distance,
                    "margin": float(self.scene_collision.margin),
                    "slack": distance - float(self.scene_collision.margin),
                })
        return {
            "terrain_violation_count": len(terrain_rows),
            "terrain_violations": terrain_rows,
            "scene_violation_count": len(scene_rows),
            "scene_violations": scene_rows,
            "minimum_terrain_signed_distance": float(
                min((float(item["signed_distance"])
                     for item in self.terrain_limit.all_measurements), default=np.inf)
            ),
            "minimum_scene_distance": float(self.scene_collision.minimum_distance),
        }

    def _terrain_violation_state(self, tolerance: float = 1e-6) -> tuple[bool, float]:
        """Inspect complete terrain proxy measurements, not only active rows."""
        measurements = self.terrain_limit.all_measurements
        if not measurements:
            return False, float("inf")
        minimum = min(float(item["signed_distance"]) for item in measurements)
        violated = any(float(item["slack"]) < -float(tolerance) for item in measurements)
        return violated, minimum

    def _try_contact_channel_correction(
        self,
        channel: str,
        item: dict[str, Any],
        contact_frame: dict[str, Any],
        solve_dt: float,
        contact_cfg: dict[str, Any],
        activation_floor: float,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Try one bounded support correction and keep it only if safe.

        This is deliberately a feasibility probe around the existing Mink QP;
        it does not alter task priorities or disable terrain/scene limits.
        """
        before = self.configuration.data.qpos.copy()
        before_residual = self._contact_channel_residuals(
            contact_frame, {channel}, activation_threshold=0.0
        ).get(channel, np.inf)
        self.contact.set_contacts({channel: item})
        self.contact.activation_floor = float(activation_floor)
        self.terrain_limit.prepare_active_set(self.configuration, solve_dt)
        self.scene_collision.prepare_active_set(self.configuration, solve_dt)
        # A support-only correction should use the articulated leg/ankle
        # chain first.  Letting the free base consume the whole correction
        # budget lowers the pelvis into a chair/riser and is the common cause
        # of a combined support polish rollback.  The limits are restored
        # immediately after this one bounded probe.
        old_linear_limit = self.frame_displacement.linear_limit
        old_angular_limit = self.frame_displacement.angular_limit
        configured_linear = contact_cfg.get(
            "support_polish_base_linear_velocity_limit", None
        )
        configured_angular = contact_cfg.get(
            "support_polish_base_angular_velocity_limit", None
        )
        if configured_linear is not None:
            self.frame_displacement.linear_limit = min(
                old_linear_limit, float(configured_linear)
            )
        if configured_angular is not None:
            self.frame_displacement.angular_limit = min(
                old_angular_limit, float(configured_angular)
            )
        limits = self.task_builder.build_limits(include_velocity=True)
        polish_tasks = [self.contact]
        if bool(contact_cfg.get("support_polish_leg_preference", True)):
            side_token = "_L_" if str(channel).startswith("left_") else "_R_"
            costs = np.full(
                self.model.nv,
                float(contact_cfg.get("support_polish_other_cost", 15.0)),
            )
            if self.model.nv >= 6:
                costs[:6] = float(
                    contact_cfg.get("support_polish_root_cost", 30.0)
                )
            for joint_id in range(self.model.njnt):
                if not is_dof_joint_type(self.model.jnt_type[joint_id]):
                    continue
                name = mj.mj_id2name(
                    self.model, mj.mjtObj.mjOBJ_JOINT, joint_id
                ) or ""
                dof = int(self.model.jnt_dofadr[joint_id])
                if side_token not in name:
                    continue
                if "ANKLE_" in name:
                    costs[dof] = float(
                        contact_cfg.get("support_polish_ankle_cost", 0.02)
                    )
                elif "KNEE_" in name:
                    costs[dof] = float(
                        contact_cfg.get("support_polish_knee_cost", 0.08)
                    )
                elif "HIP_" in name:
                    costs[dof] = float(
                        contact_cfg.get("support_polish_hip_cost", 0.20)
                    )
            support_posture = mink.PostureTask(
                self.model,
                costs,
                gain=float(contact_cfg.get("support_polish_posture_gain", 0.5)),
                lm_damping=1.0,
            )
            support_posture.set_target(before)
            polish_tasks.append(support_posture)
        original_radius = self.trust.radius
        original_normal_only = bool(self.contact.normal_only)
        # A newly bound STATIC support may still need a small normal polish,
        # but releasing its tangential rows during that probe lets the
        # articulated correction move the freshly locked feature by an entire
        # frame.  Preserve the anchor for STATIC channels while leaving
        # SLIDING/swing behavior unchanged.
        if str(item.get("state", "NONE")) == "STATIC" and bool(
            item.get("robot_static_anchor_bound", False)
        ):
            self.contact.normal_only = False
        self.trust.radius = min(
            original_radius,
            max(float(contact_cfg.get("polish_trust_region", 0.02)), 1e-4),
        )
        correction = None
        reason = ""
        try:
            correction = mink.solve_ik(
                self.configuration,
                polish_tasks,
                solve_dt,
                self.solver,
                float(self.config.get("solver", {}).get("damping", .1)),
                limits=limits,
            )
        except Exception as error:
            reason = f"qp {type(error).__name__}: {error}"
        finally:
            self.trust.radius = original_radius
            self.contact.normal_only = original_normal_only
            self.frame_displacement.linear_limit = old_linear_limit
            self.frame_displacement.angular_limit = old_angular_limit
        if correction is None:
            return False, reason or "qp returned no correction", self._constraint_snapshot(solve_dt)
        if not np.isfinite(correction).all():
            return False, "qp produced non-finite correction", self._constraint_snapshot(solve_dt)
        self.configuration.integrate_inplace(correction, solve_dt)
        self._project_limited_qpos()
        snapshot = self._constraint_snapshot(solve_dt)
        if snapshot["terrain_violation_count"] or snapshot["scene_violation_count"]:
            self.configuration.update(before)
            safe_snapshot = self._constraint_snapshot(solve_dt)
            return (
                False,
                "nonlinear terrain/scene violation after correction",
                {"candidate": snapshot, "restored": safe_snapshot},
            )
        after_residual = self._contact_channel_residuals(
            contact_frame, {channel}, activation_threshold=0.0
        ).get(channel, np.inf)
        minimum_improvement = float(
            contact_cfg.get("per_channel_min_improvement", 1e-5)
        )
        if not np.isfinite(after_residual) or after_residual > before_residual - minimum_improvement:
            self.configuration.update(before)
            safe_snapshot = self._constraint_snapshot(solve_dt)
            return (
                False,
                "correction did not reduce channel residual",
                {
                    "candidate": snapshot,
                    "restored": safe_snapshot,
                    "residual_before": float(before_residual),
                    "residual_after": float(after_residual),
                },
            )
        snapshot["residual_before"] = float(before_residual)
        snapshot["residual_after"] = float(after_residual)
        return True, "accepted", snapshot

    def _seed_initial_pose(self, source: dict[str, np.ndarray], root_quaternion: np.ndarray) -> None:
        """Choose a non-mirrored, bent-knee warm start without random IK seeds."""
        base = self.model.qpos0.copy()
        base[:3] = np.asarray(source["pelvis"], dtype=float)
        base[3:7] = np.asarray(root_quaternion, dtype=float)
        base[3:7] /= max(float(np.linalg.norm(base[3:7])), 1e-12)

        candidates = [base.copy()]
        for bend in (0.35, 0.60, 0.90):
            candidate = base.copy()
            for joint_name in ("KNEE_PITCH_L_JOINT", "KNEE_PITCH_R_JOINT"):
                if joint_name in [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]:
                    candidate[_joint_qposadr(self.model, joint_name)] = bend
            for joint_name, value in (("ANKLE_PITCH_L_JOINT", -0.12), ("ANKLE_PITCH_R_JOINT", -0.12)):
                if joint_name in [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]:
                    candidate[_joint_qposadr(self.model, joint_name)] = value
            candidates.append(candidate)

        best = candidates[0]
        best_score = np.inf
        target_names = ("left_hip", "right_hip", "left_knee", "right_knee", "left_foot", "right_foot")
        for candidate in candidates:
            self.configuration.update(candidate)
            score = 0.0
            for name in target_names:
                if name not in source or name not in self.interaction.points:
                    continue
                score += float(np.linalg.norm(self.interaction.points[name].value(self.configuration) - source[name]))
            # A seed with the knees behind their anatomical limits is never
            # preferred, even if its end-effector score is marginally lower.
            if any(candidate[_joint_qposadr(self.model, j)] < -1e-6 for j in ("KNEE_PITCH_L_JOINT", "KNEE_PITCH_R_JOINT") if mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, j) >= 0):
                score += 1e3
            if score < best_score:
                best_score, best = score, candidate.copy()
        self.configuration.update(best)
        self.nominal.set_target(best)

    def _contact_frame_at_time(self, contact_frame: dict[str, Any], timestamp: float) -> dict[str, Any]:
        """Materialize asset-local anchors against the current scene pose.

        Contact episodes store anchors in asset coordinates so a moving object
        cannot leave the robot tracking a stale world point.  Static scenes
        simply return an equivalent shallow copy.
        """
        if self.scene_model is None or not contact_frame.get("contacts"):
            return contact_frame
        assets = {str(asset.asset_id): asset for asset in self.scene_model.assets}
        contacts = {}
        changed = False
        for channel, original in contact_frame.get("contacts", {}).items():
            item = dict(original)
            object_id = str(item.get("object_id", ""))
            local = item.get("asset_local_anchor")
            asset = assets.get(object_id)
            if asset is not None and local is not None:
                pose = asset.pose_at(timestamp)
                linear = np.asarray(pose[:3, :3], dtype=float)
                world_point = linear @ np.asarray(local, dtype=float).reshape(3) + pose[:3, 3]
                item["surface_point_solver"] = world_point
                item["tangent_anchor_solver"] = world_point
                local_normal = item.get("asset_local_normal")
                if local_normal is not None:
                    normal = np.linalg.inv(linear).T @ np.asarray(local_normal, dtype=float).reshape(3)
                    normal /= max(float(np.linalg.norm(normal)), 1e-12)
                    item["surface_normal_solver"] = normal
                    item["anchor_normal_solver"] = normal
                changed = True
            contacts[channel] = item
        return {**contact_frame, "contacts": contacts} if changed else contact_frame

    def _bind_robot_static_anchors(
        self, contact_frame: dict[str, Any], timestamp: float
    ) -> dict[str, Any]:
        """Lock STATIC tangential motion at the robot's realized contact.

        Source anchors identify the desired scene surface and continue to own
        the normal-position target.  Sticking, however, is a temporal robot
        condition: once a configured robot proxy establishes STATIC contact,
        its tangential point must stop moving.  Locking the source-human point
        directly over-constrains robots with different pelvis/sole width and
        creates large lateral leg twists.
        """
        contacts = {}
        solver_config = getattr(self, "config", {})
        contact_config = solver_config.get("contact_tasks", {})
        bind_activation = float(
            contact_config.get("static_anchor_activation", 0.25)
        )
        bind_normal_error = float(
            contact_config.get("static_anchor_max_normal_error", 0.012)
        )
        default_release_steps = 3 if hasattr(self, "config") else 1
        release_steps = max(
            1, int(contact_config.get("static_anchor_release_steps", default_release_steps))
        )
        assets = {
            str(asset.asset_id): asset for asset in getattr(self.scene_model, "assets", [])
        }
        active_channels = set()
        seen_channels = set()
        for channel, original in contact_frame.get("contacts", {}).items():
            seen_channels.add(channel)
            item = dict(original)
            state = str(item.get("state", "NONE"))
            if state != "STATIC" or channel not in self.contact.points:
                anchor = self._robot_static_anchors.get(channel)
                if anchor is not None:
                    anchor["inactive_steps"] = int(
                        anchor.get("inactive_steps", 0)
                    ) + 1
                    if anchor["inactive_steps"] >= release_steps:
                        self._robot_static_anchors.pop(channel, None)
                contacts[channel] = item
                continue
            active_channels.add(channel)
            object_id = str(item.get("object_id", ""))
            episode_key = (
                object_id,
                str(item.get("anchor_surface_id", item.get("surface_id", ""))),
            )
            anchor = self._robot_static_anchors.get(channel)
            if anchor is not None and anchor.get("episode_key") != episode_key:
                anchor = None
                self._robot_static_anchors.pop(channel, None)
            activation = float(item.get("activation", item.get("score", 0.0)))
            activation_declared = "activation" in item or "score" in item
            normal = np.asarray(
                item.get("surface_normal_solver", [0.0, 0.0, 1.0]),
                dtype=float,
            )
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            surface_value = item.get("surface_point_solver")
            normal_error = 0.0
            if surface_value is not None:
                surface = np.asarray(surface_value, dtype=float).reshape(3)
                if channel in getattr(self.contact, "support_geom_ids", {}):
                    support = self.contact.support_value(
                        self.configuration, channel, normal
                    )
                else:
                    support = self.contact.points[channel].value(
                        self.configuration
                    )
                normal_error = abs(float(
                    normal @ (support - surface)
                    - float(getattr(self.contact, "clearance", 0.0))
                ))
            bind_ready = (
                (activation >= bind_activation or not activation_declared)
                and normal_error <= bind_normal_error
            )
            created_anchor = False
            if anchor is None and bind_ready:
                tangent_value = getattr(self.contact, "tangent_value", None)
                if tangent_value is not None:
                    world_point = tangent_value(
                        self.configuration, channel
                    ).copy()
                else:
                    world_point = self.contact.points[channel].value(
                        self.configuration
                    ).copy()
                asset = assets.get(object_id)
                local_point = None
                if asset is not None:
                    pose = asset.pose_at(timestamp)
                    local_point = np.linalg.solve(
                        np.asarray(pose[:3, :3], dtype=float),
                        world_point - np.asarray(pose[:3, 3], dtype=float),
                    )
                anchor = {
                    "episode_key": episode_key,
                    "world_point": world_point,
                    "asset_local_point": local_point,
                    "feature": "configured_body_local_contact_proxy",
                    "inactive_steps": 0,
                    "age": 0,
                }
                self._robot_static_anchors[channel] = anchor
                created_anchor = True
            if anchor is None:
                item["robot_static_anchor_bound"] = False
                item["robot_static_anchor_normal_error"] = float(normal_error)
                contacts[channel] = item
                continue
            anchor["inactive_steps"] = 0
            if not created_anchor:
                anchor["age"] = int(anchor.get("age", 0)) + 1
            asset = assets.get(object_id)
            if asset is not None and anchor["asset_local_point"] is not None:
                pose = asset.pose_at(timestamp)
                tangent_anchor = (
                    np.asarray(pose[:3, :3], dtype=float)
                    @ np.asarray(anchor["asset_local_point"], dtype=float)
                    + np.asarray(pose[:3, 3], dtype=float)
                )
            else:
                tangent_anchor = np.asarray(anchor["world_point"], dtype=float)
            item["source_tangent_anchor_solver"] = np.asarray(
                item.get("tangent_anchor_solver", tangent_anchor), dtype=float
            ).copy()
            item["tangent_anchor_solver"] = tangent_anchor.copy()
            item["robot_tangent_anchor_provenance"] = "realized_robot_static_episode"
            item["robot_tangent_feature"] = str(anchor["feature"])
            item["robot_static_anchor_bound"] = True
            item["robot_static_anchor_age"] = int(anchor.get("age", 0))
            item["robot_static_anchor_normal_error"] = float(normal_error)
            contacts[channel] = item
        for channel in tuple(self._robot_static_anchors):
            if channel not in seen_channels:
                self._robot_static_anchors.pop(channel, None)
        return {**contact_frame, "contacts": contacts}

    def solve(self, motion, solver_frames, contacts, source_frames=None) -> RetargetResult:
        outputs=[]
        runtime_contact_frames = []
        solve_started = time.perf_counter()
        progress_interval = max(
            0,
            int(self.config.get("solver", {}).get("progress_interval", 0)),
        )
        first_iterations=int(self.config.get("solver",{}).get("first_frame_iterations",8)); iterations=int(self.config.get("solver",{}).get("iterations",4)); fatal_failure = None
        if len(solver_frames) != len(contacts):
            raise ValueError(f"V5 solver/contact timeline mismatch: {len(solver_frames)} != {len(contacts)}")
        if source_frames is not None and len(source_frames) != len(solver_frames):
            raise ValueError(f"V5 source/solver timeline mismatch: {len(source_frames)} != {len(solver_frames)}")
        if source_frames is not None:
            source_frames, solver_frames = self._prepare_robot_reference_motion(
                source_frames, solver_frames, contacts
            )
        for index,(frame,contact_frame) in enumerate(zip(solver_frames,contacts)):
            self._update_scene_time(index * self.dt)
            contact_frame = self._contact_frame_at_time(contact_frame, index * self.dt)
            source=source_frames[index] if source_frames is not None else {k:v[0] for k,v in frame.items()}
            root=frame.get("pelvis") or frame.get("root")
            if root is None: raise ValueError("V5 solver frame lacks pelvis/root")
            if index==0:
                self._seed_initial_pose(source, root[1])
            contact_frame = self._bind_robot_static_anchors(
                contact_frame, index * self.dt
            )
            self.terrain_limit.set_contact_scores({
                channel: float(item.get("activation", item.get("score", 0.0)))
                for channel, item in contact_frame.get("contacts", {}).items()
            })
            # Preserve the exact contact realization used by the task and
            # diagnostics.  In particular STATIC tangent anchors are bound to
            # the robot proxy here, so validators and exporters must not fall
            # back to the pre-solve source schedule.
            runtime_contact_frames.append(copy.deepcopy(contact_frame))
            # All ordinary and polish integrations for this output frame share
            # one motion budget.  Without an anchor, four IK passes plus
            # contact/collision polish could each satisfy the local velocity
            # limit while the exported frame still jumped discontinuously.
            frame_anchor_qpos = self.configuration.data.qpos.copy()
            self.frame_displacement.set_frame(
                frame_anchor_qpos, self.dt, enabled=bool(index)
            )
            # Config files retain the historical source labels for backwards
            # compatibility, but V5 consumes canonical semantic names first.
            semantics={}
            for name, spec in self.config["semantic_points"].items():
                # Adapter/solver-frame construction owns all source aliases;
                # the solver consumes canonical semantic names only.
                value = source.get(name)
                if value is None:
                    raise ValueError(f"V5 source frame is missing canonical semantic point {name!r}")
                semantics[name] = np.asarray(value, dtype=float)
            tasks = self.task_builder.build_tasks(semantics, frame, contact_frame)
            failures=[]; passes=first_iterations if index==0 else iterations
            polish_iterations = max(0, int(self.config.get("solver", {}).get("collision_polish_iterations", 2)))
            polish_count = 0
            collision_query_total = 0.0
            qp_solve_total = 0.0
            qp_iterations = 0
            qp_retries = 0
            nonlinear_safe_step_fraction = 1.0
            for _ in range(passes):
                collision_start = time.perf_counter()
                self.terrain_limit.prepare_active_set(self.configuration,self.dt/max(passes,1)); self.scene_collision.prepare_active_set(self.configuration,self.dt/max(passes,1))
                collision_query_time = time.perf_counter() - collision_start
                collision_query_total += collision_query_time
                limits=self.task_builder.build_limits(include_velocity=bool(index))
                solve_dt = self.dt / max(passes, 1)
                base_damping = float(self.config.get("solver", {}).get("damping", .1))
                base_radius = float(self.trust.radius)
                collision_radius = float(
                    self.config.get("solver", {}).get("collision_trust_region", 0.03)
                )
                collision_near = any(
                    float(item["distance"]) <= self.scene_collision.margin
                    + self.scene_collision.hard_band
                    + 0.01
                    for item in self.scene_collision.active_pairs
                )
                terrain_near = any(
                    float(item["signed_distance"]) <= self.terrain_limit.margin + 0.01
                    for item in self.terrain_limit.all_measurements
                )
                if collision_near or terrain_near:
                    self.trust.radius = min(base_radius, max(collision_radius, 1e-4))
                velocity = None
                retry_errors = []
                solve_start = time.perf_counter()
                for damping_factor, trust_factor in zip(
                    self.retry_damping_factors, self.retry_trust_factors
                ):
                    self.trust.radius = base_radius * trust_factor
                    try:
                        velocity = mink.solve_ik(
                            self.configuration,
                            tasks,
                            solve_dt,
                            self.solver,
                            base_damping * damping_factor,
                            limits=limits,
                        )
                        break
                    except Exception as error:
                        qp_retries += 1
                        retry_errors.append(f"{type(error).__name__}: {error}")
                self.trust.radius = base_radius
                if velocity is None:
                    # A second QP backend is a deterministic numerical retry,
                    # not a different retargeting algorithm.  Never export a
                    # silently partial frame as VALID.
                    try:
                        velocity = mink.solve_ik(
                            self.configuration, tasks, solve_dt, "proxqp",
                            base_damping * 6.0, limits=limits,
                        )
                    except Exception as error:
                        retry_errors.append(f"{type(error).__name__}: {error}")
                if velocity is None:
                    failures.extend(retry_errors)
                    break
                qp_solve_time = time.perf_counter() - solve_start
                qp_solve_total += qp_solve_time
                qp_iterations += 1
                self.configuration.integrate_inplace(velocity, solve_dt)
                self._project_limited_qpos()
            # A contact is a soft objective, so the full Omni/GMR solve may
            # legitimately leave a small normal residual when it competes
            # with interaction preservation.  Before exporting the frame,
            # perform a bounded feasibility polish for reliable contacts only.
            # This keeps the primary task and scene constraints intact while
            # preventing a supported foot from remaining visibly floating over
            # a stair tread.
            contact_cfg = self.config.get("contact_tasks", {})
            contact_polish_iterations = max(
                0, int(contact_cfg.get("polish_iterations", 1))
            )
            contact_polish_threshold = float(
                contact_cfg.get("polish_residual_threshold", 0.008)
            )
            support_polish_threshold = float(
                contact_cfg.get("support_polish_residual_threshold", 0.001)
            )
            support_polish_activation = float(
                contact_cfg.get("support_polish_activation", 0.25)
            )
            support_channels = {
                "left_heel", "left_toe", "right_heel", "right_toe"
            }
            contact_polish_failures = []
            contact_polish_attempts = []
            contact_polish_accepted_channels = []
            contact_unreachable_channels = []
            collision_polish_failures = []
            if not failures and contact_polish_iterations:
                full_contact_items = dict(contact_frame.get("contacts", {}))
                support_contact_items = {
                    channel: item
                    for channel, item in full_contact_items.items()
                    if channel in support_channels
                }
                self.contact.set_contacts(full_contact_items)
                # Tangential sticking remains part of the ordinary soft task;
                # enabling it inside the bounded polish is opt-in because an
                # unreachable static source anchor can otherwise turn a
                # feasible frame into repeated safety rollbacks.
                polish_tangent = bool(contact_cfg.get("polish_tangent", False))
                polish_tangent_threshold = float(contact_cfg.get(
                    "polish_tangent_threshold", contact_polish_threshold
                ))
                self.contact.normal_only = not polish_tangent
                for _ in range(contact_polish_iterations):
                    normal_residual, tangent_residual = self._contact_residuals(contact_frame)
                    support_normal_residual, support_tangent_residual = self._contact_residuals(
                        contact_frame,
                        support_channels,
                        activation_threshold=0.0,
                    )
                    if (
                        normal_residual <= contact_polish_threshold
                        and support_normal_residual <= support_polish_threshold
                        and (not polish_tangent or tangent_residual <= polish_tangent_threshold)
                    ):
                        break
                    # Probe support channels independently.  A combined
                    # heel/toe correction is overly conservative at stair
                    # edges: one foot can be feasible while the other foot's
                    # target requires a transient height jump or collides
                    # with a riser.  Keep every accepted correction, but only
                    # mark a channel unreachable after its own nonlinear
                    # feasibility check fails.
                    per_channel_polish = bool(
                        contact_cfg.get("per_channel_polish", True)
                    )
                    if (
                        per_channel_polish
                        and support_normal_residual > support_polish_threshold
                        and support_contact_items
                    ):
                        channel_residuals = self._contact_channel_residuals(
                            contact_frame,
                            support_channels,
                            activation_threshold=0.0,
                        )
                        candidates = sorted(
                            (
                                (residual, channel)
                                for channel, residual in channel_residuals.items()
                                if residual > support_polish_threshold
                            ),
                            reverse=True,
                        )
                        max_probe_attempts = max(
                            1,
                            int(contact_cfg.get("per_channel_max_attempts", 1)),
                        )
                        accepted = False
                        for residual, channel in candidates[:max_probe_attempts]:
                            ok, reason, snapshot = self._try_contact_channel_correction(
                                channel,
                                support_contact_items[channel],
                                contact_frame,
                                solve_dt,
                                contact_cfg,
                                support_polish_activation,
                            )
                            attempt = {
                                "channel": channel,
                                "residual_before": float(residual),
                                "accepted": bool(ok),
                                "reason": str(reason),
                                "constraints": snapshot,
                            }
                            contact_polish_attempts.append(attempt)
                            if ok:
                                contact_polish_accepted_channels.append(channel)
                                accepted = True
                                break
                            contact_unreachable_channels.append(channel)
                            contact_polish_failures.append(
                                f"{channel}: {reason}"
                            )
                        if accepted:
                            # Recompute all residuals at the newly accepted
                            # configuration on the next bounded iteration.
                            continue
                        if candidates:
                            contact_polish_failures.append(
                                "no feasible per-channel support correction"
                            )
                            break
                    # When only the physical foot support is out of tolerance,
                    # solve a support-only polish.  Butt/back/palm normals
                    # remain in the ordinary contact objective, but cannot
                    # consume the limited correction budget needed to bring a
                    # real heel/toe collision surface onto its support plane.
                    self.contact.set_contacts(
                        support_contact_items
                        if support_normal_residual > support_polish_threshold
                        and support_contact_items
                        else full_contact_items
                    )
                    # A source-confirmed support channel may still be inside
                    # the detector's entry ramp.  Give only this bounded
                    # support correction a modest minimum weight; ordinary
                    # IK and all non-foot contacts keep their smooth source
                    # activation unchanged.
                    self.contact.activation_floor = (
                        support_polish_activation
                        if support_normal_residual > support_polish_threshold
                        and support_contact_items
                        else 0.0
                    )
                    # Contact polish integrates the configuration between
                    # iterations.  Re-linearize both hard limits at that
                    # updated state; reusing the previous pass's active set
                    # can push a foot/butt into a newly encountered scene
                    # facet and make the subsequent collision repair QP
                    # artificially infeasible.
                    self.terrain_limit.prepare_active_set(self.configuration, solve_dt)
                    self.scene_collision.prepare_active_set(self.configuration, solve_dt)
                    limits = self.task_builder.build_limits(include_velocity=bool(index))
                    original_radius = self.trust.radius
                    self.trust.radius = min(
                        original_radius,
                        max(float(contact_cfg.get("polish_trust_region", 0.02)), 1e-4),
                    )
                    try:
                        correction = mink.solve_ik(
                            self.configuration,
                            [self.contact],
                            solve_dt,
                            self.solver,
                            float(self.config.get("solver", {}).get("damping", .1)),
                            limits=limits,
                        )
                    except Exception as error:
                        # Contact polish is an optional soft-objective
                        # refinement.  A conflicting contact/scene geometry
                        # must not invalidate an otherwise feasible primary
                        # QP result; retain the frame and expose the event in
                        # diagnostics instead of treating it as a solver
                        # failure.
                        contact_polish_failures.append(
                            f"{type(error).__name__}: {error}"
                        )
                        correction = None
                    self.trust.radius = original_radius
                    if correction is None or not np.isfinite(correction).all():
                        if correction is not None:
                            contact_polish_failures.append(
                                "produced non-finite correction"
                            )
                        break
                    polish_qpos = self.configuration.data.qpos.copy()
                    self.configuration.integrate_inplace(correction, solve_dt)
                    self._project_limited_qpos()
                    # The contact objective is soft; it must never leave a
                    # nonlinear terrain/scene violation for the later hard
                    # repair pass.  CoACD seams can invalidate a linearized
                    # collision row after a seemingly feasible integration.
                    # Roll back only this polish step and keep the last safe
                    # configuration, so the next output frame can retry from
                    # a valid state without turning a soft-task conflict into
                    # a fatal QP failure.
                    self.terrain_limit.prepare_active_set(self.configuration, solve_dt)
                    self.scene_collision.prepare_active_set(self.configuration, solve_dt)
                    candidate_snapshot = self._constraint_snapshot(solve_dt)
                    terrain_violation, _ = self._terrain_violation_state(1e-6)
                    collision_violation = any(
                        float(item["distance"]) < self.scene_collision.margin - 1e-6
                        for item in self.scene_collision.active_pairs
                    )
                    if terrain_violation or collision_violation:
                        self.configuration.update(polish_qpos)
                        self.terrain_limit.prepare_active_set(self.configuration, solve_dt)
                        self.scene_collision.prepare_active_set(self.configuration, solve_dt)
                        rollback_snapshot = self._constraint_snapshot(solve_dt)
                        contact_polish_failures.append(
                            "rolled back soft contact correction that violated "
                            "terrain/scene non-penetration"
                        )
                        combined_attempt = {
                            "channel": "combined",
                            "residual_before": float(support_normal_residual),
                            "accepted": False,
                            "reason": "combined correction violated nonlinear hard limits",
                            "constraints": {
                                "candidate": candidate_snapshot,
                                "restored": rollback_snapshot,
                            },
                        }
                        contact_polish_attempts.append(combined_attempt)
                        # A joint correction can be rejected because one
                        # channel collides with a riser/chair facet even
                        # though another channel has a locally feasible ankle
                        # correction.  Retry only after the combined attempt
                        # has been restored; ordinary frames retain the stable
                        # joint correction path above.
                        fallback_enabled = bool(
                            contact_cfg.get("per_channel_fallback", True)
                        )
                        fallback_accepted = False
                        if fallback_enabled and support_contact_items:
                            channel_residuals = self._contact_channel_residuals(
                                contact_frame,
                                support_channels,
                                activation_threshold=0.0,
                            )
                            candidates = sorted(
                                (
                                    (residual, channel)
                                    for channel, residual in channel_residuals.items()
                                    if residual > support_polish_threshold
                                ),
                                reverse=True,
                            )
                            max_probe_attempts = max(
                                1,
                                int(contact_cfg.get("per_channel_max_attempts", 1)),
                            )
                            for residual, channel in candidates[:max_probe_attempts]:
                                ok, reason, snapshot = self._try_contact_channel_correction(
                                    channel,
                                    support_contact_items[channel],
                                    contact_frame,
                                    solve_dt,
                                    contact_cfg,
                                    support_polish_activation,
                                )
                                contact_polish_attempts.append({
                                    "channel": channel,
                                    "residual_before": float(residual),
                                    "accepted": bool(ok),
                                    "reason": str(reason),
                                    "constraints": snapshot,
                                })
                                if ok:
                                    contact_polish_accepted_channels.append(channel)
                                    fallback_accepted = True
                                    break
                                contact_unreachable_channels.append(channel)
                                contact_polish_failures.append(
                                    f"{channel}: {reason}"
                                )
                        if fallback_accepted:
                            # Continue the bounded polish loop from the
                            # accepted single-channel state.  The next pass
                            # re-evaluates every contact and can still perform
                            # a combined correction if it becomes feasible.
                            continue
                        break
                self.contact.normal_only = False
                self.contact.activation_floor = 0.0
                self.contact.set_contacts(full_contact_items)
            # The final integration can cross a nonlinear mesh/CoACD seam
            # that was not active at the start of the last ordinary pass.
            # Re-query at the actual post-integration configuration and apply
            # a bounded correction so exported qpos never relies on the next
            # frame to discover a scene penetration.
            for _ in range(polish_iterations):
                if failures:
                    break
                self.terrain_limit.prepare_active_set(self.configuration, solve_dt)
                self.scene_collision.prepare_active_set(self.configuration, solve_dt)
                terrain_violation, _ = self._terrain_violation_state(1e-6)
                collision_violation = any(
                    float(item["distance"]) < self.scene_collision.margin - 1e-6
                    for item in self.scene_collision.active_pairs
                )
                if not (terrain_violation or collision_violation):
                    break
                polish_count += 1
                limits = self.task_builder.build_limits(include_velocity=bool(index))
                polish_radius = float(
                    self.config.get("solver", {}).get("collision_polish_trust_region", 0.025)
                )
                original_radius = self.trust.radius
                self.trust.radius = min(original_radius, max(polish_radius, 1e-4))
                # Collision polish is a feasibility projection, not a second
                # retargeting solve.  A zero-task DAQP solve can nevertheless
                # be numerically singular when several local contact normals
                # meet at a convex-decomposition seam.  Retry the *same*
                # frozen active set with the alternate backend before marking
                # the frame failed; this does not change task priorities or
                # relax the collision inequality.
                correction = None
                polish_errors = []
                for backend in (self.solver, "proxqp"):
                    try:
                        correction = mink.solve_ik(
                            self.configuration,
                            [],
                            solve_dt,
                            backend,
                            float(self.config.get("solver", {}).get("damping", .1)) * 2.0,
                            limits=limits,
                        )
                        if correction is not None and np.isfinite(correction).all():
                            break
                        polish_errors.append(f"{backend}: non-finite correction")
                        correction = None
                    except Exception as error:
                        polish_errors.append(
                            f"{backend} {type(error).__name__}: {error}"
                        )
                if correction is None:
                    # A nonlinear seam can make the local collision repair
                    # infeasible even though the previous exported frame was
                    # valid.  Recover by restoring the frame-start state and
                    # replaying the hard queries at the current scene pose;
                    # never disable the limit or export the penetrating
                    # intermediate state.  If the anchor itself is unsafe,
                    # retain the fatal failure because no local recovery is
                    # physically justified.
                    self.configuration.update(frame_anchor_qpos)
                    self.terrain_limit.prepare_active_set(self.configuration, solve_dt)
                    self.scene_collision.prepare_active_set(self.configuration, solve_dt)
                    anchor_terrain_violation, _ = self._terrain_violation_state(1e-6)
                    anchor_collision_violation = any(
                        float(item["distance"]) < self.scene_collision.margin - 1e-6
                        for item in self.scene_collision.active_pairs
                    )
                    if not anchor_terrain_violation and not anchor_collision_violation:
                        collision_polish_failures.append(
                            "rolled back frame after infeasible collision polish: "
                            + "; ".join(polish_errors)
                        )
                        self.trust.radius = original_radius
                        break
                    failures.append("collision_polish " + "; ".join(polish_errors))
                    self.trust.radius = original_radius
                    break
                self.trust.radius = original_radius
                self.configuration.integrate_inplace(correction, solve_dt)
                self._project_limited_qpos()
            # Never export a configuration that remains inside a scene geom
            # after the bounded nonlinear repair.  CoACD stair seams can make
            # a local linear projection converge to a different facet while
            # leaving the original pair slightly penetrating.  Replaying the
            # frame-start state is the only conservative recovery that keeps
            # the robot collision-free without teleporting the root or
            # disabling scene collision for this frame.
            remaining_terrain_violation, remaining_collision_violation = (
                self._hard_constraint_violations(solve_dt)
            )
            if (remaining_terrain_violation or remaining_collision_violation) and not failures:
                proposed_qpos = self.configuration.data.qpos.copy()
                fraction, safe = self._backtrack_to_safe_configuration(
                    frame_anchor_qpos, proposed_qpos, solve_dt
                )
                nonlinear_safe_step_fraction = min(
                    nonlinear_safe_step_fraction, fraction
                )
                if safe:
                    collision_polish_failures.append(
                        "clipped nonlinear IK step to collision-free fraction "
                        f"{fraction:.6f} after bounded collision polish"
                    )
                else:
                    failures.append(
                        "collision repair left a penetration and the frame-start "
                        "configuration was also unsafe"
                    )
            # Diagnostics and the exported frame must describe the same
            # post-polish configuration, not the pre-correction active set.
            self.terrain_limit.prepare_active_set(self.configuration, solve_dt)
            self.scene_collision.prepare_active_set(self.configuration, solve_dt)
            output=self.configuration.data.qpos.copy()
            self.foot_temporal.end_frame(self.configuration)
            interaction_error = float(np.linalg.norm(self.interaction.compute_error(self.configuration)))
            contact_metrics = {}
            for channel, item in contact_frame.get("contacts", {}).items():
                actual_normal = 0.0
                actual_tangent = 0.0
                robot_point = None
                if channel in self.contact.points and str(item.get("state", "NONE")) != "NONE":
                    state = str(item.get("state", "NONE"))
                    normal = np.asarray(item.get(
                        "anchor_normal_solver" if state == "STATIC" else "surface_normal_solver",
                        [0., 0., 1.],
                    ), dtype=float)
                    normal /= max(float(np.linalg.norm(normal)), 1e-12)
                    if channel in self.contact.support_geom_ids:
                        robot_point = self.contact.support_value(
                            self.configuration, channel, normal
                        )
                    else:
                        robot_point = self.contact.points[channel].value(self.configuration)
                    state = str(item.get("state", "NONE"))
                    normal_key = "anchor_normal_solver" if state == "STATIC" and "anchor_normal_solver" in item else "surface_normal_solver"
                    normal = np.asarray(item.get(normal_key, [0., 0., 1.]), dtype=float)
                    normal /= max(float(np.linalg.norm(normal)), 1e-12)
                    surface = np.asarray(item.get("surface_point_solver", robot_point), dtype=float)
                    actual_normal = abs(float(normal @ (robot_point - surface) - self.contact.clearance))
                    anchor = np.asarray(item.get("tangent_anchor_solver", surface), dtype=float)
                    tangent = self.contact._tangent_basis(normal)
                    if state == "STATIC":
                        tangent_value = getattr(self.contact, "tangent_value", None)
                        tangent_point = (
                            tangent_value(self.configuration, channel)
                            if tangent_value is not None
                            else self.contact.points[channel].value(self.configuration)
                        )
                        if bool(item.get("robot_static_anchor_bound", True)):
                            actual_tangent = float(
                                np.linalg.norm(tangent @ (tangent_point - anchor))
                            )
                contact_metrics[channel] = {
                    "source_state": str(item.get("source_state", item.get("state", "NONE"))),
                    "robot_state": str(item.get("state", "NONE")),
                    "activation": float(item.get("activation", item.get("score", 0.0))),
                    "object_id": str(item.get("object_id", "")),
                    "surface_id": str(item.get("surface_id", "")),
                    "signed_distance": float(item.get("signed_distance", np.nan)),
                    "contact_distance": float(item.get("contact_distance", np.nan)),
                    "normal_error": actual_normal,
                    "tangent_error": actual_tangent,
                    "robot_point": None if robot_point is None else robot_point.copy(),
                    "target_surface_point": np.asarray(item.get("surface_point_solver", [0., 0., 0.]), dtype=float).copy(),
                }
            terrain_distances = [float(item["signed_distance"]) for item in self.terrain_limit.all_measurements]
            terrain_slacks = [float(item["slack"]) for item in self.terrain_limit.all_measurements]
            self.diagnostics.append({"frame":index,"qp_failures":failures,"contact_polish_failures":contact_polish_failures,"contact_polish_attempts":contact_polish_attempts,"contact_polish_accepted_channels":contact_polish_accepted_channels,"contact_unreachable_channels":sorted(set(contact_unreachable_channels)),"collision_polish_failures":collision_polish_failures,"nonlinear_safe_step_fraction":float(nonlinear_safe_step_fraction),"interaction_error":interaction_error,"interaction_scene_points":int(self.interaction.environment_count),"interaction_scene_selected_points":int(self.interaction.environment_count),"minimum_terrain_distance":float(min(terrain_distances,default=np.inf)),"minimum_terrain_slack":float(min(terrain_slacks,default=np.inf)),"active_terrain_constraints":len(self.terrain_limit.active),"terrain_candidate_count":len(self.terrain_limit.all_measurements),"terrain_active_indices":[int(item["index"]) for item in self.terrain_limit.all_measurements if item.get("active")],"scene_collision_candidate_pairs":int(len(self.scene_collision.robot_geoms)*len(self.scene_collision.scene_geoms)),"scene_collision_exact_query_pairs":int(self.scene_collision.exact_query_pairs),"scene_collision_broadphase_culled_pairs":int(self.scene_collision.broadphase_culled_pairs),"scene_collision_active_pairs":len(self.scene_collision.active_pairs),"scene_collision_polish_iterations":int(polish_count),"scene_collision_active_pair_details":[{"robot_geom":mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_GEOM,int(x["robot_geom"])),"scene_geom":mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_GEOM,int(x["scene_geom"])),"distance":float(x["distance"])} for x in self.scene_collision.active_pairs],"minimum_scene_distance":float(self.scene_collision.minimum_distance),"maximum_scene_penetration":float(self.scene_collision.maximum_penetration),"collision_query_time":float(collision_query_total),"scene_collision_query_runtime_seconds":float(collision_query_total),"qp_solve_time":float(qp_solve_total),"qp_solve_runtime_seconds":float(qp_solve_total),"qp_iterations":int(qp_iterations),"qp_retries":int(qp_retries),"contacts":contact_metrics,"contact_states":contact_metrics})
            if progress_interval and (
                index % progress_interval == 0 or index == len(solver_frames) - 1
            ):
                elapsed = max(time.perf_counter() - solve_started, 1e-9)
                rate = float(index + 1) / elapsed
                print(
                    f"[V5] frame {index + 1}/{len(solver_frames)} "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f} fps "
                    f"qp_failures={len(failures)}",
                    flush=True,
                )
            if failures:
                fatal_failure = f"Frame {index}: {failures[-1]}"
                break
            outputs.append(output)
            self.previous=output.copy()
            self.frame_index+=1
        status="VALID" if fatal_failure is None else "INVALID"
        runtime_plan = ContactPlan(
            episodes=(),
            per_frame_states=tuple(runtime_contact_frames),
            fps=self.fps,
            metadata={
                "provenance": "v5_solver_runtime_realization",
                "source_frame_count": len(contacts),
                "robot_reference": copy.deepcopy(self.reference_metadata),
            },
        )
        return RetargetResult(
            np.asarray(outputs, dtype=float).reshape((-1, self.model.nq)),
            self.fps, self.diagnostics,
            contact_plan=runtime_plan, status=status, failure=fatal_failure
        )

    def forward_kinematics(self, qpos_sequence: np.ndarray) -> dict[str, np.ndarray]:
        """Recompute exported body states from final qpos, never solver caches."""
        qpos_sequence = np.asarray(qpos_sequence, dtype=float)
        body_names = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i) or str(i)
                      for i in range(self.model.nbody)]
        positions = np.empty((len(qpos_sequence), self.model.nbody, 3), dtype=float)
        quaternions = np.empty((len(qpos_sequence), self.model.nbody, 4), dtype=float)
        for frame, qpos in enumerate(qpos_sequence):
            self.configuration.update(qpos)
            positions[frame] = self.configuration.data.xpos
            quaternions[frame] = self.configuration.data.xquat
        linear = np.zeros_like(positions); angular = np.zeros_like(positions)
        if len(qpos_sequence) > 1:
            linear[1:] = np.diff(positions, axis=0) / self.dt
            linear[0] = linear[1]
            for frame in range(1, len(qpos_sequence)):
                for body in range(self.model.nbody):
                    previous = Rotation.from_quat(quaternions[frame - 1, body], scalar_first=True)
                    current = Rotation.from_quat(quaternions[frame, body], scalar_first=True)
                    angular[frame, body] = (previous.inv() * current).as_rotvec() / self.dt
            angular[0] = angular[1]
        joint_velocity = np.zeros((len(qpos_sequence), self.model.nv), dtype=float)
        for frame in range(1, len(qpos_sequence)):
            mj.mj_differentiatePos(self.model, joint_velocity[frame], self.dt,
                                    qpos_sequence[frame - 1], qpos_sequence[frame])
        if len(qpos_sequence) > 1:
            joint_velocity[0] = joint_velocity[1]
        robot_joint_names = []
        robot_qpos_indices = []
        robot_dof_indices = []
        for joint_id in range(self.model.njnt):
            joint_type = self.model.jnt_type[joint_id]
            if not is_dof_joint_type(joint_type):
                continue
            robot_joint_names.append(
                mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, joint_id) or str(joint_id)
            )
            robot_qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
            robot_dof_indices.append(int(self.model.jnt_dofadr[joint_id]))
        return {"body_names": body_names, "body_pos_w": positions, "body_quat_w": quaternions,
                "body_lin_vel_w": linear, "body_ang_vel_w": angular,
                "joint_vel": joint_velocity[:, robot_dof_indices],
                "joint_pos": qpos_sequence[:, robot_qpos_indices],
                "robot_joint_names": robot_joint_names,
                "robot_dof_names": robot_joint_names}


WholeBodyOmniGMRV5 = WholeBodyRetargetSolver
