"""WholeBody V5 solver.

V5 owns its orchestration and QP construction.  It does not subclass or call
the historical V3/V4 retargeters.  The only deliberately shared pieces are
the canonical motion adapters, terrain geometry and the NE01 configuration
format.  This keeps the old pipelines available as regression baselines while
making the task-level data flow explicit.
"""

from __future__ import annotations

import json
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
from .robot_profile import RobotProfile
from .task_builder import TaskBuilder
from .terrain_geometry import TerrainField


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
        self.target_position = np.zeros(3); self.target_yaw = 0.0
        super().__init__(cost=np.asarray(cost, dtype=float), gain=.45, lm_damping=1.0)

    def set_target(self, position, quaternion):
        self.target_position = np.asarray(position, dtype=float)
        self.target_yaw = float(Rotation.from_quat(quaternion, scalar_first=True).as_euler("zyx")[0])

    def compute_error(self, configuration):
        p = configuration.data.xpos[self.body_id]
        r = Rotation.from_matrix(configuration.data.xmat[self.body_id].reshape(3, 3))
        yaw = float(r.as_euler("zyx")[0]); error = (yaw - self.target_yaw + np.pi) % (2*np.pi) - np.pi
        return np.r_[p - self.target_position, error]

    def compute_jacobian(self, configuration):
        jp = np.zeros((3, self.model.nv)); jr = np.zeros_like(jp)
        mj.mj_jacBody(self.model, configuration.data, jp, jr, self.body_id)
        return np.vstack((jp, np.array([0., 0., 1.]) @ jr))


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
        # Every channel has one normal and two tangent rows.  Keeping the row
        # layout fixed is important: changing task dimensions as contacts
        # enter/leave would make the QP warm-start and damping discontinuous.
        super().__init__(cost=np.zeros(3 * len(names)), gain=.5, lm_damping=1.0)

    def set_contacts(self, contacts): self.contacts = contacts or {}

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
            point = self.points[name].value(configuration)
            jacobian = self.points[name].jacobian(configuration)
            state = str(item.get("state", "NONE"))
            # STATIC episodes own one surface frame.  Reusing the per-frame
            # triangle hit here makes a tessellated chair seat move the
            # target under the robot and is the main source of seated drift.
            # SLIDING contacts intentionally follow the current surface hit.
            normal_key = "anchor_normal_solver" if state == "STATIC" and "anchor_normal_solver" in item else "surface_normal_solver"
            normal = np.asarray(item.get(normal_key, [0., 0., 1.]), dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            surface_key = "tangent_anchor_solver" if state == "STATIC" and "tangent_anchor_solver" in item else "surface_point_solver"
            surface = np.asarray(item.get(surface_key, [0., 0., 0.]), dtype=float)
            # ``state`` is the discrete diagnostic label; ``activation`` is
            # the frame-smoothed task weight supplied by the source contact
            # detector.  Keeping these separate avoids a hard QP objective
            # discontinuity when a heel or butt contact enters/leaves.
            active = float(item.get("activation", item.get("score", 0.0)))
            tangent = self._tangent_basis(normal)
            # STATIC contacts stick to the episode anchor.  SLIDING contacts
            # retain normal contact but follow the currently observed tangent
            # point, so they cannot be accidentally locked to world XY.
            anchor = np.asarray(
                item.get("tangent_anchor_solver", item.get("surface_point_solver", surface)),
                dtype=float,
            )
            target = anchor if state == "STATIC" else surface
            errors.extend((normal @ (point - surface) - self.clearance,
                           *(tangent @ (point - target))))
            jacobians.extend((normal @ jacobian, tangent[0] @ jacobian, tangent[1] @ jacobian))
            weights.extend((self.normal_cost * active,
                            self.tangent_cost * active if state == "STATIC" else 0.,
                            self.tangent_cost * active if state == "STATIC" else 0.))
        return np.asarray(errors), np.asarray(jacobians), np.asarray(weights)

    def compute_error(self, configuration):
        error, _, weights = self._rows(configuration)
        self.cost = weights
        return error

    def compute_jacobian(self, configuration):
        _, jacobian, weights = self._rows(configuration)
        self.cost = weights
        return np.asarray(jacobian)


class TerrainNonPenetrationLimit(Limit):
    def __init__(self, model, terrain: TerrainField, config):
        self.model = model; self.terrain = terrain; self.config = dict(config)
        self.margin = float(config.get("margin", .004)); self.activate = float(config.get("activate_distance", .06))
        self.shells = self._discover(); self.active = []

    def _discover(self):
        shells = []
        scene_bodies = {
            body_id
            for body_id in range(self.model.nbody)
            if (mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body_id) or "").startswith("scene_")
        }
        raw_shells = []
        for body_id in range(1, self.model.nbody):
            if body_id in scene_bodies:
                continue
            verts = []
            for gid in range(self.model.ngeom):
                if int(self.model.geom_bodyid[gid]) != body_id or self.model.geom_type[gid] != mj.mjtGeom.mjGEOM_MESH: continue
                mid = int(self.model.geom_dataid[gid]); start = int(self.model.mesh_vertadr[mid]); count = int(self.model.mesh_vertnum[mid])
                # Mesh vertices are in mesh coordinates.  Keep the geom-local
                # transform so shell samples and their Jacobians refer to the
                # same physical surface.
                geom_rotation = np.zeros((3, 3), dtype=float)
                mj.mju_quat2Mat(geom_rotation.reshape(-1), self.model.geom_quat[gid])
                verts.append(
                    (np.asarray(self.model.mesh_vert[start:start+count]) @ geom_rotation.T)
                    + self.model.geom_pos[gid]
                )
            if verts:
                samples = np.concatenate(verts)
                raw_shells.append((body_id, samples))
        # A fixed, deterministic proxy budget keeps exact terrain queries
        # bounded for high-resolution robot meshes while retaining coverage
        # on every articulated body.  The hard constraint still evaluates the
        # exact terrain SDF at each selected point.
        budget = max(1, int(self.config.get("mesh_proxy_points", 96)))
        per_shell = max(4, int(np.ceil(budget / max(1, len(raw_shells)))) )
        for body_id, samples in raw_shells:
            stride = max(1, int(np.ceil(len(samples) / per_shell)))
            selected = samples[::stride][:per_shell]
            shells.append((body_id, selected))
        return shells

    def prepare_active_set(self, configuration, dt=0.):
        del dt; configuration.update(); active=[]
        for body_id, local in self.shells:
            rot = configuration.data.xmat[body_id].reshape(3, 3)
            points = configuration.data.xpos[body_id] + local @ rot.T
            hits = self.terrain.nearest_surface_batch(points)
            for point, hit in zip(points, hits):
                if hit.signed_distance <= self.activate: active.append((body_id, point.copy(), hit))
        self.active = active

    def measure_all(self, configuration):
        """Return every discovered proxy's current terrain hit and slack."""
        configuration.update(); values = []
        for body_id, local in self.shells:
            rotation = configuration.data.xmat[body_id].reshape(3, 3)
            points = configuration.data.xpos[body_id] + local @ rotation.T
            for point, hit in zip(points, self.terrain.nearest_surface_batch(points)):
                values.append({"body_id": int(body_id), "point": point.copy(),
                               "signed_distance": float(hit.signed_distance),
                               "margin": float(self.margin),
                               "slack": float(hit.signed_distance - self.margin),
                               "surface_id": str(hit.surface_id),
                               "surface_normal": np.asarray(hit.normal).copy()})
        return values

    def compute_qp_inequalities(self, configuration, dt):
        del dt; rows=[]; bounds=[]
        for body_id, point, hit in self.active:
            jp=np.zeros((3,self.model.nv)); jr=np.zeros_like(jp); mj.mj_jac(self.model, configuration.data, jp, jr, point, body_id)
            rows.append(-hit.normal @ jp); bounds.append(float(hit.signed_distance - self.margin))
        return Constraint(G=np.asarray(rows), h=np.asarray(bounds)) if rows else Constraint()


class _SceneCollisionLimit(Limit):
    """Independent MuJoCo robot-scene active-set limit for V5."""
    def __init__(self, model, config):
        self.model = model
        self.enabled = bool(config.get("enabled", True))
        self.activate = float(config.get("activate_distance", .06))
        self.margin = float(config.get("margin", .004))
        self.hard_band = max(0.0, float(config.get("hard_activation_band", 0.0)))
        prefix = str(config.get("scene_body_prefix", "scene_"))
        self.scene_geoms = [g for g in range(model.ngeom) if model.geom_contype[g] and (mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or "").startswith(prefix)]
        self.robot_geoms = [g for g in range(model.ngeom) if model.geom_contype[g] and g not in self.scene_geoms and int(model.geom_bodyid[g]) != 0]
        self.active_pairs = []
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
                # Positive-distance pairs farther than the hard safety band
                # are useful diagnostics but needlessly constrain the motion
                # objective.  Any penetration (distance < 0) and every pair
                # inside margin remain unconditionally active.
                if distance > self.margin + self.hard_band:
                    continue
                pairs.append({"robot_geom":robot_geom,"scene_geom":scene_geom,"distance":distance,"normal":sign * vector/norm,"robot_point":fromto[:3].copy()})
        # A convex decomposition can return many overlapping/near-identical
        # pieces for one robot geom.  Enforcing every pair at once creates
        # contradictory normals at a penetration seam.  Keep the closest
        # deterministic pair per robot geom; the next SQP substep re-queries
        # after the robot has moved away.
        selected = {}
        for item in pairs:
            key = int(item["robot_geom"])
            previous = selected.get(key)
            if previous is None or (item["distance"], item["scene_geom"]) < (previous["distance"], previous["scene_geom"]):
                selected[key] = item
        self.active_pairs = list(selected.values())
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
        self.terrain_limit=TerrainNonPenetrationLimit(self.model,terrain,self.config.get("terrain_nonpenetration",{})); self.trust=_TrustLimit(self.model,self.config.get("solver",{}).get("trust_region",.12)); self.scene_collision=_SceneCollisionLimit(self.model,self.config.get("scene_collision",{}))
        self.config_limit=mink.ConfigurationLimit(self.model); self.velocity=mink.VelocityLimit(self.model,{mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_JOINT,i):float(self.config.get("solver",{}).get("joint_velocity_limit",10.)) for i in range(self.model.njnt) if self.model.jnt_type[i] in (mj.mjtJoint.mjJNT_HINGE,mj.mjtJoint.mjJNT_SLIDE)})
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

    def solve(self, motion, solver_frames, contacts, source_frames=None) -> RetargetResult:
        outputs=[]; first_iterations=int(self.config.get("solver",{}).get("first_frame_iterations",8)); iterations=int(self.config.get("solver",{}).get("iterations",4)); fatal_failure = None
        if len(solver_frames) != len(contacts):
            raise ValueError(f"V5 solver/contact timeline mismatch: {len(solver_frames)} != {len(contacts)}")
        if source_frames is not None and len(source_frames) != len(solver_frames):
            raise ValueError(f"V5 source/solver timeline mismatch: {len(source_frames)} != {len(solver_frames)}")
        for index,(frame,contact_frame) in enumerate(zip(solver_frames,contacts)):
            self._update_scene_time(index * self.dt)
            contact_frame = self._contact_frame_at_time(contact_frame, index * self.dt)
            source=source_frames[index] if source_frames is not None else {k:v[0] for k,v in frame.items()}
            root=frame.get("pelvis") or frame.get("root")
            if root is None: raise ValueError("V5 solver frame lacks pelvis/root")
            if index==0:
                self._seed_initial_pose(source, root[1])
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
                    float(item[2].signed_distance) <= self.terrain_limit.margin + 0.01
                    for item in self.terrain_limit.active
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
                terrain_violation = any(
                    float(item[2].signed_distance) < self.terrain_limit.margin
                    for item in self.terrain_limit.active
                )
                collision_violation = any(
                    float(item["distance"]) < self.scene_collision.margin
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
                try:
                    correction = mink.solve_ik(
                        self.configuration,
                        # A polish step is a feasibility projection, not a
                        # second retargeting solve.  Reapplying the full
                        # interaction/contact objective here can trade a
                        # small collision violation for a large limb jump.
                        # An empty task list gives the QP a minimum-norm
                        # correction under the same hard limits.
                        [],
                        solve_dt,
                        self.solver,
                        float(self.config.get("solver", {}).get("damping", .1)) * 2.0,
                        limits=limits,
                    )
                except Exception as error:
                    failures.append(f"collision_polish {type(error).__name__}: {error}")
                    self.trust.radius = original_radius
                    break
                self.trust.radius = original_radius
                if correction is None or not np.isfinite(correction).all():
                    failures.append("collision_polish produced non-finite correction")
                    break
                self.configuration.integrate_inplace(correction, solve_dt)
            # Diagnostics and the exported frame must describe the same
            # post-polish configuration, not the pre-correction active set.
            self.terrain_limit.prepare_active_set(self.configuration, solve_dt)
            self.scene_collision.prepare_active_set(self.configuration, solve_dt)
            output=self.configuration.data.qpos.copy()
            interaction_error = float(np.linalg.norm(self.interaction.compute_error(self.configuration)))
            contact_metrics = {}
            for channel, item in contact_frame.get("contacts", {}).items():
                actual_normal = 0.0
                actual_tangent = 0.0
                robot_point = None
                if channel in self.contact.points and str(item.get("state", "NONE")) != "NONE":
                    robot_point = self.contact.points[channel].value(self.configuration)
                    state = str(item.get("state", "NONE"))
                    normal_key = "anchor_normal_solver" if state == "STATIC" and "anchor_normal_solver" in item else "surface_normal_solver"
                    normal = np.asarray(item.get(normal_key, [0., 0., 1.]), dtype=float)
                    normal /= max(float(np.linalg.norm(normal)), 1e-12)
                    surface_key = "tangent_anchor_solver" if state == "STATIC" and "tangent_anchor_solver" in item else "surface_point_solver"
                    surface = np.asarray(item.get(surface_key, robot_point), dtype=float)
                    actual_normal = abs(float(normal @ (robot_point - surface) - self.contact.clearance))
                    anchor = np.asarray(item.get("tangent_anchor_solver", surface), dtype=float)
                    tangent = self.contact._tangent_basis(normal)
                    if state == "STATIC":
                        actual_tangent = float(np.linalg.norm(tangent @ (robot_point - anchor)))
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
            self.diagnostics.append({"frame":index,"qp_failures":failures,"interaction_error":interaction_error,"interaction_scene_points":int(self.interaction.environment_count),"minimum_terrain_distance":float(min((x[2].signed_distance for x in self.terrain_limit.active),default=np.inf)),"active_terrain_constraints":len(self.terrain_limit.active),"scene_collision_candidate_pairs":int(len(self.scene_collision.robot_geoms)*len(self.scene_collision.scene_geoms)),"scene_collision_exact_query_pairs":int(self.scene_collision.exact_query_pairs),"scene_collision_broadphase_culled_pairs":int(self.scene_collision.broadphase_culled_pairs),"scene_collision_active_pairs":len(self.scene_collision.active_pairs),"scene_collision_polish_iterations":int(polish_count),"scene_collision_active_pair_details":[{"robot_geom":mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_GEOM,int(x["robot_geom"])),"scene_geom":mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_GEOM,int(x["scene_geom"])),"distance":float(x["distance"])} for x in self.scene_collision.active_pairs],"minimum_scene_distance":float(self.scene_collision.minimum_distance),"maximum_scene_penetration":float(self.scene_collision.maximum_penetration),"collision_query_time":float(collision_query_total),"scene_collision_query_runtime_seconds":float(collision_query_total),"qp_solve_time":float(qp_solve_total),"qp_solve_runtime_seconds":float(qp_solve_total),"qp_iterations":int(qp_iterations),"qp_retries":int(qp_retries),"contacts":contact_metrics,"contact_states":contact_metrics})
            if failures:
                fatal_failure = f"Frame {index}: {failures[-1]}"
                break
            outputs.append(output)
            self.previous=output.copy()
            self.frame_index+=1
        status="VALID" if fatal_failure is None else "INVALID"
        return RetargetResult(np.asarray(outputs),self.fps,self.diagnostics,status=status, failure=fatal_failure)

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
            if joint_type not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
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
