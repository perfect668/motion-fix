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
from .terrain_geometry import TerrainField


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
                costs[int(model.joint(name).dofadr)] = float(config.get(f"{key}_cost", config.get("cost", 0.25)))
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
            joint = self.model.joint(name)
            lower, upper = self.model.jnt_range[int(joint.id)]
            value = float(np.clip(self._wrap(value), lower, upper))
            old = self.previous[key]
            if old is not None:
                value = float(np.clip(old + self.alpha * self._wrap(value - old), lower, upper))
            self.previous[key] = value
            target[int(joint.qposadr)] = value
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
            normal = np.asarray(item.get("surface_normal_solver", [0., 0., 1.]), dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            surface = np.asarray(item.get("surface_point_solver", [0., 0., 0.]), dtype=float)
            active = float(item.get("score", 0.0)) if item.get("state", "NONE") != "NONE" else 0.
            state = str(item.get("state", "NONE"))
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


class _TerrainLimit(Limit):
    def __init__(self, model, terrain: TerrainField, config):
        self.model = model; self.terrain = TerrainField(floor_z=terrain.floor_z) if getattr(terrain, "is_mesh_scene", False) else terrain; self.margin = float(config.get("margin", .004)); self.activate = float(config.get("activate_distance", .06))
        self.shells = self._discover(); self.active = []

    def _discover(self):
        shells = []
        scene_bodies = {
            body_id
            for body_id in range(self.model.nbody)
            if (mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body_id) or "").startswith("scene_")
        }
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
                samples = np.concatenate(verts); shells.append((body_id, samples[::max(1, len(samples)//48)]))
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
        prefix = str(config.get("scene_body_prefix", "scene_"))
        self.scene_geoms = [g for g in range(model.ngeom) if model.geom_contype[g] and (mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or "").startswith(prefix)]
        self.robot_geoms = [g for g in range(model.ngeom) if model.geom_contype[g] and g not in self.scene_geoms and int(model.geom_bodyid[g]) != 0]
        self.active_pairs = []
        self.minimum_distance = np.inf
        self.maximum_penetration = 0.

    def prepare_active_set(self, configuration, dt=0.):
        del dt; configuration.update(); pairs=[]
        for robot_geom in self.robot_geoms:
            for scene_geom in self.scene_geoms:
                fromto=np.zeros(6)
                distance=float(mj.mj_geomDistance(self.model,configuration.data,robot_geom,scene_geom,self.activate,fromto))
                if distance > self.activate: continue
                vector=fromto[:3]-fromto[3:]; norm=float(np.linalg.norm(vector))
                if norm < 1e-10: continue
                # mj_geomDistance's first point belongs to geom1 (the robot),
                # so p_robot-p_scene is the free-space normal even for a
                # negative (penetrating) distance.
                pairs.append({"robot_geom":robot_geom,"scene_geom":scene_geom,"distance":distance,"normal":vector/norm,"robot_point":fromto[:3].copy()})
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
        self.minimum_distance=min((x["distance"] for x in self.active_pairs),default=np.inf); self.maximum_penetration=max(0.,-self.minimum_distance)

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
    def __init__(self, task: RetargetTask, terrain: TerrainField, environment_pool: np.ndarray, fps: float = 50.0, solver: str = "daqp"):
        self.task=task; self.config=_load_config(task.config_path); self.terrain=terrain; self.fps=float(fps); self.dt=1./self.fps; self.solver=solver
        # The resolved RobotModelSpec owns the actual MJCF path.  Do not infer
        # it from the temporary config location (which may live outside the
        # repository when a user chooses an arbitrary output directory).
        self.robot_xml = task.robot.mjcf_path.resolve()
        self.model=mj.MjModel.from_xml_path(str(self.robot_xml)); self.configuration=mink.Configuration(self.model)
        for name,bounds in self.config.get("joint_position_limits",{}).items():
            if name in [mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_JOINT,i) for i in range(self.model.njnt)]:
                jid=int(self.model.joint(name).id)
                xml_lower, xml_upper = self.model.jnt_range[jid]
                lower, upper = max(float(xml_lower), float(bounds[0])), min(float(xml_upper), float(bounds[1]))
                if not lower < upper:
                    raise ValueError(f"Invalid V5 joint range for {name}: {bounds}")
                self.model.jnt_range[jid] = (lower, upper); self.model.jnt_limited[jid]=1
        self.interaction=_InteractionTask(self.model,self.config["semantic_points"],environment_pool,self.config.get("interaction_graph",{}))
        root=self.config["global_anchor"]; self.root=_RootTask(self.model,root["robot_body"],root["cost"])
        self.torso = _TorsoCoherenceTask(self.model, self.config.get("torso_pelvis_coherence", {}))
        self.contact=_ContactTask(self.model,self.config.get("contact_tasks",{}).get("robot_points",{}),self.config.get("contact_tasks",{}))
        self.terrain_limit=_TerrainLimit(self.model,terrain,self.config.get("terrain_nonpenetration",{})); self.trust=_TrustLimit(self.model,self.config.get("solver",{}).get("trust_region",.12)); self.scene_collision=_SceneCollisionLimit(self.model,self.config.get("scene_collision",{}))
        self.config_limit=mink.ConfigurationLimit(self.model); self.velocity=mink.VelocityLimit(self.model,{mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_JOINT,i):float(self.config.get("solver",{}).get("joint_velocity_limit",10.)) for i in range(self.model.njnt) if self.model.jnt_type[i] in (mj.mjtJoint.mjJNT_HINGE,mj.mjtJoint.mjJNT_SLIDE)})
        posture=self.config.get("posture",{}); costs=np.full(self.model.nv,float(posture.get("nominal_cost",.01))); self.nominal=mink.PostureTask(self.model,costs,gain=.3,lm_damping=1.); self.nominal.set_target(self.model.qpos0)
        self.previous=None; self.frame_index=0; self.diagnostics=[]
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
                    candidate[int(self.model.joint(joint_name).qposadr)] = bend
            for joint_name, value in (("ANKLE_PITCH_L_JOINT", -0.12), ("ANKLE_PITCH_R_JOINT", -0.12)):
                if joint_name in [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]:
                    candidate[int(self.model.joint(joint_name).qposadr)] = value
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
            if any(candidate[int(self.model.joint(j).qposadr)] < -1e-6 for j in ("KNEE_PITCH_L_JOINT", "KNEE_PITCH_R_JOINT") if j in [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]):
                score += 1e3
            if score < best_score:
                best_score, best = score, candidate.copy()
        self.configuration.update(best)
        self.nominal.set_target(best)

    def solve(self, motion, solver_frames, contacts, source_frames=None) -> RetargetResult:
        outputs=[]; first_iterations=int(self.config.get("solver",{}).get("first_frame_iterations",8)); iterations=int(self.config.get("solver",{}).get("iterations",4)); fatal_failure = None
        for index,(frame,contact_frame) in enumerate(zip(solver_frames,contacts)):
            source=source_frames[index] if source_frames is not None else {k:v[0] for k,v in frame.items()}
            root=frame.get("pelvis") or frame.get("root")
            if root is None: raise ValueError("V5 solver frame lacks pelvis/root")
            if index==0:
                self._seed_initial_pose(source, root[1])
            # Config files retain the historical source labels for backwards
            # compatibility, but V5 consumes canonical semantic names first.
            semantics={}
            for name, spec in self.config["semantic_points"].items():
                source_name = str(spec["source"])
                candidates = (name, source_name)
                value = next((source[key] for key in candidates if key in source), None)
                if value is None:
                    raise ValueError(f"V5 source frame is missing semantic point {name!r} (aliases: {candidates})")
                semantics[name] = np.asarray(value, dtype=float)
            self.interaction.set_target(semantics)
            self.root.set_target(source.get("pelvis",root[0]),root[1])
            self.torso.set_source(source, root[1], self.configuration.data.qpos)
            self.contact.set_contacts(contact_frame.get("contacts",{}))
            tasks=[self.interaction,self.contact,self.root,self.torso.task,self.nominal]
            if self.previous is not None:
                temporal=mink.PostureTask(self.model,np.full(self.model.nv,.05),gain=.35,lm_damping=1.); temporal.set_target(self.previous); tasks.append(temporal)
            failures=[]; passes=first_iterations if index==0 else iterations
            for _ in range(passes):
                collision_start = time.perf_counter()
                self.terrain_limit.prepare_active_set(self.configuration,self.dt/max(passes,1)); self.scene_collision.prepare_active_set(self.configuration,self.dt/max(passes,1))
                collision_query_time = time.perf_counter() - collision_start
                limits=[self.config_limit,self.terrain_limit,self.scene_collision,self.trust]
                if index: limits.append(self.velocity)
                solve_dt = self.dt / max(passes, 1)
                base_damping = float(self.config.get("solver", {}).get("damping", .1))
                base_radius = float(self.trust.radius)
                velocity = None
                retry_errors = []
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
                self.configuration.integrate_inplace(velocity, solve_dt)
            output=self.configuration.data.qpos.copy()
            interaction_error = float(np.linalg.norm(self.interaction.compute_error(self.configuration)))
            self.diagnostics.append({"frame":index,"qp_failures":failures,"interaction_error":interaction_error,"interaction_scene_points":int(self.interaction.environment_count),"active_terrain_constraints":len(self.terrain_limit.active),"minimum_terrain_distance":float(min((x[2].signed_distance for x in self.terrain_limit.active),default=np.inf)),"scene_collision_candidate_pairs":int(len(self.scene_collision.robot_geoms)*len(self.scene_collision.scene_geoms)),"scene_collision_active_pairs":len(self.scene_collision.active_pairs),"minimum_scene_distance":float(self.scene_collision.minimum_distance),"maximum_scene_penetration":float(self.scene_collision.maximum_penetration),"collision_query_time":float(collision_query_time)})
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
        return {"body_names": body_names, "body_pos_w": positions, "body_quat_w": quaternions,
                "body_lin_vel_w": linear, "body_ang_vel_w": angular, "joint_vel": joint_velocity}


WholeBodyOmniGMRV5 = WholeBodyRetargetSolver
