"""Whole-body, ground-interaction extension of the Laplacian soft GMR path.

This module is intentionally independent of the existing Laplacian pipeline.
It adds virtual surface sites, all-body non-penetration inequalities, and
human-reference contact scheduling for prone, supine and crawling motions.
"""

from __future__ import annotations

from typing import Any

import mink
import mujoco as mj
import numpy as np
from mink.limits.limit import Constraint, Limit
from mink.tasks.task import Task

from .laplacian_soft_retarget import LaplacianSoftContactRetargeting
from .wholebody_contact_utils import human_surface_points


class GroundNonPenetrationLimit(Limit):
    """Hard floor inequalities for fixed sites and dynamic robot mesh surfaces.

    Mesh vertices are kept in body-local coordinates.  At every QP
    linearization we transform the complete candidate surface, select the
    current world-Z support points, and evaluate their point Jacobians.  This
    covers support points that move around a curved knee or foot shell as the
    link rotates.
    """

    def __init__(
        self,
        model: mj.MjModel,
        virtual_sites: dict[str, dict],
        guard_sites: dict[str, float],
        collision_geoms: dict[str, float],
        mesh_guards: dict[str, dict],
        floor_z: float,
        clearance: float,
    ):
        self.model = model
        self.floor_z = float(floor_z)
        self.clearance = float(clearance)
        self.body_ids = {name: model.body(spec["robot_body"]).id for name, spec in virtual_sites.items()}
        self.local_offsets = {name: np.asarray(spec["robot_offset"], dtype=float) for name, spec in virtual_sites.items()}
        self.guard_site_ids = {name: model.site(name).id for name in guard_sites}
        self.guard_margins = {name: float(margin) for name, margin in guard_sites.items()}
        self.collision_geom_ids = {name: model.geom(name).id for name in collision_geoms}
        self.collision_geom_margins = {name: float(margin) for name, margin in collision_geoms.items()}
        self.mesh_guards = self._build_mesh_guards(mesh_guards)
        self.last_heights: dict[str, float] = {}
        self.last_required_margins: dict[str, float] = {}

    @staticmethod
    def _quat_matrix(quat: np.ndarray) -> np.ndarray:
        matrix = np.empty(9, dtype=float)
        mj.mju_quat2Mat(matrix, quat)
        return matrix.reshape(3, 3)

    def _build_mesh_guards(self, specs: dict[str, dict]) -> dict[str, dict]:
        guards = {}
        for name, spec in specs.items():
            body_id = self.model.body(spec["body"]).id
            mesh_geom_ids = [
                int(geom_id) for geom_id in np.flatnonzero(self.model.geom_bodyid == body_id)
                if self.model.geom_type[geom_id] == mj.mjtGeom.mjGEOM_MESH
            ]
            if not mesh_geom_ids:
                raise ValueError(f"No mesh geom found on dynamic guard body {spec['body']}")
            body_vertices = []
            for geom_id in mesh_geom_ids:
                mesh_id = int(self.model.geom_dataid[geom_id])
                start = int(self.model.mesh_vertadr[mesh_id])
                count = int(self.model.mesh_vertnum[mesh_id])
                vertices = np.asarray(self.model.mesh_vert[start : start + count], dtype=float)
                geom_rotation = self._quat_matrix(self.model.geom_quat[geom_id])
                body_vertices.append(self.model.geom_pos[geom_id] + vertices @ geom_rotation.T)
            guards[name] = {
                "body_id": body_id,
                "vertices": np.concatenate(body_vertices),
                "margin": float(spec.get("margin", self.clearance)),
                "points": max(1, int(spec.get("points", 5))),
                "separation": float(spec.get("separation", 0.015)),
            }
        return guards

    def _world_point(self, configuration: mink.Configuration, name: str) -> np.ndarray:
        body_id = self.body_ids[name]
        mat = configuration.data.xmat[body_id].reshape(3, 3)
        return configuration.data.xpos[body_id] + mat @ self.local_offsets[name]

    def _site_point_and_jacobian(self, configuration: mink.Configuration, site_id: int):
        point = configuration.data.site_xpos[site_id].copy()
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mj.mj_jacSite(self.model, configuration.data, jacp, jacr, site_id)
        return point, jacp

    def _dynamic_mesh_points(self, configuration: mink.Configuration, spec: dict) -> tuple[np.ndarray, np.ndarray]:
        body_id = spec["body_id"]
        rotation = configuration.data.xmat[body_id].reshape(3, 3)
        world = configuration.data.xpos[body_id] + spec["vertices"] @ rotation.T
        order = np.argsort(world[:, 2], kind="stable")
        selected = []
        separation2 = spec["separation"] ** 2
        # Search the low surface band first.  Fall back to the next-lowest
        # vertex if a narrow/pointed shell cannot provide spatial separation.
        for index in order:
            point = world[index]
            if not selected or all(np.sum((point - other) ** 2) >= separation2 for other in selected):
                selected.append(point)
                if len(selected) == spec["points"]:
                    break
        return np.asarray(selected), world

    def _measure(self, configuration: mink.Configuration, *, include_mesh_vertices: bool) -> dict[str, dict[str, float]]:
        measurements: dict[str, dict[str, float]] = {}
        for name in self.body_ids:
            height = float(self._world_point(configuration, name)[2] - self.floor_z)
            measurements[f"virtual:{name}"] = {"height": height, "required_margin": self.clearance, "slack": height - self.clearance}
        for name, site_id in self.guard_site_ids.items():
            height = float(configuration.data.site_xpos[site_id, 2] - self.floor_z)
            margin = self.guard_margins[name]
            measurements[f"site:{name}"] = {"height": height, "required_margin": margin, "slack": height - margin}
        for name, geom_id in self.collision_geom_ids.items():
            height = float(configuration.data.geom_xpos[geom_id, 2] - self.floor_z)
            margin = self.collision_geom_margins[name]
            measurements[f"collision:{name}"] = {"height": height, "required_margin": margin, "slack": height - margin}
        for name, spec in self.mesh_guards.items():
            selected, world = self._dynamic_mesh_points(configuration, spec)
            points = world if include_mesh_vertices else selected
            height = float(np.min(points[:, 2]) - self.floor_z)
            margin = spec["margin"]
            measurements[f"mesh:{name}"] = {"height": height, "required_margin": margin, "slack": height - margin}
        return measurements

    def measure_current_slacks(self, configuration: mink.Configuration) -> dict[str, dict[str, float]]:
        """Measure final-qpos hard-point and complete visual-mesh clearances."""
        configuration.update()
        return self._measure(configuration, include_mesh_vertices=True)

    def min_signed_slack(self, configuration: mink.Configuration) -> float:
        measurements = self._measure(configuration, include_mesh_vertices=True)
        return min((item["slack"] for item in measurements.values()), default=np.inf)

    def compute_qp_inequalities(self, configuration: mink.Configuration, dt: float) -> Constraint:
        del dt
        rows, bounds = [], []
        self.last_heights = {}
        self.last_required_margins = {}
        for name, body_id in self.body_ids.items():
            point = self._world_point(configuration, name)
            jacp = np.zeros((3, self.model.nv), dtype=float)
            jacr = np.zeros((3, self.model.nv), dtype=float)
            mj.mj_jac(self.model, configuration.data, jacp, jacr, point, body_id)
            # -Jz Δq <= z(q) - (floor + clearance)
            rows.append(-jacp[2])
            bounds.append(float(point[2] - self.floor_z - self.clearance))
            self.last_heights[name] = float(point[2] - self.floor_z)
            self.last_required_margins[name] = self.clearance
        for name, site_id in self.guard_site_ids.items():
            point, jacp = self._site_point_and_jacobian(configuration, site_id)
            # Mink optimizes tangent displacement Δq, so dt is already absorbed
            # by solve_ik.  This is exactly -Jz Δq <= z-floor-margin.
            rows.append(-jacp[2])
            margin = self.guard_margins[name]
            bounds.append(float(point[2] - self.floor_z - margin))
            self.last_heights[name] = float(point[2] - self.floor_z)
            self.last_required_margins[name] = margin
        for name, geom_id in self.collision_geom_ids.items():
            point = configuration.data.geom_xpos[geom_id].copy()
            body_id = int(self.model.geom_bodyid[geom_id])
            jacp = np.zeros((3, self.model.nv), dtype=float)
            jacr = np.zeros((3, self.model.nv), dtype=float)
            mj.mj_jac(self.model, configuration.data, jacp, jacr, point, body_id)
            margin = self.collision_geom_margins[name]
            rows.append(-jacp[2])
            bounds.append(float(point[2] - self.floor_z - margin))
            self.last_heights[name] = float(point[2] - self.floor_z)
            self.last_required_margins[name] = margin
        for name, spec in self.mesh_guards.items():
            selected, _ = self._dynamic_mesh_points(configuration, spec)
            for point_id, point in enumerate(selected):
                jacp = np.zeros((3, self.model.nv), dtype=float)
                jacr = np.zeros((3, self.model.nv), dtype=float)
                mj.mj_jac(self.model, configuration.data, jacp, jacr, point, spec["body_id"])
                rows.append(-jacp[2])
                bounds.append(float(point[2] - self.floor_z - spec["margin"]))
                key = f"{name}:{point_id}"
                self.last_heights[key] = float(point[2] - self.floor_z)
                self.last_required_margins[key] = spec["margin"]
        return Constraint(G=np.asarray(rows), h=np.asarray(bounds))


class SoftBodyGroundTask(Task):
    """Soft normal attraction for surface regions selected by human contact."""

    def __init__(self, model: mj.MjModel, sites: dict[str, dict], floor_z: float, clearance: float, cost: float):
        self.model = model
        self.floor_z = float(floor_z)
        self.clearance = float(clearance)
        self.names = list(sites)
        self.body_ids = np.asarray([model.body(sites[name]["robot_body"]).id for name in self.names], dtype=int)
        self.local_offsets = np.asarray([sites[name]["robot_offset"] for name in self.names], dtype=float)
        self.activation = np.zeros(len(self.names), dtype=float)
        super().__init__(cost=np.full(len(self.names), float(cost)), gain=0.55, lm_damping=1.0)

    def set_activations(self, scores: dict[str, float], scale: float) -> None:
        self.activation = np.asarray([np.clip(scores.get(name, 0.0) * scale, 0.0, 1.0) for name in self.names])

    def _points(self, configuration: mink.Configuration) -> np.ndarray:
        matrices = configuration.data.xmat[self.body_ids].reshape(-1, 3, 3)
        return configuration.data.xpos[self.body_ids] + np.einsum("nij,nj->ni", matrices, self.local_offsets)

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        heights = self._points(configuration)[:, 2] - self.floor_z - self.clearance
        # Penetration always contributes; near-floor attraction is contact gated.
        return np.minimum(heights, 0.0) + self.activation * np.where(np.abs(heights) < 0.05, heights, 0.0)

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        jacobian = np.zeros((len(self.names), self.model.nv), dtype=float)
        points = self._points(configuration)
        heights = points[:, 2] - self.floor_z - self.clearance
        for i, (body_id, point) in enumerate(zip(self.body_ids, points)):
            jacp = np.zeros((3, self.model.nv), dtype=float)
            jacr = np.zeros((3, self.model.nv), dtype=float)
            mj.mj_jac(self.model, configuration.data, jacp, jacr, point, int(body_id))
            penetration_coeff = 1.0 if heights[i] < 0.0 else 0.0
            contact_coeff = self.activation[i] if abs(heights[i]) < 0.05 else 0.0
            jacobian[i] = (penetration_coeff + contact_coeff) * jacp[2]
        return jacobian


class SurfaceInteractionLaplacianTask(Task):
    """Surface-to-ground local graph task using virtual robot surface sites.

    Each active surface is represented by a body node linked to a local ground
    patch.  The residual tracks its adapted human body-to-patch relation; patch
    locations are held for STATIC contact and follow the human reference for
    SLIDING contact.  The task stays soft, while the limit enforces safety.
    """

    def __init__(self, model: mj.MjModel, sites: dict[str, dict], cost: float, patch_half_size: float):
        self.model = model
        self.names = list(sites)
        self.body_ids = np.asarray([model.body(sites[name]["robot_body"]).id for name in self.names], dtype=int)
        self.local_offsets = np.asarray([sites[name]["robot_offset"] for name in self.names], dtype=float)
        self.site_ids = np.asarray([model.site(sites[name]["robot_site"]).id if sites[name].get("robot_site") else -1 for name in self.names], dtype=int)
        self.target_relative = np.zeros((len(self.names), 3), dtype=float)
        self.human_patch_centers = np.zeros((len(self.names), 3), dtype=float)
        self.robot_patch_centers = np.zeros((len(self.names), 3), dtype=float)
        self.robot_patch_initialized = np.zeros(len(self.names), dtype=bool)
        self.contact_states = np.full(len(self.names), "NONE", dtype=object)
        self.weights = np.zeros(len(self.names), dtype=float)
        self.patch_half_size = float(patch_half_size)
        self.patch_offsets = np.asarray(
            [
                [-self.patch_half_size, -self.patch_half_size, 0.0],
                [-self.patch_half_size, self.patch_half_size, 0.0],
                [self.patch_half_size, -self.patch_half_size, 0.0],
                [self.patch_half_size, self.patch_half_size, 0.0],
            ],
            dtype=float,
        )
        # Four fixed nodes per local floor patch.  Their Jacobians are zero;
        # retaining all four makes the interaction topology explicit instead
        # of reducing a wide chest/forearm contact to a global ground point.
        super().__init__(cost=np.full(3 * len(self.names), float(cost)), gain=0.45, lm_damping=1.0)

    def set_contact_targets(self, contacts: dict[str, dict], ground_weight: float, floor_z: float) -> None:
        for i, name in enumerate(self.names):
            contact = contacts.get(name)
            if contact is None:
                self.weights[i] = 0.0
                self.contact_states[i] = "NONE"
                self.robot_patch_initialized[i] = False
                continue
            score = float(contact["score"])
            point = np.asarray(contact["point"], dtype=float)
            state = contact["state"]
            if state == "NONE":
                self.robot_patch_initialized[i] = False
            elif state == "STATIC" and self.contact_states[i] != "STATIC":
                self.human_patch_centers[i] = np.array([point[0], point[1], floor_z])
                self.robot_patch_initialized[i] = False
            elif state == "SLIDING":
                self.human_patch_centers[i] = np.array([point[0], point[1], floor_z])
            self.target_relative[i] = point - self.human_patch_centers[i]
            self.weights[i] = score * float(ground_weight) if state != "NONE" else 0.0
            self.contact_states[i] = state

    def _points(self, configuration: mink.Configuration) -> np.ndarray:
        matrices = configuration.data.xmat[self.body_ids].reshape(-1, 3, 3)
        points = configuration.data.xpos[self.body_ids] + np.einsum("nij,nj->ni", matrices, self.local_offsets)
        selected = self.site_ids >= 0
        points[selected] = configuration.data.site_xpos[self.site_ids[selected]]
        return points

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        body_points = self._points(configuration)
        for i, point in enumerate(body_points):
            if self.contact_states[i] == "SLIDING":
                self.robot_patch_centers[i] = np.array([point[0], point[1], self.human_patch_centers[i, 2]])
                self.robot_patch_initialized[i] = True
            elif self.contact_states[i] == "STATIC" and not self.robot_patch_initialized[i]:
                self.robot_patch_centers[i] = np.array([point[0], point[1], self.human_patch_centers[i, 2]])
                self.robot_patch_initialized[i] = True
        # Explicit body-to-average-ground-neighbour Laplacian edge.  Ground
        # nodes are fixed during each QP and therefore have zero Jacobian.
        # Robot and human patches are deliberately distinct; using one shared
        # center would algebraically reduce this task to point tracking.
        robot_edges = body_points - self.robot_patch_centers
        residual = robot_edges - self.target_relative
        return (residual * self.weights[:, None]).reshape(-1)

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        jacobian = np.zeros((3 * len(self.names), self.model.nv), dtype=float)
        for i, (body_id, point) in enumerate(zip(self.body_ids, self._points(configuration))):
            jacp = np.zeros((3, self.model.nv), dtype=float)
            jacr = np.zeros((3, self.model.nv), dtype=float)
            if self.site_ids[i] >= 0:
                mj.mj_jacSite(self.model, configuration.data, jacp, jacr, int(self.site_ids[i]))
            else:
                mj.mj_jac(self.model, configuration.data, jacp, jacr, point, int(body_id))
            row = 3 * i
            jacobian[row : row + 3] = self.weights[i] * jacp
        return jacobian


class FootTemporalContactTask(Task):
    """ProtoMotions-style temporal sole contact with heel/toe channels.

    Each channel penalizes the true 50 Hz site velocity.  The previous point
    is the previous exported frame, not the current QP substep, so repeated
    near-ground linearization does not change the temporal target.
    """

    CHANNELS = ("left_heel", "left_toe", "right_heel", "right_toe")

    def __init__(self, model: mj.MjModel, site_groups: dict[str, list[str]], dt: float, velocity_cost: float, flat_cost: float):
        self.model = model
        self.dt = max(float(dt), 1e-6)
        self.site_ids = {
            name: np.asarray([model.site(site).id for site in site_groups[name]], dtype=int)
            for name in self.CHANNELS
        }
        self.foot_body_ids = {
            side: model.body(f"ANKLE_ROLL_{side[0].upper()}_LINK").id
            for side in ("left", "right")
        }
        self.velocity_cost = float(velocity_cost)
        self.flat_cost = float(flat_cost)
        self.scores = {name: 0.0 for name in self.CHANNELS}
        self.walk_alpha = 0.0
        self.previous_points: dict[str, np.ndarray] = {}
        self.frame_previous_points: dict[str, np.ndarray] = {}
        self.debug: dict[str, float] = {}
        # 4 channels * xyz velocity + 2 feet * xy normal residual.
        super().__init__(cost=np.ones(16), gain=0.45, lm_damping=1.0)

    def begin_frame(self, configuration: mink.Configuration, scores: dict[str, float], walk_alpha: float) -> None:
        self.walk_alpha = float(np.clip(walk_alpha, 0.0, 1.0))
        self.scores = {name: float(np.clip(scores.get(name, 0.0), 0.0, 1.0)) for name in self.CHANNELS}
        current = self._channel_points(configuration)
        self.frame_previous_points = {
            name: self.previous_points.get(name, point).copy() for name, point in current.items()
        }

    def end_frame(self, configuration: mink.Configuration) -> None:
        self.previous_points = {name: point.copy() for name, point in self._channel_points(configuration).items()}

    def _channel_points(self, configuration: mink.Configuration) -> dict[str, np.ndarray]:
        return {name: np.mean(configuration.data.site_xpos[ids], axis=0) for name, ids in self.site_ids.items()}

    def _point_jacobian(self, configuration: mink.Configuration, ids: np.ndarray) -> np.ndarray:
        total = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        for site_id in ids:
            jacp = np.zeros((3, self.model.nv), dtype=float)
            mj.mj_jacSite(self.model, configuration.data, jacp, jacr, int(site_id))
            total += jacp
        return total / len(ids)

    def _flat_activation(self, side: str) -> float:
        return self.walk_alpha * min(self.scores[f"{side}_heel"], self.scores[f"{side}_toe"])

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        points = self._channel_points(configuration)
        residual = []
        for name in self.CHANNELS:
            score = self.walk_alpha * self.scores[name]
            residual.extend(score * (points[name] - self.frame_previous_points[name]) / self.dt)
        for side in ("left", "right"):
            body_id = self.foot_body_ids[side]
            rotation = configuration.data.xmat[body_id].reshape(3, 3)
            residual.extend(self._flat_activation(side) * rotation[:2, 2])
        return np.asarray(residual, dtype=float)

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        jacobian = np.zeros((16, self.model.nv), dtype=float)
        row = 0
        for name in self.CHANNELS:
            score = self.walk_alpha * self.scores[name]
            jacobian[row : row + 3] = score * self._point_jacobian(configuration, self.site_ids[name]) / self.dt
            row += 3
        for side in ("left", "right"):
            body_id = self.foot_body_ids[side]
            jacp = np.zeros((3, self.model.nv), dtype=float)
            jacr = np.zeros((3, self.model.nv), dtype=float)
            mj.mj_jacBody(self.model, configuration.data, jacp, jacr, body_id)
            rotation = configuration.data.xmat[body_id].reshape(3, 3)
            # d(world sole-z)/dq = -[Rz]_x * angular Jacobian.
            axis = rotation[:, 2]
            skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
            jacobian[row : row + 2] = self._flat_activation(side) * (-skew @ jacr)[:2]
            row += 2
        return jacobian

    def update_debug(self, configuration: mink.Configuration) -> dict[str, float]:
        points = self._channel_points(configuration)
        debug = {"walk_alpha": self.walk_alpha}
        for name in self.CHANNELS:
            velocity = (points[name] - self.frame_previous_points.get(name, points[name])) / self.dt
            debug[f"{name}_score"] = self.scores[name]
            debug[f"{name}_height"] = float(points[name][2])
            debug[f"{name}_xy_speed"] = float(np.linalg.norm(velocity[:2]))
        for side in ("left", "right"):
            debug[f"{side}_flat_foot"] = self._flat_activation(side)
        self.debug = debug
        return debug


class WholeBodyOmniGMR(LaplacianSoftContactRetargeting):
    """Whole-body prone/crawl/supine extension, without altering the base path."""

    def __init__(self, *args, **kwargs):
        # Use the dedicated XML that contains orange contact and red guard sites.
        self._wholebody_verbose = bool(kwargs.get("verbose", True))
        if kwargs.get("tgt_robot") == "ne01":
            kwargs["tgt_robot"] = "ne01_wholebody_omni_gmr"
        super().__init__(*args, **kwargs)
        cfg = self.graph_config.get("whole_body_ground", {})
        interaction = self.graph_config.get("ground_interaction_graph", {})
        self.surface_regions = interaction.get("regions", {})
        if not cfg.get("enabled", True) or not self.surface_regions:
            raise ValueError("whole_body_ground and ground_interaction_graph.regions are required")
        self.mode_weights = self.graph_config.get("motion_mode_weights", {})
        self.mode_blend_frames = max(1, int(self.graph_config.get("motion_mode_blend_frames", 8)))
        self.current_mode = "normal"
        self.target_mode = "normal"
        self.mode_alpha = 1.0
        self._primary_base_costs = {task: task.cost.copy() for task in self.primary_tasks}
        self._orientation_base_costs = {
            task: task.cost[3:].copy() for task in self.primary_tasks
        }
        clearance = float(cfg.get("clearance", 0.002))
        floor_z = float(cfg.get("floor_z", self.ground[2]))
        nonpenetration = self.graph_config.get("ground_nonpenetration", {})
        self.near_ground_threshold = float(nonpenetration.get("near_ground_threshold", 0.025))
        self.near_ground_inner_iterations = max(1, int(nonpenetration.get("near_ground_inner_iterations", 3)))
        default_guard_margin = float(nonpenetration.get("margin", clearance))
        guard_sites_config = nonpenetration.get("always_active_sites", {})
        if isinstance(guard_sites_config, list):
            guard_sites_config = {name: default_guard_margin for name in guard_sites_config}
        self.whole_body_limit = GroundNonPenetrationLimit(
            self.model,
            self.surface_regions,
            {name: float(margin) for name, margin in guard_sites_config.items()},
            {name: float(margin) for name, margin in nonpenetration.get("collision_geoms", {}).items()},
            nonpenetration.get("dynamic_mesh_guards", {}),
            floor_z,
            clearance,
        )
        self.residual_lift_max_violation = float(nonpenetration.get("residual_lift_max_violation", 0.003))
        self.whole_body_ground_task = SoftBodyGroundTask(
            self.model, self.surface_regions, floor_z, clearance,
            float(interaction.get("contact_normal_weight", 0.3)),
        )
        self.interaction_task = SurfaceInteractionLaplacianTask(
            self.model, self.surface_regions,
            float(interaction.get("ground_edge_weight", 1.0)),
            float(interaction.get("patch_half_size", 0.04)),
        )
        self.whole_body_max_iterations = max(1, int(cfg.get("max_ik_iterations", 8)))
        self.last_contact_frame: dict[str, Any] = {"contacts": {}}
        self._orientation_valid = True
        temporal = self.graph_config.get("foot_temporal_contact", {})
        self.foot_temporal_enabled = bool(temporal.get("enabled", True))
        self.walk_blend_frames = max(1, int(temporal.get("walk_blend_frames", 7)))
        self.walk_alpha = 1.0
        self.foot_temporal_debug = bool(temporal.get("debug", False))
        self.foot_temporal_debug_interval = max(1, int(temporal.get("debug_interval", 25)))
        default_sites = {
            "left_heel": ["ground_guard_left_heel_inner", "ground_guard_left_heel_outer"],
            "left_toe": ["ground_guard_left_forefoot_inner", "ground_guard_left_forefoot_outer"],
            "right_heel": ["ground_guard_right_heel_inner", "ground_guard_right_heel_outer"],
            "right_toe": ["ground_guard_right_forefoot_inner", "ground_guard_right_forefoot_outer"],
        }
        self.foot_temporal_task = FootTemporalContactTask(
            self.model,
            temporal.get("site_groups", default_sites),
            self.motion_dt,
            float(temporal.get("velocity_cost", 0.08)),
            float(temporal.get("flat_foot_cost", 0.12)),
        )
        self.foot_temporal_task.cost[:12] *= self.foot_temporal_task.velocity_cost
        self.foot_temporal_task.cost[12:] *= self.foot_temporal_task.flat_cost

    def set_orientation_valid(self, valid: bool) -> None:
        """Declare whether input rotations are measured or position-derived.

        Position-only prepared NPZ files do not contain joint rotations.  Their
        unit quaternions are placeholders and must never create orientation
        residuals in FrameTask.  Pelvis/spine orientation, when supplied by
        the wholebody loader, is the only position-derived orientation retained.
        """
        self._orientation_valid = bool(valid)
        if valid:
            return
        for task in self.primary_tasks:
            frame = self.task_frame_names.get(task, "")
            original_orientation = self._orientation_base_costs.get(task, np.zeros(0))
            task.cost[3:] = 0.0
            # The loader supplies stable pelvis/spine directions, but position
            # tracking remains the authority for position-only data.
            if frame in {"base_link", "TORSO_LINK"}:
                task.cost[3:] = np.minimum(original_orientation, 0.25)

    def build_contact_schedule(self, frames):
        from .wholebody_contact_utils import build_foot_temporal_contact_schedule, build_whole_body_contact_schedule
        # Build features in the same scaled/offset/ground-aligned coordinate
        # frame used later by update_targets(), while keeping all state purely
        # human-reference based.
        adapted_frames = []
        for frame in frames:
            adapted, _ = self._prepare_target_data(frame)
            self._rescale_surface_points(frame, adapted)
            adapted_frames.append(adapted)
        schedule = build_whole_body_contact_schedule(
            adapted_frames, self.surface_regions, fps=self.motion_fps,
            floor_z=float(self.ground[2]),
            contact_height=float(self.graph_config.get("contact_enter_height", 0.03)),
            release_height=float(self.graph_config.get("contact_exit_height", 0.05)),
            vertical_speed_limit=float(self.graph_config.get("soft_contact_vertical_speed", 0.35)),
            static_speed_limit=float(self.graph_config.get("contact_static_speed", 0.08)),
            smoothing=float(self.graph_config.get("whole_body_contact_smoothing", 0.35)),
        )
        temporal = self.graph_config.get("foot_temporal_contact", {})
        temporal_schedule = build_foot_temporal_contact_schedule(
            adapted_frames,
            fps=self.motion_fps,
            floor_z=float(self.ground[2]),
            enter_height=float(temporal.get("enter_height", 0.035)),
            exit_height=float(temporal.get("exit_height", 0.055)),
            horizontal_speed_limit=float(temporal.get("horizontal_speed_limit", 0.18)),
            vertical_speed_limit=float(temporal.get("vertical_speed_limit", 0.20)),
            smoothing_frames=int(temporal.get("contact_smoothing_frames", 7)),
        )
        for wholebody_frame, temporal_frame in zip(schedule, temporal_schedule):
            wholebody_frame.update(temporal_frame)
        return schedule

    def _rescale_surface_points(self, source_frame: dict, adapted_frame: dict) -> None:
        """Scale heel/toe offsets with the foot segment in this pipeline only."""
        for side in ("left", "right"):
            foot = f"{side}_foot"
            if foot not in source_frame or foot not in adapted_frame:
                continue
            source_foot = np.asarray(source_frame[foot][0], dtype=float)
            target_foot = np.asarray(adapted_frame[foot][0], dtype=float)
            scale = float(self.human_scale_table.get(foot, 1.0))
            for name in (f"{side}_heel", f"{side}_big_toe", f"{side}_small_toe"):
                if name in source_frame:
                    point = source_foot + scale * (np.asarray(source_frame[name][0], dtype=float) - source_foot)
                    adapted_frame[name] = [target_foot + (point - source_foot), adapted_frame[foot][1]]

    def _infer_mode(self, contacts: dict[str, dict]) -> str:
        score = lambda name: float(contacts.get(name, {}).get("score", 0.0))
        if score("chest_front_left") + score("chest_front_right") > 0.45:
            return "prone"
        if score("chest_back_left") + score("chest_back_right") > 0.45:
            return "supine"
        crawl_regions = ("left_forearm", "right_forearm", "left_knee", "right_knee", "left_shin", "right_shin")
        if sum(score(name) > 0.18 for name in crawl_regions) >= 3:
            return "crawl"
        return "normal"

    def _update_mode_weights(self, contacts: dict[str, dict]) -> None:
        target = self._infer_mode(contacts)
        if target != self.target_mode:
            self.target_mode, self.mode_alpha = target, 0.0
        self.mode_alpha = min(1.0, self.mode_alpha + 1.0 / self.mode_blend_frames)
        if self.mode_alpha >= 1.0:
            self.current_mode = self.target_mode
        normal = self.mode_weights.get("normal", {})
        active = self.mode_weights.get(self.target_mode, normal)
        def value(key, default):
            a, b = float(normal.get(key, default)), float(active.get(key, default))
            return (1.0 - self.mode_alpha) * a + self.mode_alpha * b
        primary_scale = value("primary", 1.0)
        foot_scale = value("foot_orientation", 1.0)
        for task, original in self._primary_base_costs.items():
            task.cost[:] = original * primary_scale
            frame = self.task_frame_names.get(task, "")
            if "toe_link" in frame or "ANKLE" in frame:
                task.cost[3:] *= foot_scale
        self.graph_task.cost[:] = self.graph_cost * value("laplacian", 1.0)
        self.whole_body_ground_task.set_activations(
            {name: value_["score"] for name, value_ in contacts.items()}, value("contact_normal", 0.3)
        )
        self.interaction_ground_weight = value("ground_graph", 1.0)
        target_walk = 1.0 if self.target_mode == "normal" else 0.0
        step = 1.0 / self.walk_blend_frames
        self.walk_alpha += np.clip(target_walk - self.walk_alpha, -step, step)

    def update_targets(self, human_data, offset_to_ground=False, contact_frame=None):
        super().update_targets(human_data, offset_to_ground)
        self._rescale_surface_points(human_data, self.scaled_human_data)
        contacts = (contact_frame or {}).get("contacts", {})
        # Recreate surface points after the same scale/offset/ground adaptation
        # used by the primary robot targets.  Scores/states remain precomputed.
        surface_points = human_surface_points(self.scaled_human_data, self.surface_regions)
        adapted_contacts = {}
        for name, contact in contacts.items():
            if name in surface_points:
                adapted_contacts[name] = {**contact, "point": surface_points[name]}
        self.last_contact_frame = {"contacts": adapted_contacts}
        self._update_mode_weights(adapted_contacts)
        self.interaction_task.set_contact_targets(
            adapted_contacts, self.interaction_ground_weight, float(self.ground[2])
        )
        temporal_contacts = (contact_frame or {}).get("foot_temporal", {})
        self.foot_temporal_task.begin_frame(
            self.configuration,
            {name: item.get("score", 0.0) for name, item in temporal_contacts.items()},
            self.walk_alpha if self.foot_temporal_enabled else 0.0,
        )

    def retarget(self, human_data, offset_to_ground=False, contact_frame=None):
        self.update_targets(human_data, offset_to_ground, contact_frame)
        min_slack = self.whole_body_limit.min_signed_slack(self.configuration)
        passes = self.near_ground_inner_iterations if min_slack < self.near_ground_threshold else 1
        if self.retarget_call_count == 0:
            passes = max(passes, self.whole_body_max_iterations)
        tasks = list(self.primary_tasks) + [self.graph_task]
        # Upright motion uses the ordinary Laplacian task set.  The wholebody
        # interaction graph and soft body-ground attraction are reserved for
        # prone/crawl/supine contacts; the hard limit remains active always.
        if self.target_mode != "normal":
            tasks.extend([self.interaction_task, self.whole_body_ground_task])
            tasks.extend(self.soft_ground_tasks.values())
            tasks.extend(self.soft_tangential_tasks.values())
        if self.foot_temporal_enabled:
            tasks.append(self.foot_temporal_task)
        if self._q_prev is not None:
            posture = mink.PostureTask(self.model, cost=0.02, lm_damping=1.0)
            posture.set_target(self._q_prev)
            tasks.append(posture)
        limits = list(self.ik_limits) + [self.whole_body_limit]
        sub_dt = self.motion_dt / passes
        for _ in range(passes):
            # The Limit receives the updated configuration each call, therefore
            # FK, site_xpos and mj_jacSite are re-linearized at every substep.
            velocity = mink.solve_ik(self.configuration, tasks, sub_dt, self.solver, self.damping, limits=limits)
            self.configuration.integrate_inplace(velocity, sub_dt)
        self._apply_whole_body_residual_lift()
        debug = self.foot_temporal_task.update_debug(self.configuration)
        self.foot_temporal_task.end_frame(self.configuration)
        if self.foot_temporal_debug and self.retarget_call_count % self.foot_temporal_debug_interval == 0:
            print(
                "FootTemporal "
                f"walk={debug['walk_alpha']:.2f} "
                f"L[h={debug['left_heel_score']:.2f}/{debug['left_heel_height']:.3f}/{debug['left_heel_xy_speed']:.3f}, "
                f"t={debug['left_toe_score']:.2f}/{debug['left_toe_height']:.3f}/{debug['left_toe_xy_speed']:.3f}, flat={debug['left_flat_foot']:.2f}] "
                f"R[h={debug['right_heel_score']:.2f}/{debug['right_heel_height']:.3f}/{debug['right_heel_xy_speed']:.3f}, "
                f"t={debug['right_toe_score']:.2f}/{debug['right_toe_height']:.3f}/{debug['right_toe_xy_speed']:.3f}, flat={debug['right_flat_foot']:.2f}]"
            )
        self.retarget_call_count += 1
        self._q_prev2 = self._q_prev
        self._q_prev = self.configuration.data.qpos.copy()
        return self.configuration.data.qpos.copy()

    def _apply_whole_body_residual_lift(self) -> None:
        # Numerical final guard only: never change XY and never compensate a
        # material penetration by a large root teleport.
        measurements = self.whole_body_limit.measure_current_slacks(self.configuration)
        worst_name, worst = min(measurements.items(), key=lambda item: item[1]["slack"])
        violation = max(0.0, -float(worst["slack"]))
        # Root lift is only a numerical cleanup.  Material local penetration
        # must remain visible to validation rather than being hidden globally.
        correction = 0.0
        if violation <= self.residual_lift_max_violation:
            correction = min(violation, self.soft_root_lift_rate)
        if correction > 1e-9:
            qpos = self.configuration.data.qpos.copy()
            qpos[2] += correction
            self.configuration.update(qpos)
            measurements = self.whole_body_limit.measure_current_slacks(self.configuration)
        final_name, final = min(measurements.items(), key=lambda item: item[1]["slack"])
        if self._wholebody_verbose and final["slack"] < -1e-5:
            print(
                f"WholeBody final mesh/hard-point violation: {final_name} "
                f"height={final['height']:.6f} margin={final['required_margin']:.6f} "
                f"slack={final['slack']:.6f}"
            )
