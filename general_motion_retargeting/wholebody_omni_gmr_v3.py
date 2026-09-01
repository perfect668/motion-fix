"""Omni-first whole-body retargeting for NE01 and static terrain.

This module is intentionally independent from the GMR Table1/Table2 pipeline.
Absolute human landmarks are used only for a weak global root anchor.  Limb
motion is driven by a per-frame human--terrain interaction Laplacian.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mink
import mujoco as mj
import numpy as np
from mink.limits.limit import Constraint, Limit
from mink.tasks.task import Task
from scipy.spatial import Delaunay, QhullError
from scipy.spatial.transform import Rotation

from .terrain_geometry import TerrainField
from .terrain_tasks import RobotPointGroup, TerrainFootOrientationTask, TerrainPointContactTask


def _farthest_sample(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) <= count:
        return points.copy()
    center = points.mean(axis=0)
    selected = [int(np.argmax(np.sum((points - center) ** 2, axis=1)))]
    nearest = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for _ in range(1, count):
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(nearest, np.sum((points - points[index]) ** 2, axis=1))
    return points[np.asarray(selected, dtype=int)]


def sample_terrain_surface_pool(
    terrain: TerrainField,
    reference_points: np.ndarray,
    count: int | None = None,
    *,
    points_per_square_meter: float = 60.0,
    minimum: int = 96,
    maximum: int = 320,
) -> np.ndarray:
    """Build a deterministic, geometry-dependent environment point pool."""
    reference = np.asarray(reference_points, dtype=float).reshape((-1, 3))
    if count is None:
        surface_area = sum(
            8.0 * (
                box.half_extents[0] * box.half_extents[1]
                + box.half_extents[0] * box.half_extents[2]
                + box.half_extents[1] * box.half_extents[2]
            )
            for box in terrain.boxes
        )
        if terrain.floor_z is not None:
            extent = np.ptp(reference[:, :2], axis=0) + 0.7
            surface_area += float(np.prod(extent))
        count = int(np.clip(
            np.ceil(surface_area * float(points_per_square_meter)),
            int(minimum),
            int(maximum),
        ))
    candidates: list[np.ndarray] = []
    grid = np.linspace(-1.0, 1.0, 7)
    for box in terrain.boxes:
        for axis in range(3):
            other = [value for value in range(3) if value != axis]
            for sign in (-1.0, 1.0):
                local = np.zeros((len(grid) ** 2, 3), dtype=float)
                local[:, axis] = sign * box.half_extents[axis]
                mesh_u, mesh_v = np.meshgrid(grid, grid, indexing="ij")
                local[:, other[0]] = mesh_u.reshape(-1) * box.half_extents[other[0]]
                local[:, other[1]] = mesh_v.reshape(-1) * box.half_extents[other[1]]
                candidates.append(box.center + local @ box.rotation.T)

    if terrain.floor_z is not None:
        lower = reference[:, :2].min(axis=0) - 0.35
        upper = reference[:, :2].max(axis=0) + 0.35
        side = max(8, int(np.ceil(np.sqrt(max(count, 16)))))
        xs, ys = np.meshgrid(
            np.linspace(lower[0], upper[0], side),
            np.linspace(lower[1], upper[1], side),
            indexing="ij",
        )
        candidates.append(
            np.column_stack((xs.reshape(-1), ys.reshape(-1), np.full(xs.size, terrain.floor_z)))
        )
    if not candidates:
        raise ValueError("Cannot sample an empty terrain")
    return _farthest_sample(np.concatenate(candidates), max(8, int(count)))


class RobotSemanticPoint:
    """A named robot point represented by a body-local offset or site group."""

    def __init__(self, model: mj.MjModel, specification: dict) -> None:
        self.group = RobotPointGroup(model, specification)

    def point(self, configuration: mink.Configuration) -> np.ndarray:
        return self.group.point(configuration)

    def jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        return self.group.jacobian(configuration)


class InteractionLaplacianTask(Task):
    """Per-frame interaction Laplacian over human/robot and terrain vertices."""

    def __init__(
        self,
        model: mj.MjModel,
        semantic_points: dict[str, dict],
        environment_pool: np.ndarray,
        environment_points: int,
        semantic_cost: float,
        environment_cost: float,
        gain: float,
    ) -> None:
        self.model = model
        self.names = list(semantic_points)
        self.robot_points = {
            name: RobotSemanticPoint(model, specification)
            for name, specification in semantic_points.items()
        }
        self.environment_pool = np.asarray(environment_pool, dtype=float).reshape((-1, 3))
        self.environment_count = min(max(4, int(environment_points)), len(self.environment_pool))
        self.vertex_count = len(self.names) + self.environment_count
        self.laplacian = np.zeros((self.vertex_count, self.vertex_count), dtype=float)
        self.target = np.zeros((self.vertex_count, 3), dtype=float)
        self.environment = self.environment_pool[: self.environment_count].copy()
        costs = np.repeat(
            np.r_[
                np.full(len(self.names), float(semantic_cost)),
                np.full(self.environment_count, float(environment_cost)),
            ],
            3,
        )
        super().__init__(cost=costs, gain=float(gain), lm_damping=1.0)

    @staticmethod
    def _laplacian(vertices: np.ndarray) -> np.ndarray:
        count = len(vertices)
        edges: set[tuple[int, int]] = set()
        try:
            simplices = Delaunay(vertices, qhull_options="QJ").simplices
            for simplex in simplices:
                for row in range(len(simplex)):
                    for column in range(row + 1, len(simplex)):
                        a, b = sorted((int(simplex[row]), int(simplex[column])))
                        if a < count and b < count:
                            edges.add((a, b))
        except QhullError:
            distances = np.linalg.norm(vertices[:, None] - vertices[None, :], axis=-1)
            for index in range(count):
                for neighbor in np.argsort(distances[index])[1:5]:
                    edges.add(tuple(sorted((index, int(neighbor)))))
        adjacency = [[] for _ in range(count)]
        for first, second in sorted(edges):
            adjacency[first].append(second)
            adjacency[second].append(first)
        matrix = np.zeros((count, count), dtype=float)
        for index, neighbors in enumerate(adjacency):
            if not neighbors:
                continue
            matrix[index, index] = 1.0
            matrix[index, neighbors] = -1.0 / len(neighbors)
        return matrix

    def set_source(self, source_points: dict[str, np.ndarray]) -> None:
        missing = [name for name in self.names if name not in source_points]
        if missing:
            raise ValueError(f"Interaction source is missing semantic points: {missing}")
        human = np.asarray([source_points[name] for name in self.names], dtype=float)
        distances = np.linalg.norm(
            self.environment_pool[:, None, :] - human[None, :, :], axis=-1
        ).min(axis=1)
        selected = np.argsort(distances, kind="stable")[: self.environment_count]
        self.environment = self.environment_pool[selected].copy()
        vertices = np.vstack((human, self.environment))
        self.laplacian = self._laplacian(vertices)
        self.target = self.laplacian @ vertices

    def _current(self, configuration: mink.Configuration):
        points = np.asarray(
            [self.robot_points[name].point(configuration) for name in self.names]
        )
        jacobians = [
            self.robot_points[name].jacobian(configuration) for name in self.names
        ]
        return np.vstack((points, self.environment)), jacobians

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        vertices, _ = self._current(configuration)
        return (self.laplacian @ vertices - self.target).reshape(-1)

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        _, semantic_jacobians = self._current(configuration)
        vertex_jacobian = np.zeros((3 * self.vertex_count, self.model.nv), dtype=float)
        for index, jacobian in enumerate(semantic_jacobians):
            vertex_jacobian[3 * index : 3 * index + 3] = jacobian
        return np.kron(self.laplacian, np.eye(3)) @ vertex_jacobian


class RootGaugeTask(Task):
    """Weak global translation/yaw gauge that removes Laplacian null modes."""

    def __init__(self, model: mj.MjModel, body_name: str, costs: list[float]) -> None:
        self.model = model
        self.body_id = model.body(body_name).id
        self.target_position = np.zeros(3)
        self.target_yaw = 0.0
        super().__init__(cost=np.asarray(costs, dtype=float), gain=0.45, lm_damping=1.0)

    def set_target(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> None:
        self.target_position = np.asarray(position, dtype=float).reshape(3)
        self.target_yaw = float(
            Rotation.from_quat(quaternion_wxyz, scalar_first=True).as_euler("zyx")[0]
        )

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        position = configuration.data.xpos[self.body_id]
        rotation = configuration.data.xmat[self.body_id].reshape(3, 3)
        yaw = float(Rotation.from_matrix(rotation).as_euler("zyx")[0])
        yaw_error = (yaw - self.target_yaw + np.pi) % (2.0 * np.pi) - np.pi
        return np.r_[position - self.target_position, yaw_error]

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mj.mj_jacBody(self.model, configuration.data, jacp, jacr, self.body_id)
        return np.vstack((jacp, np.array([0.0, 0.0, 1.0]) @ jacr))


class TorsoPelvisCoherenceTask:
    """Track feasible chest motion relative to the pelvis using waist DoFs only."""

    def __init__(self, model: mj.MjModel, config: dict) -> None:
        costs = np.zeros(model.nv, dtype=float)
        self.joints = {}
        for key in ("waist_yaw", "torso_roll"):
            specification = config[key]
            joint = model.joint(specification["joint"])
            self.joints[key] = {
                "qpos": int(model.jnt_qposadr[joint.id]),
                "dof": int(model.jnt_dofadr[joint.id]),
                "minimum": float(specification["minimum"]),
                "maximum": float(specification["maximum"]),
            }
            costs[self.joints[key]["dof"]] = float(specification["cost"])
        self.task = mink.PostureTask(
            model,
            costs,
            gain=float(config.get("gain", 0.45)),
            lm_damping=1.0,
        )
        blend_frames = max(1, int(config.get("blend_frames", 7)))
        self.alpha = 2.0 / (blend_frames + 1.0)
        self.filtered = {"waist_yaw": None, "torso_roll": None}
        self.targets = {"waist_yaw": 0.0, "torso_roll": 0.0}
        self.task.set_target(model.qpos0)

    @staticmethod
    def _wrap(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def set_target(
        self,
        configuration: mink.Configuration,
        pelvis_quaternion: np.ndarray,
        chest_quaternion: np.ndarray,
    ) -> None:
        pelvis = Rotation.from_quat(pelvis_quaternion, scalar_first=True)
        chest = Rotation.from_quat(chest_quaternion, scalar_first=True)
        yaw, _, roll = (pelvis.inv() * chest).as_euler("zyx")
        raw = {"waist_yaw": float(yaw), "torso_roll": float(roll)}
        target_q = configuration.q.copy()
        for key, angle in raw.items():
            item = self.joints[key]
            bounded = float(np.clip(self._wrap(angle), item["minimum"], item["maximum"]))
            previous = self.filtered[key]
            if previous is None:
                filtered = bounded
            else:
                delta = self._wrap(bounded - previous)
                filtered = previous + self.alpha * delta
                filtered = float(np.clip(filtered, item["minimum"], item["maximum"]))
            self.filtered[key] = filtered
            self.targets[key] = filtered
            target_q[item["qpos"]] = filtered
        self.task.set_target(target_q)


class TangentTrustRegionLimit(Limit):
    """Infinity-norm trust region for one SQP linearization."""

    def __init__(self, model: mj.MjModel, radius: float) -> None:
        self.model = model
        self.radius = float(radius)

    def compute_qp_inequalities(self, configuration, dt: float) -> Constraint:
        del configuration, dt
        identity = np.eye(self.model.nv)
        return Constraint(
            G=np.vstack((identity, -identity)),
            h=np.full(2 * self.model.nv, self.radius),
        )


class AutomaticMeshTerrainLimit(Limit):
    """Terrain non-penetration from automatically sampled visual shells."""

    def __init__(self, model: mj.MjModel, terrain: TerrainField, config: dict) -> None:
        self.model = model
        self.terrain = terrain
        self.margin = float(config["margin"])
        self.activate_distance = float(config["activate_distance"])
        self.deactivate_distance = float(config["deactivate_distance"])
        self.prediction_horizon = float(config["prediction_horizon"])
        self.hold_steps = max(1, int(config["deactivate_hold_steps"]))
        self.points_per_shell = max(1, int(config["active_points_per_shell"]))
        self.separation = float(config["point_separation"])
        self.proxy_density = float(config["mesh_proxy_density"])
        self.proxy_minimum = max(8, int(config["mesh_proxy_minimum"]))
        self.proxy_maximum = max(self.proxy_minimum, int(config["mesh_proxy_maximum"]))
        self.shells = self._discover_shells()
        self.previous = {name: np.nan for name in self.shells}
        self.active = {name: False for name in self.shells}
        self.release_count = {name: 0 for name in self.shells}
        self.selected: dict[str, list[tuple[np.ndarray, Any]]] = {}
        self.measurements: dict[str, dict] = {}

    @staticmethod
    def _quat_matrix(quaternion: np.ndarray) -> np.ndarray:
        matrix = np.empty(9, dtype=float)
        mj.mju_quat2Mat(matrix, quaternion)
        return matrix.reshape(3, 3)

    def _discover_shells(self) -> dict[str, dict]:
        shells = {}
        for body_id in range(1, self.model.nbody):
            vertices = []
            for geom_id in np.flatnonzero(self.model.geom_bodyid == body_id):
                if self.model.geom_type[geom_id] != mj.mjtGeom.mjGEOM_MESH:
                    continue
                mesh_id = int(self.model.geom_dataid[geom_id])
                start = int(self.model.mesh_vertadr[mesh_id])
                count = int(self.model.mesh_vertnum[mesh_id])
                raw = np.asarray(self.model.mesh_vert[start : start + count], dtype=float)
                rotation = self._quat_matrix(self.model.geom_quat[geom_id])
                vertices.append(self.model.geom_pos[geom_id] + raw @ rotation.T)
            if not vertices:
                continue
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body_id)
            combined = np.concatenate(vertices)
            extent = np.ptp(combined, axis=0)
            box_area = 2.0 * (
                extent[0] * extent[1]
                + extent[0] * extent[2]
                + extent[1] * extent[2]
            )
            proxy_count = int(np.clip(
                np.ceil(box_area * self.proxy_density),
                self.proxy_minimum,
                self.proxy_maximum,
            ))
            shells[name] = {
                "body_id": body_id,
                "proxies": _farthest_sample(combined, proxy_count),
            }
        if not shells:
            raise ValueError("Robot model contains no mesh shells for terrain collision")
        return shells

    def _world_points(self, configuration, shell: dict) -> np.ndarray:
        body_id = shell["body_id"]
        rotation = configuration.data.xmat[body_id].reshape(3, 3)
        return configuration.data.xpos[body_id] + shell["proxies"] @ rotation.T

    def _spatial_selection(self, points, hits):
        order = sorted(range(len(points)), key=lambda index: (hits[index].signed_distance, index))
        selected = []
        separation2 = self.separation**2
        for index in order:
            item = (points[index], hits[index])
            if not selected or all(np.sum((item[0] - old[0]) ** 2) >= separation2 for old in selected):
                selected.append(item)
                if len(selected) >= self.points_per_shell:
                    break
        return selected

    def prepare_active_set(self, configuration, dt: float) -> None:
        configuration.update()
        self.selected = {}
        self.measurements = {}
        for name, shell in self.shells.items():
            points = self._world_points(configuration, shell)
            hits = self.terrain.nearest_surface_batch(points)
            minimum = min(hit.signed_distance for hit in hits)
            old = self.previous[name]
            speed = (minimum - old) / dt if dt > 1e-8 and np.isfinite(old) else 0.0
            predicted = minimum + self.prediction_horizon * dt * min(speed, 0.0)
            if minimum <= self.activate_distance or predicted <= self.activate_distance:
                enabled = True
                self.release_count[name] = 0
            elif self.active[name] and minimum <= self.deactivate_distance:
                enabled = True
                self.release_count[name] = 0
            elif self.active[name]:
                self.release_count[name] += 1
                enabled = self.release_count[name] < self.hold_steps
            else:
                enabled = False
            self.active[name] = enabled
            self.previous[name] = minimum
            if enabled:
                self.selected[name] = self._spatial_selection(points, hits)
            closest = min(range(len(hits)), key=lambda index: (hits[index].signed_distance, index))
            self.measurements[name] = {
                "signed_distance": float(hits[closest].signed_distance),
                "slack": float(hits[closest].signed_distance - self.margin),
                "surface_id": hits[closest].surface_id,
                "point": points[closest].copy(),
            }

    def min_slack(self, configuration) -> float:
        self.prepare_active_set(configuration, 0.0)
        return min(item["slack"] for item in self.measurements.values())

    def compute_qp_inequalities(self, configuration, dt: float) -> Constraint:
        del dt
        rows, bounds = [], []
        for name, selected in self.selected.items():
            body_id = self.shells[name]["body_id"]
            for point, hit in selected:
                jacp = np.zeros((3, self.model.nv), dtype=float)
                jacr = np.zeros((3, self.model.nv), dtype=float)
                mj.mj_jac(self.model, configuration.data, jacp, jacr, point, body_id)
                rows.append(-hit.normal @ jacp)
                bounds.append(float(hit.signed_distance - self.margin))
        if not rows:
            return Constraint()
        return Constraint(G=np.asarray(rows), h=np.asarray(bounds))


class AutomaticSelfCollisionLimit(Limit):
    """Closest-point constraints for non-adjacent robot collision geoms."""

    def __init__(self, model: mj.MjModel, config: dict) -> None:
        self.model = model
        self.enabled = bool(config.get("enabled", True))
        self.activate_distance = float(config["activate_distance"])
        self.margin = float(config["margin"])
        self.excluded_kinematic_hops = max(0, int(config["excluded_kinematic_hops"]))
        self.geom_pairs = self._build_pairs()
        self.active_pairs: list[dict[str, Any]] = []

    def _body_distance(self, first: int, second: int) -> int:
        first_ancestors = {}
        body, distance = first, 0
        while body > 0:
            first_ancestors[body] = distance
            body = int(self.model.body_parentid[body])
            distance += 1
        body, distance = second, 0
        while body > 0:
            if body in first_ancestors:
                return distance + first_ancestors[body]
            body = int(self.model.body_parentid[body])
            distance += 1
        return 10**6

    def _build_pairs(self) -> list[tuple[int, int]]:
        if not self.enabled:
            return []
        geoms = [
            geom_id for geom_id in range(self.model.ngeom)
            if self.model.geom_contype[geom_id] != 0
            and (mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, geom_id) or "") != "floor"
        ]
        pairs = []
        for offset, first in enumerate(geoms):
            body_first = int(self.model.geom_bodyid[first])
            for second in geoms[offset + 1 :]:
                body_second = int(self.model.geom_bodyid[second])
                if body_first == body_second:
                    continue
                if self._body_distance(body_first, body_second) <= self.excluded_kinematic_hops:
                    continue
                pairs.append((first, second))
        return pairs

    def prepare_active_set(self, configuration) -> None:
        if not self.enabled:
            self.active_pairs = []
            return
        configuration.update()
        active = []
        for first, second in self.geom_pairs:
            fromto = np.zeros(6, dtype=float)
            distance = float(mj.mj_geomDistance(
                self.model,
                configuration.data,
                first,
                second,
                self.activate_distance,
                fromto,
            ))
            if distance >= self.activate_distance:
                continue
            vector = fromto[:3] - fromto[3:]
            norm = float(np.linalg.norm(vector))
            if norm < 1e-10:
                continue
            normal = np.sign(distance if abs(distance) > 1e-12 else 1.0) * vector / norm
            active.append({
                "first": first,
                "second": second,
                "point_first": fromto[:3].copy(),
                "point_second": fromto[3:].copy(),
                "normal": normal,
                "distance": distance,
            })
        self.active_pairs = active

    def compute_qp_inequalities(self, configuration, dt: float) -> Constraint:
        del dt
        rows, bounds = [], []
        for item in self.active_pairs:
            first, second = item["first"], item["second"]
            jac_first = np.zeros((3, self.model.nv), dtype=float)
            jac_second = np.zeros((3, self.model.nv), dtype=float)
            jacr = np.zeros((3, self.model.nv), dtype=float)
            mj.mj_jac(
                self.model,
                configuration.data,
                jac_first,
                jacr,
                item["point_first"],
                int(self.model.geom_bodyid[first]),
            )
            mj.mj_jac(
                self.model,
                configuration.data,
                jac_second,
                jacr,
                item["point_second"],
                int(self.model.geom_bodyid[second]),
            )
            relative = item["normal"] @ (jac_first - jac_second)
            rows.append(-relative)
            bounds.append(float(item["distance"] - self.margin))
        if not rows:
            return Constraint()
        return Constraint(G=np.asarray(rows), h=np.asarray(bounds))


class WholeBodyOmniGMRV3:
    """Independent Omni-first retargeter with no GMR per-link FrameTasks."""

    def __init__(
        self,
        config_path: str | Path,
        terrain: TerrainField,
        environment_pool: np.ndarray,
        fps: float = 50.0,
        solver: str = "daqp",
    ) -> None:
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text())
        robot_xml = Path(self.config["robot_xml"])
        if not robot_xml.is_absolute():
            robot_xml = self.config_path.parents[2] / robot_xml
        self.robot_xml = robot_xml.resolve()
        self.model = mj.MjModel.from_xml_path(str(self.robot_xml))
        for joint_name, bounds in self.config.get("joint_position_limits", {}).items():
            joint_id = self.model.joint(joint_name).id
            if self.model.jnt_type[joint_id] not in (
                mj.mjtJoint.mjJNT_HINGE,
                mj.mjtJoint.mjJNT_SLIDE,
            ):
                raise ValueError(f"V3 custom limit requires a hinge or slide joint: {joint_name}")
            lower, upper = (float(bounds[0]), float(bounds[1]))
            if not lower < upper:
                raise ValueError(f"Invalid V3 joint range for {joint_name}: {bounds}")
            self.model.jnt_range[joint_id] = (lower, upper)
            self.model.jnt_limited[joint_id] = 1
        self.configuration = mink.Configuration(self.model)
        self.terrain = terrain
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.solver = solver
        self.damping = float(self.config["solver"]["damping"])
        self.semantic_mapping = self.config["semantic_points"]
        robot_specs = {name: item["robot"] for name, item in self.semantic_mapping.items()}
        interaction = self.config["interaction_graph"]
        environment_points = int(np.clip(
            np.ceil(len(robot_specs) * float(interaction["environment_points_per_semantic"])),
            int(interaction["environment_points_minimum"]),
            min(int(interaction["environment_points_maximum"]), len(environment_pool)),
        ))
        self.interaction_task = InteractionLaplacianTask(
            self.model,
            robot_specs,
            environment_pool,
            environment_points,
            interaction["semantic_cost"],
            interaction["environment_cost"],
            interaction["gain"],
        )
        root = self.config["global_anchor"]
        self.root_task = RootGaugeTask(self.model, root["robot_body"], root["cost"])
        self.torso_task = TorsoPelvisCoherenceTask(
            self.model, self.config["torso_pelvis_coherence"]
        )

        contact = self.config["contact_tasks"]
        self.contact_task = TerrainPointContactTask(
            self.model,
            contact["robot_points"],
            contact["normal_cost"],
            contact["tangent_cost"],
            contact["clearance"],
        )
        self.foot_orientation_task = TerrainFootOrientationTask(
            self.model,
            contact["foot_bodies"],
            np.asarray(contact["sole_local_normal"], dtype=float),
            contact["foot_orientation_cost"],
        )
        self.terrain_limit = AutomaticMeshTerrainLimit(
            self.model, terrain, self.config["terrain_nonpenetration"]
        )
        self.self_collision_limit = AutomaticSelfCollisionLimit(
            self.model, self.config["self_collision"]
        )
        self.trust_limit = TangentTrustRegionLimit(
            self.model, self.config["solver"]["trust_region"]
        )
        self.configuration_limit = mink.ConfigurationLimit(self.model)
        velocity = float(self.config["solver"]["joint_velocity_limit"])
        velocity_limits = {}
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, joint_id)
                velocity_limits[name] = velocity
        self.velocity_limit = mink.VelocityLimit(self.model, velocity_limits)

        posture = self.config["posture"]
        nominal_cost = np.full(self.model.nv, float(posture["nominal_cost"]), dtype=float)
        temporal_cost = np.full(self.model.nv, float(posture["temporal_cost"]), dtype=float)
        for joint_name, cost in posture.get("joint_costs", {}).items():
            joint_id = self.model.joint(joint_name).id
            dof_address = int(self.model.jnt_dofadr[joint_id])
            nominal_cost[dof_address] = float(cost)
            temporal_cost[dof_address] = max(temporal_cost[dof_address], float(cost))
        self.nominal_task = mink.PostureTask(self.model, nominal_cost, gain=0.35, lm_damping=1.0)
        self.temporal_task = mink.PostureTask(self.model, temporal_cost, gain=0.45, lm_damping=1.0)
        self.nominal_q = self.model.qpos0.copy()
        self.nominal_task.set_target(self.nominal_q)
        self.previous_q: np.ndarray | None = None
        self.frame_index = 0
        self.diagnostics: list[dict[str, Any]] = []

    def _source_semantics(self, source_frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        missing = [
            item["source"] for item in self.semantic_mapping.values()
            if item["source"] not in source_frame
        ]
        if missing:
            raise ValueError(f"Source frame is missing V3 semantic joints: {sorted(set(missing))}")
        return {
            name: np.asarray(source_frame[item["source"]], dtype=float)
            for name, item in self.semantic_mapping.items()
        }

    def _initialize(self, source_frame: dict[str, np.ndarray], root_quaternion: np.ndarray) -> None:
        root_source = self.config["global_anchor"]["source"]
        q = self.model.qpos0.copy()
        q[:3] = np.asarray(source_frame[root_source], dtype=float)
        q[3:7] = np.asarray(root_quaternion, dtype=float)
        q[3:7] /= max(float(np.linalg.norm(q[3:7])), 1e-12)
        self.configuration.update(q)
        self.nominal_q[:7] = q[:7]
        self.nominal_task.set_target(self.nominal_q)

    def retarget(
        self,
        source_frame: dict[str, np.ndarray],
        root_quaternion: np.ndarray,
        contact_frame: dict,
        chest_quaternion: np.ndarray | None = None,
    ) -> np.ndarray:
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
        self.foot_orientation_task.set_contacts(contacts, contact_frame.get("flat_foot", {}))
        if self.previous_q is not None:
            self.temporal_task.set_target(self.previous_q)

        solver_cfg = self.config["solver"]
        first = self.frame_index == 0
        passes = int(solver_cfg["first_frame_iterations"] if first else solver_cfg["iterations"])
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
            if not first:
                limits.append(self.velocity_limit)
            try:
                velocity = mink.solve_ik(
                    self.configuration,
                    tasks,
                    solve_dt,
                    self.solver,
                    self.damping,
                    limits=limits,
                )
                self.configuration.integrate_inplace(velocity, solve_dt)
            except Exception as error:
                failures.append(f"{type(error).__name__}: {error}")
                break

        self.terrain_limit.prepare_active_set(self.configuration, self.dt)
        self.self_collision_limit.prepare_active_set(self.configuration)
        minimum_slack = min(
            (item["slack"] for item in self.terrain_limit.measurements.values()),
            default=np.inf,
        )
        output = self.configuration.data.qpos.copy()
        delta = np.zeros(self.model.nv)
        if self.previous_q is not None:
            mj.mj_differentiatePos(self.model, delta, self.dt, self.previous_q, output)
        self.diagnostics.append({
            "frame": self.frame_index,
            "qp_failures": failures,
            "active_collision_shells": sorted(self.terrain_limit.selected),
            "active_collision_points": int(sum(len(value) for value in self.terrain_limit.selected.values())),
            "active_self_collision_pairs": int(len(self.self_collision_limit.active_pairs)),
            "minimum_self_distance": float(min(
                (item["distance"] for item in self.self_collision_limit.active_pairs),
                default=np.inf,
            )),
            "minimum_terrain_slack": float(minimum_slack),
            "max_velocity": float(np.max(np.abs(delta), initial=0.0)),
            "torso_pelvis_targets": {
                key: float(value) for key, value in self.torso_task.targets.items()
            },
            "contact_states": {
                name: {
                    "score": float(item.get("score", 0.0)),
                    "state": str(item.get("state", "NONE")),
                    "surface_id": str(item.get("surface_id", "")),
                }
                for name, item in contacts.items()
            },
        })
        self.previous_q = output.copy()
        self.frame_index += 1
        return output
