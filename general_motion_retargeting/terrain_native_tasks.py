"""Terrain-native interaction, sole-patch tasks and hard constraints."""
from __future__ import annotations
from typing import Any
import mujoco as mj
import numpy as np
from mink.limits.limit import Constraint, Limit
from mink.tasks.task import Task
from scipy.spatial import Delaunay, QhullError
from .terrain_tasks import TerrainPointContactTask, tangent_basis


def _unit(value: np.ndarray, fallback=(0.0, 0.0, 1.0)) -> np.ndarray:
    value = np.asarray(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        return np.asarray(fallback, dtype=float)
    return value / norm


class WeightedInteractionLaplacianTask(Task):
    """HoloSoMo-style interaction mesh with sparse distance-weighted edges."""

    def __init__(
        self,
        model: mj.MjModel,
        robot_points: dict[str, Any],
        environment_pool: np.ndarray,
        environment_points: int,
        semantic_cost: float,
        environment_cost: float,
        gain: float,
        distance_decay: float = 3.0,
        points_per_semantic: int = 5,
    ) -> None:
        self.model = model
        self.robot_points = dict(robot_points)
        self.names = list(robot_points)
        self.environment_pool = np.asarray(environment_pool, dtype=float).reshape((-1, 3))
        self.environment_count = min(max(4, int(environment_points)), len(self.environment_pool))
        self.distance_decay = max(0.0, float(distance_decay))
        self.points_per_semantic = max(1, int(points_per_semantic))
        self.vertex_count = len(self.names) + self.environment_count
        self.laplacian = np.zeros((self.vertex_count, self.vertex_count), dtype=float)
        self.target = np.zeros((self.vertex_count, 3), dtype=float)
        self.environment = self.environment_pool[: self.environment_count].copy()
        costs = np.repeat(np.r_[
            np.full(len(self.names), float(semantic_cost)),
            np.full(self.environment_count, float(environment_cost)),
        ], 3)
        super().__init__(cost=costs, gain=float(gain), lm_damping=1.0)

    def _select_environment(self, human: np.ndarray) -> np.ndarray:
        if len(self.environment_pool) <= self.environment_count:
            return self.environment_pool.copy()
        selected: list[int] = []
        for point in human:
            d2 = np.sum((self.environment_pool - point[None, :]) ** 2, axis=1)
            for index in np.argsort(d2, kind="stable")[: self.points_per_semantic]:
                if int(index) not in selected:
                    selected.append(int(index))
        if len(selected) < self.environment_count:
            d2 = np.min(
                np.sum((self.environment_pool[:, None, :] - human[None, :, :]) ** 2, axis=-1),
                axis=1,
            )
            for index in np.argsort(d2, kind="stable"):
                if int(index) not in selected:
                    selected.append(int(index))
                if len(selected) >= self.environment_count:
                    break
        return self.environment_pool[np.asarray(selected[: self.environment_count], dtype=int)].copy()

    def _laplacian(self, vertices: np.ndarray) -> np.ndarray:
        count = len(vertices)
        edges: set[tuple[int, int]] = set()
        try:
            simplices = Delaunay(vertices, qhull_options="QJ").simplices
            for simplex in simplices:
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        edges.add(tuple(sorted((int(simplex[i]), int(simplex[j])))))
        except QhullError:
            distance = np.linalg.norm(vertices[:, None] - vertices[None, :], axis=-1)
            for i in range(count):
                for j in np.argsort(distance[i])[1:5]:
                    edges.add(tuple(sorted((i, int(j)))))
        adjacency: list[list[tuple[int, float]]] = [[] for _ in range(count)]
        for a, b in sorted(edges):
            length = float(np.linalg.norm(vertices[a] - vertices[b]))
            weight = float(np.exp(-self.distance_decay * length)) if self.distance_decay > 0.0 else 1.0
            adjacency[a].append((b, weight))
            adjacency[b].append((a, weight))
        matrix = np.zeros((count, count), dtype=float)
        for i, neighbors in enumerate(adjacency):
            if not neighbors:
                continue
            total = max(sum(weight for _, weight in neighbors), 1e-12)
            matrix[i, i] = 1.0
            for j, weight in neighbors:
                matrix[i, j] = -weight / total
        return matrix

    def set_source(self, source_points: dict[str, np.ndarray]) -> None:
        human = np.asarray([source_points[name] for name in self.names], dtype=float)
        self.environment = self._select_environment(human)
        vertices = np.vstack((human, self.environment))
        self.laplacian = self._laplacian(vertices)
        self.target = self.laplacian @ vertices

    def _current(self, configuration):
        points = np.asarray([self.robot_points[name].point(configuration) for name in self.names])
        jacobians = [self.robot_points[name].jacobian(configuration) for name in self.names]
        return np.vstack((points, self.environment)), jacobians

    def compute_error(self, configuration) -> np.ndarray:
        vertices, _ = self._current(configuration)
        return (self.laplacian @ vertices - self.target).reshape(-1)

    def compute_jacobian(self, configuration) -> np.ndarray:
        _, semantic_jacobians = self._current(configuration)
        vertex_jacobian = np.zeros((3 * self.vertex_count, self.model.nv), dtype=float)
        for index, jacobian in enumerate(semantic_jacobians):
            vertex_jacobian[3 * index: 3 * index + 3] = jacobian
        return np.kron(self.laplacian, np.eye(3)) @ vertex_jacobian


class RobotSole:
    def __init__(self, model: mj.MjModel, site_names: list[str]):
        self.model = model
        self.site_ids = [model.site(name).id for name in site_names]
        if len(self.site_ids) != 4:
            raise ValueError("Terrain-native sole patch requires exactly four sole sites per foot")

    def points(self, configuration) -> np.ndarray:
        return np.asarray(configuration.data.site_xpos[self.site_ids], dtype=float).copy()

    def jacobians(self, configuration) -> np.ndarray:
        rows = []
        for site_id in self.site_ids:
            jacp = np.zeros((3, self.model.nv), dtype=float)
            jacr = np.zeros((3, self.model.nv), dtype=float)
            mj.mj_jacSite(self.model, configuration.data, jacp, jacr, int(site_id))
            rows.append(jacp)
        return np.asarray(rows)


class SolePatchConstraintLimit(Limit):
    """Hard stance patch and swing-clearance constraints for both soles."""

    def __init__(self, model: mj.MjModel, soles: dict[str, RobotSole], config: dict[str, Any] | None = None):
        self.model = model
        self.soles = soles
        cfg = config or {}
        self.clearance = float(cfg.get("clearance", 0.004))
        self.normal_tolerance = float(cfg.get("normal_tolerance", 0.003))
        self.tangent_tolerance = float(cfg.get("tangent_tolerance", 0.010))
        self.swing_margin = float(cfg.get("swing_margin", 0.0))
        self.support_inset = float(cfg.get("support_inset", 0.003))
        self.capture_normal_distance = float(cfg.get("capture_normal_distance", 0.03))
        self.capture_normal_spread = float(cfg.get("capture_normal_spread", 0.025))
        self.capture_polygon_tolerance = float(cfg.get("capture_polygon_tolerance", 0.01))
        self.plans: dict[str, dict[str, Any]] = {}
        self.captured: dict[str, bool] = {"left": False, "right": False}
        self._capture_key: dict[str, tuple | None] = {"left": None, "right": None}
        self.active_count = 0

    def set_plan(self, plans: dict[str, dict[str, Any]]) -> None:
        self.plans = plans

    def update_capture_state(self, configuration) -> None:
        """Latch hard XY sticking only after the robot sole actually reaches a planned tread.

        HoloSoMo does not teleport a foot onto source contact geometry.  Its
        hard sticking constraint is relative to the robot's previous foot
        position.  We follow the same principle: source/terrain planning picks
        the intended patch, soft interaction/contact terms guide the foot
        there, and hard sticking activates only once the robot sole is already
        geometrically compatible with that patch.
        """
        for side, sole in self.soles.items():
            plan = self.plans.get(side, {})
            if plan.get("mode") != "stance":
                self.captured[side] = False
                self._capture_key[side] = None
                continue
            capture_key = tuple(plan.get("episode", ())) + (str(plan.get("patch_id", "")),)
            if self._capture_key.get(side) != capture_key:
                self.captured[side] = False
                self._capture_key[side] = capture_key
            if self.captured.get(side, False):
                continue
            points = sole.points(configuration)
            normal = _unit(plan["surface_normal"])
            plane = np.asarray(plan["surface_point"], dtype=float)
            distances = (points - plane) @ normal
            normal_error = np.abs(distances - self.clearance)
            halfspaces = np.asarray(
                plan.get("patch_xy_halfspaces", np.empty((0, 3))), dtype=float
            ).reshape((-1, 3))
            if len(halfspaces):
                homogeneous = np.c_[points[:, :2], np.ones(len(points))]
                polygon_violation = float(np.max(homogeneous @ halfspaces.T))
            else:
                polygon_violation = 0.0
            if (
                float(np.max(normal_error)) <= self.capture_normal_distance
                and float(np.ptp(distances)) <= self.capture_normal_spread
                and polygon_violation <= self.capture_polygon_tolerance
            ):
                self.captured[side] = True

    def compute_qp_inequalities(self, configuration, dt: float) -> Constraint:
        del dt
        rows, bounds = [], []
        active = 0
        for side, sole in self.soles.items():
            plan = self.plans.get(side, {})
            points = sole.points(configuration)
            jacobians = sole.jacobians(configuration)
            mode = plan.get("mode", "free")
            if mode == "stance":
                # Before capture, stance geometry is a soft objective only.
                # Hard constraints here would require one QP step to move a
                # sole that may still be 10--20 cm away from the source patch,
                # which is exactly what made the previous implementation
                # infeasible on stairs_0037.
                if self.captured.get(side, False):
                    halfspaces = np.asarray(
                        plan.get("patch_xy_halfspaces", np.empty((0, 3))),
                        dtype=float,
                    ).reshape((-1, 3))
                    for point, jac in zip(points, jacobians):
                        for a, b, c in halfspaces:
                            value = float(a * point[0] + b * point[1] + c)
                            rows.append(a * jac[0] + b * jac[1])
                            bounds.append(float(-self.support_inset - value))
                            active += 1
                    if "hard_anchor" in plan:
                        center = points.mean(axis=0)
                        center_jac = jacobians.mean(axis=0)
                        anchor = np.asarray(plan["hard_anchor"], dtype=float)
                        # Match HoloSoMo foot sticking: constrain only XY /
                        # tangential motion relative to the robot's captured
                        # stance point. Z/sole leveling remain soft objectives
                        # while non-penetration provides the hard lower bound.
                        tangents = tangent_basis(_unit(plan["surface_normal"]))
                        for tangent in tangents.T:
                            error = float(tangent @ (center - anchor))
                            jac = tangent @ center_jac
                            rows.extend([jac, -jac])
                            bounds.extend([
                                self.tangent_tolerance - error,
                                self.tangent_tolerance + error,
                            ])
                            active += 2
            elif mode == "swing":
                # Swing clearance is represented by TerrainNativeContactTask's
                # hinge loss. Scene non-penetration stays hard. Keeping the
                # look-ahead corridor soft avoids a new infeasible QP when a
                # frame enters swing below the desired arc.
                pass
        self.active_count = active
        if not rows:
            return Constraint()
        return Constraint(G=np.asarray(rows, dtype=float), h=np.asarray(bounds, dtype=float))


class TerrainNativeSceneLimit(Limit):
    """Compose existing scene collision with hard terrain-native sole constraints."""

    def __init__(self, scene_limit: Limit | None, sole_limit: SolePatchConstraintLimit):
        self.scene_limit = scene_limit
        self.sole_limit = sole_limit
        self.scene_geoms = list(getattr(scene_limit, "scene_geoms", ()))
        self.robot_geoms = list(getattr(scene_limit, "robot_geoms", ()))
        self.active_pairs = []
        self.query_runtime_seconds = 0.0

    def prepare_active_set(self, configuration, dt: float = 0.0) -> None:
        if self.scene_limit is not None:
            self.scene_limit.prepare_active_set(configuration, dt)
            self.active_pairs = list(getattr(self.scene_limit, "active_pairs", ()))
            self.query_runtime_seconds = float(getattr(self.scene_limit, "query_runtime_seconds", 0.0))
        else:
            self.active_pairs = []
            self.query_runtime_seconds = 0.0

    @staticmethod
    def _append_constraint(rows: list[np.ndarray], bounds: list[np.ndarray], constraint: Constraint) -> None:
        G = getattr(constraint, "G", None)
        h = getattr(constraint, "h", None)
        if G is not None and h is not None and np.size(G):
            rows.append(np.asarray(G, dtype=float))
            bounds.append(np.asarray(h, dtype=float).reshape(-1))

    def compute_qp_inequalities(self, configuration, dt: float) -> Constraint:
        rows: list[np.ndarray] = []
        bounds: list[np.ndarray] = []
        if self.scene_limit is not None:
            self._append_constraint(rows, bounds, self.scene_limit.compute_qp_inequalities(configuration, dt))
        self._append_constraint(rows, bounds, self.sole_limit.compute_qp_inequalities(configuration, dt))
        if not rows:
            return Constraint()
        return Constraint(G=np.vstack(rows), h=np.concatenate(bounds))

    def diagnostics(self) -> dict[str, Any]:
        base = self.scene_limit.diagnostics() if self.scene_limit is not None else {
            "active_scene_collision_pairs": 0,
            "minimum_scene_distance": float("inf"),
            "maximum_penetration": 0.0,
        }
        base = dict(base)
        base["terrain_native_hard_rows"] = int(self.sole_limit.active_count)
        return base

    def measure_current_distances(self, configuration) -> dict[str, Any]:
        if self.scene_limit is None:
            return {
                "minimum_scene_distance": float("inf"),
                "maximum_penetration": 0.0,
                "closest_scene_collision_pair": None,
            }
        return self.scene_limit.measure_current_distances(configuration)

    def finite_difference_check(self, configuration, *args, **kwargs):
        if self.scene_limit is None or not hasattr(self.scene_limit, "finite_difference_check"):
            return {"pairs_tested": 0, "max_abs_error": 0.0}
        return self.scene_limit.finite_difference_check(configuration, *args, **kwargs)


class NullFootTask(Task):
    def __init__(self, model: mj.MjModel):
        self.model = model
        super().__init__(cost=np.asarray([1e-12]), gain=1.0, lm_damping=1.0)

    def set_contacts(self, contacts: dict, flat_foot: dict) -> None:
        del contacts, flat_foot

    def compute_error(self, configuration) -> np.ndarray:
        del configuration
        return np.zeros(1)

    def compute_jacobian(self, configuration) -> np.ndarray:
        del configuration
        return np.zeros((1, self.model.nv))


class TerrainNativeContactTask(Task):
    """Soft guide paired with a hard four-point sole-patch constraint."""

    FOOT_NAMES = {"left_heel", "left_toe", "right_heel", "right_toe"}

    def __init__(
        self,
        model: mj.MjModel,
        channel_specs: dict[str, dict],
        soles: dict[str, RobotSole],
        sole_limit: SolePatchConstraintLimit,
        config: dict[str, Any],
    ) -> None:
        self.model = model
        self.soles = soles
        self.sole_limit = sole_limit
        nonfoot = {name: spec for name, spec in channel_specs.items() if name not in self.FOOT_NAMES}
        self.legacy = TerrainPointContactTask(
            model, nonfoot,
            float(config.get("legacy_normal_cost", 35.0)),
            float(config.get("legacy_tangent_cost", 18.0)),
            float(config.get("clearance", 0.004)),
        ) if nonfoot else None
        self.points = {} if self.legacy is None else self.legacy.points
        self.plans: dict[str, dict[str, Any]] = {}
        self._episode_keys: dict[str, tuple | None] = {"left": None, "right": None}
        self._hard_anchors: dict[str, np.ndarray] = {}
        self.sole_normal_cost = float(config.get("sole_normal_cost", 70.0))
        self.sole_tangent_cost = float(config.get("sole_tangent_cost", 18.0))
        self.swing_cost = float(config.get("swing_clearance_cost", 45.0))
        legacy_cost = np.tile(
            [float(config.get("legacy_normal_cost", 35.0)),
             float(config.get("legacy_tangent_cost", 18.0)),
             float(config.get("legacy_tangent_cost", 18.0))],
            len(nonfoot),
        )
        sole_cost = np.tile(
            np.r_[np.full(4, self.sole_normal_cost), np.full(2, self.sole_tangent_cost), np.full(4, self.swing_cost)],
            2,
        )
        super().__init__(cost=np.r_[legacy_cost, sole_cost], gain=0.55, lm_damping=1.0)

    def set_contacts(self, configuration, contacts: dict) -> None:
        if self.legacy is not None:
            self.legacy.set_contacts(configuration, {k: v for k, v in contacts.items() if k in self.points})
        plans = {}
        for side in ("left", "right"):
            plan = dict(contacts.get(f"{side}_heel", {}).get("terrain_native", {}))
            if plan.get("mode") == "stance":
                episode = tuple(plan.get("episode", ())) + (str(plan.get("patch_id", "")),)
                if self._episode_keys.get(side) != episode:
                    self._episode_keys[side] = episode
                    self._hard_anchors.pop(side, None)
            else:
                self._episode_keys[side] = None
                self._hard_anchors.pop(side, None)
            plans[side] = plan
        self.plans = plans
        self.sole_limit.set_plan(plans)
        self.sole_limit.update_capture_state(configuration)

        # Establish the robot-relative sticking anchor only after geometric
        # capture. This mirrors HoloSoMo's previous-robot-foot sticking
        # constraint instead of forcing source-space XY onto the robot.
        for side in ("left", "right"):
            plan = self.plans.get(side, {})
            if plan.get("mode") != "stance" or not self.sole_limit.captured.get(side, False):
                continue
            if side not in self._hard_anchors:
                points = self.soles[side].points(configuration)
                normal = _unit(plan["surface_normal"])
                plane = np.asarray(plan["surface_point"], dtype=float)
                center = points.mean(axis=0)
                self._hard_anchors[side] = center - normal * float(normal @ (center - plane))
            plan["hard_anchor"] = self._hard_anchors[side].copy()
        self.sole_limit.set_plan(self.plans)

    def _foot_error(self, configuration, side: str) -> np.ndarray:
        plan = self.plans.get(side, {})
        points = self.soles[side].points(configuration)
        normal_error = np.zeros(4)
        tangent_error = np.zeros(2)
        swing_error = np.zeros(4)
        if plan.get("mode") == "stance":
            normal = _unit(plan["surface_normal"])
            plane = np.asarray(plan["surface_point"], dtype=float)
            normal_error = (points - plane) @ normal - self.sole_limit.clearance
            center = points.mean(axis=0)
            tangent_error = tangent_basis(normal).T @ (center - np.asarray(plan["anchor"], dtype=float))
        elif plan.get("mode") == "swing" and np.isfinite(plan.get("clearance_floor_z", np.nan)):
            swing_error = np.maximum(0.0, float(plan["clearance_floor_z"]) - points[:, 2])
        return np.r_[normal_error, tangent_error, swing_error]

    def compute_error(self, configuration) -> np.ndarray:
        legacy = self.legacy.compute_error(configuration) if self.legacy is not None else np.empty(0)
        feet = np.concatenate([self._foot_error(configuration, side) for side in ("left", "right")])
        return np.r_[legacy, feet]

    def compute_jacobian(self, configuration) -> np.ndarray:
        legacy = self.legacy.compute_jacobian(configuration) if self.legacy is not None else np.empty((0, self.model.nv))
        foot_rows = []
        for side in ("left", "right"):
            plan = self.plans.get(side, {})
            points = self.soles[side].points(configuration)
            jacobians = self.soles[side].jacobians(configuration)
            normal_rows = np.zeros((4, self.model.nv))
            tangent_rows = np.zeros((2, self.model.nv))
            swing_rows = np.zeros((4, self.model.nv))
            if plan.get("mode") == "stance":
                normal = _unit(plan["surface_normal"])
                normal_rows = np.asarray([normal @ jac for jac in jacobians])
                tangent_rows = tangent_basis(normal).T @ jacobians.mean(axis=0)
            elif plan.get("mode") == "swing" and np.isfinite(plan.get("clearance_floor_z", np.nan)):
                floor = float(plan["clearance_floor_z"])
                for i, (point, jac) in enumerate(zip(points, jacobians)):
                    if point[2] < floor:
                        swing_rows[i] = -jac[2]
            foot_rows.append(np.vstack((normal_rows, tangent_rows, swing_rows)))
        return np.vstack([legacy, *foot_rows])

    def diagnostics(self, configuration) -> dict[str, Any]:
        result = {}
        for side in ("left", "right"):
            plan = self.plans.get(side, {})
            points = self.soles[side].points(configuration)
            record = {
                "mode": str(plan.get("mode", "free")),
                "sole_points": points.tolist(),
                "captured": bool(self.sole_limit.captured.get(side, False)),
            }
            if plan.get("mode") == "stance":
                normal = _unit(plan["surface_normal"])
                plane = np.asarray(plan["surface_point"], dtype=float)
                distances = (points - plane) @ normal
                record.update({
                    "patch_id": str(plan.get("patch_id", "")),
                    "normal_distances": distances.tolist(),
                    "normal_spread": float(np.ptp(distances)),
                    "anchor_error": float(np.linalg.norm(tangent_basis(normal).T @ (points.mean(axis=0) - np.asarray(plan["anchor"], dtype=float)))),
                })
            elif plan.get("mode") == "swing":
                floor = float(plan.get("clearance_floor_z", -np.inf))
                record.update({
                    "landing_patch_id": str(plan.get("landing_patch_id", "")),
                    "clearance_floor_z": floor,
                    "minimum_clearance": float(points[:, 2].min() - floor),
                })
            result[side] = record
        return result


def make_ne01_soles(model: mj.MjModel) -> dict[str, RobotSole]:
    return {
        "left": RobotSole(model, [
            "ground_guard_left_heel_inner", "ground_guard_left_heel_outer",
            "ground_guard_left_toe_inner", "ground_guard_left_toe_outer",
        ]),
        "right": RobotSole(model, [
            "ground_guard_right_heel_inner", "ground_guard_right_heel_outer",
            "ground_guard_right_toe_inner", "ground_guard_right_toe_outer",
        ]),
    }
