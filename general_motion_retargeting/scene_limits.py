"""MuJoCo-backed robot-to-scene non-penetration limit for WholeBody V3."""

from __future__ import annotations

from typing import Any
from time import perf_counter

import mujoco as mj
import numpy as np
from mink.limits.limit import Constraint, Limit


class AutomaticSceneCollisionLimit(Limit):
    """Active-set collision limit using MuJoCo GJK distance queries.

    The supplied MuJoCo model must contain both robot and static scene geoms.
    Scene geoms are selected by ``scene_body_ids`` or ``scene_body_prefix``;
    no simulation step or contact impulse is used.
    """

    def __init__(self, model: mj.MjModel, config: dict[str, Any], scene_body_ids: set[int] | None = None) -> None:
        self.model = model
        self.enabled = bool(config.get("enabled", True))
        self.activate_distance = float(config.get("activate_distance", 0.06))
        self.deactivate_distance = float(config.get("deactivate_distance", 0.09))
        self.margin = float(config.get("margin", 0.004))
        self.hold_steps = max(1, int(config.get("hold_steps", 3)))
        self.max_per_robot_geom = max(1, int(config.get("max_active_per_robot_geom", 3)))
        self.scene_body_ids = set(scene_body_ids or set())
        prefix = str(config.get("scene_body_prefix", "scene_"))
        self.scene_geoms = [
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom_contype[geom_id] != 0
            and (
                int(model.geom_bodyid[geom_id]) in self.scene_body_ids
                or (
                    mj.mj_id2name(
                        model,
                        mj.mjtObj.mjOBJ_BODY,
                        int(model.geom_bodyid[geom_id]),
                    )
                    or ""
                ).startswith(prefix)
            )
        ]
        # Static world geoms (notably the floor) are environment geometry, not
        # articulated robot geometry. Analytic terrain constraints own them.
        self.robot_geoms = [
            geom_id
            for geom_id in range(model.ngeom)
            if geom_id not in self.scene_geoms
            and int(model.geom_bodyid[geom_id]) != 0
            and model.geom_contype[geom_id] != 0
        ]
        self.active_pairs: list[dict[str, Any]] = []
        self._held: dict[tuple[int, int], int] = {}
        self.query_runtime_seconds = 0.0
        self.minimum_distance = np.inf
        self.maximum_penetration = 0.0

    def prepare_active_set(self, configuration, dt: float = 0.0) -> None:
        del dt
        started = perf_counter()
        if not self.enabled:
            self.active_pairs = []
            self.query_runtime_seconds = perf_counter() - started
            return
        configuration.update()
        candidates = []
        for robot_geom in self.robot_geoms:
            per_geom = []
            for scene_geom in self.scene_geoms:
                fromto = np.zeros(6, dtype=float)
                distance = float(mj.mj_geomDistance(self.model, configuration.data, robot_geom, scene_geom, self.deactivate_distance, fromto))
                key = (robot_geom, scene_geom)
                if distance <= self.activate_distance or key in self._held:
                    vector = fromto[:3] - fromto[3:]
                    norm = float(np.linalg.norm(vector))
                    if norm < 1e-10:
                        continue
                    sign = np.sign(distance if abs(distance) > 1e-12 else 1.0)
                    per_geom.append((distance, sign * vector / norm, fromto.copy(), key))
            per_geom.sort(key=lambda item: (item[0], item[3]))
            # CoACD often creates adjacent pieces with nearly identical
            # closest normals. Keep only independent escape directions for a
            # robot geom, avoiding artificial over-constraint of one limb.
            reduced = []
            for item in per_geom:
                if any(float(item[1] @ other[1]) > 0.995 and
                       np.linalg.norm(item[2][:3] - other[2][:3]) < 0.03
                       for other in reduced):
                    continue
                reduced.append(item)
                if len(reduced) >= self.max_per_robot_geom:
                    break
            candidates.extend(reduced)
        selected = []
        for distance, normal, fromto, key in candidates:
            if distance <= self.activate_distance:
                self._held[key] = 0
            elif key in self._held:
                self._held[key] += 1
                if self._held[key] >= self.hold_steps and distance > self.deactivate_distance:
                    self._held.pop(key, None)
                    continue
            selected.append({"robot_geom": key[0], "scene_geom": key[1], "distance": float(distance), "normal": normal, "robot_point": fromto[:3].copy(), "scene_point": fromto[3:].copy()})
        self.active_pairs = selected
        distances = [item["distance"] for item in selected]
        self.minimum_distance = float(min(distances, default=np.inf))
        self.maximum_penetration = float(max(0.0, -min(distances, default=0.0)))
        self.query_runtime_seconds = perf_counter() - started

    def measure_current_distances(self, configuration) -> dict[str, Any]:
        """Query the final FK state without mutating active-set hysteresis."""
        started = perf_counter()
        configuration.update()
        minimum = np.inf
        maximum_penetration = 0.0
        closest: dict[str, Any] | None = None
        for robot_geom in self.robot_geoms:
            for scene_geom in self.scene_geoms:
                fromto = np.zeros(6, dtype=float)
                distance = float(
                    mj.mj_geomDistance(
                        self.model,
                        configuration.data,
                        robot_geom,
                        scene_geom,
                        self.deactivate_distance,
                        fromto,
                    )
                )
                if distance < minimum:
                    minimum = distance
                    closest = {
                        "robot_geom_id": int(robot_geom),
                        "robot_geom": mj.mj_id2name(
                            self.model, mj.mjtObj.mjOBJ_GEOM, robot_geom
                        )
                        or str(robot_geom),
                        "scene_geom_id": int(scene_geom),
                        "scene_geom": mj.mj_id2name(
                            self.model, mj.mjtObj.mjOBJ_GEOM, scene_geom
                        )
                        or str(scene_geom),
                    }
                maximum_penetration = max(maximum_penetration, -distance)
        return {
            "minimum_scene_distance": float(minimum),
            "maximum_penetration": float(max(0.0, maximum_penetration)),
            "closest_scene_collision_pair": closest,
            "scene_collision_final_query_runtime_seconds": perf_counter() - started,
        }

    def compute_qp_inequalities(self, configuration, dt: float) -> Constraint:
        del dt
        rows, bounds = [], []
        for item in self.active_pairs:
            body_id = int(self.model.geom_bodyid[item["robot_geom"]])
            jacp = np.zeros((3, self.model.nv), dtype=float)
            jacr = np.zeros((3, self.model.nv), dtype=float)
            mj.mj_jac(self.model, configuration.data, jacp, jacr, item["robot_point"], body_id)
            rows.append(-item["normal"] @ jacp)
            bounds.append(float(item["distance"] - self.margin))
        return Constraint(G=np.asarray(rows), h=np.asarray(bounds)) if rows else Constraint()

    def diagnostics(self) -> dict[str, Any]:
        distances = [item["distance"] for item in self.active_pairs]
        pairs = [
            {
                "robot_geom_id": int(item["robot_geom"]),
                "robot_geom": mj.mj_id2name(
                    self.model, mj.mjtObj.mjOBJ_GEOM, item["robot_geom"]
                )
                or str(item["robot_geom"]),
                "scene_geom_id": int(item["scene_geom"]),
                "scene_geom": mj.mj_id2name(
                    self.model, mj.mjtObj.mjOBJ_GEOM, item["scene_geom"]
                )
                or str(item["scene_geom"]),
                "distance": float(item["distance"]),
            }
            for item in self.active_pairs
        ]
        return {
            "active_scene_collision_pairs": len(self.active_pairs),
            "minimum_scene_distance": float(min(distances, default=np.inf)),
            "maximum_penetration": float(max(0.0, -min(distances, default=0.0))),
            "active_scene_collision_pair_details": pairs,
            "scene_collision_query_runtime_seconds": float(
                self.query_runtime_seconds
            ),
        }

    def finite_difference_check(self, configuration, eps: float = 1e-6, samples: int = 8) -> dict[str, Any]:
        """Numerically verify n^T J direction for current active pairs."""
        if not self.active_pairs:
            return {"pairs_tested": 0, "max_abs_error": 0.0}
        q0 = configuration.q.copy(); rng = np.random.default_rng(0); errors = []
        try:
            for item in self.active_pairs[:samples]:
                body = int(self.model.geom_bodyid[item["robot_geom"]]); jp = np.zeros((3,self.model.nv)); jr = np.zeros((3,self.model.nv))
                mj.mj_jac(self.model, configuration.data, jp, jr, item["robot_point"], body)
                v = rng.standard_normal(self.model.nv); v /= max(np.linalg.norm(v), 1e-12)
                predicted = float(item["normal"] @ jp @ v)
                q = q0.copy(); q[:self.model.nv] += eps * v; configuration.update(q)
                fp = np.zeros(6); dp = float(mj.mj_geomDistance(self.model, configuration.data, item["robot_geom"], item["scene_geom"], self.deactivate_distance, fp))
                q[:self.model.nv] -= 2*eps*v; configuration.update(q)
                fm = np.zeros(6); dm = float(mj.mj_geomDistance(self.model, configuration.data, item["robot_geom"], item["scene_geom"], self.deactivate_distance, fm))
                errors.append(abs(predicted - (dp-dm)/(2*eps)))
        finally:
            configuration.update(q0)
        return {"pairs_tested": len(errors), "max_abs_error": float(max(errors, default=0.0))}
