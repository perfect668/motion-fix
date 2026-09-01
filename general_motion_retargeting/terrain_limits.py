"""Terrain signed-distance non-penetration constraints for Mink IK."""

from __future__ import annotations

import mujoco as mj
import numpy as np
from mink.limits.limit import Constraint, Limit

from .terrain_geometry import TerrainField, TerrainSurfaceHit


class TerrainNonPenetrationLimit(Limit):
    def __init__(
        self,
        model: mj.MjModel,
        terrain: TerrainField,
        regions: dict,
        guard_sites: dict,
        collision_geoms: dict,
        collision_points: dict,
        mesh_guards: dict,
        config: dict,
    ) -> None:
        self.model = model
        self.terrain = terrain
        self.default_margin = float(config.get("default_margin", 0.008))
        self.adaptive = bool(config.get("adaptive_activation", True))
        self.activate_distance = float(config.get("activate_distance", 0.06))
        self.deactivate_distance = float(config.get("deactivate_distance", 0.09))
        self.deactivate_hold_steps = max(1, int(config.get("deactivate_hold_steps", 3)))
        self.prediction_horizon = float(config.get("prediction_horizon", 2.0))
        self.contact_activation_score = float(config.get("contact_activation_score", 0.15))
        self.mesh_proxy_points = max(8, int(config.get("mesh_proxy_points", 96)))
        self.active_points_per_shell = max(1, int(config.get("active_points_per_shell", 5)))
        self.entries: list[dict] = []
        for name, spec in regions.items():
            self.entries.append(self._body_entry(name, spec, str(name)))
        for name, margin in guard_sites.items():
            self.entries.append({
                "name": name,
                "kind": "site",
                "site_id": model.site(name).id,
                "region": self._region(name),
                "margin": float(margin),
            })
        for name, margin in collision_geoms.items():
            geom_id = model.geom(name).id
            self.entries.append({
                "name": name,
                "kind": "geom",
                "geom_id": geom_id,
                "body_id": int(model.geom_bodyid[geom_id]),
                "region": self._region(name),
                "margin": float(margin),
            })
        for name, spec in collision_points.items():
            self.entries.append(self._body_entry(name, spec, str(spec.get("region", name))))
        names = [entry["name"] for entry in self.entries]
        if len(names) != len(set(names)):
            raise ValueError("Terrain collision candidate names must be unique")

        count = len(self.entries)
        self.active_mask = np.zeros(count, dtype=bool)
        self.previous_distance = np.full(count, np.nan)
        self.deactivate_counter = np.zeros(count, dtype=np.int32)
        self.active_indices: list[int] = []
        self.entry_hits: dict[str, TerrainSurfaceHit] = {}
        self.entry_points: dict[str, np.ndarray] = {}
        self.mesh_guards = self._build_mesh_guards(mesh_guards)
        self.mesh_previous_distance = {name: np.nan for name in self.mesh_guards}
        self.mesh_active_mask = {name: False for name in self.mesh_guards}
        self.mesh_deactivate_counter = {name: 0 for name in self.mesh_guards}
        self.mesh_selected: dict[str, list[tuple[np.ndarray, TerrainSurfaceHit]]] = {}
        self.all_measurements: dict[str, dict] = {}
        self.active_count = 0

    def _body_entry(self, name: str, spec: dict, region: str) -> dict:
        return {
            "name": name,
            "kind": "body",
            "body_id": self.model.body(spec["robot_body"]).id,
            "offset": np.asarray(spec.get("robot_offset", [0, 0, 0]), dtype=float),
            "region": region,
            "margin": float(spec.get("margin", self.default_margin)),
        }

    @staticmethod
    def _quat_matrix(quaternion: np.ndarray) -> np.ndarray:
        matrix = np.empty(9, dtype=float)
        mj.mju_quat2Mat(matrix, quaternion)
        return matrix.reshape(3, 3)

    @staticmethod
    def _region(name: str) -> str:
        side = "left" if "left" in name else "right"
        if "palm" in name or "hand" in name:
            return f"{side}_palm"
        if "knee" in name:
            return f"{side}_knee"
        if "shin" in name:
            return f"{side}_shin"
        if "thigh" in name:
            return f"{side}_knee"
        return f"{side}_foot"

    @staticmethod
    def _mesh_regions(name: str) -> tuple[str, ...]:
        side = "left" if name.startswith("left") else "right"
        if "palm" in name or "hand" in name:
            return (f"{side}_palm",)
        if "knee" in name or "shin" in name:
            return (f"{side}_knee", f"{side}_shin")
        if "thigh" in name:
            return (f"{side}_knee",)
        return (f"{side}_heel", f"{side}_toe")

    @staticmethod
    def _deterministic_proxy_sample(vertices: np.ndarray, count: int) -> np.ndarray:
        if len(vertices) <= count:
            return vertices.copy()
        center = vertices.mean(axis=0)
        selected = [int(np.argmax(np.sum((vertices - center) ** 2, axis=1)))]
        min_distance = np.sum((vertices - vertices[selected[0]]) ** 2, axis=1)
        for _ in range(1, count):
            index = int(np.argmax(min_distance))
            selected.append(index)
            min_distance = np.minimum(min_distance, np.sum((vertices - vertices[index]) ** 2, axis=1))
        return vertices[np.asarray(selected, dtype=int)]

    def _build_mesh_guards(self, specifications: dict) -> dict:
        guards = {}
        for name, spec in specifications.items():
            body_id = self.model.body(spec["body"]).id
            vertices = []
            for geom_id in np.flatnonzero(self.model.geom_bodyid == body_id):
                if self.model.geom_type[geom_id] != mj.mjtGeom.mjGEOM_MESH:
                    continue
                mesh_id = int(self.model.geom_dataid[geom_id])
                start, count = int(self.model.mesh_vertadr[mesh_id]), int(self.model.mesh_vertnum[mesh_id])
                mesh_vertices = np.asarray(self.model.mesh_vert[start:start + count], dtype=float)
                rotation = self._quat_matrix(self.model.geom_quat[geom_id])
                vertices.append(self.model.geom_pos[geom_id] + mesh_vertices @ rotation.T)
            if not vertices:
                raise ValueError(f"No mesh geometry found on terrain guard body {spec['body']}")
            local_vertices = np.concatenate(vertices)
            guards[name] = {
                "body_id": body_id,
                "proxies": self._deterministic_proxy_sample(local_vertices, self.mesh_proxy_points),
                "margin": float(spec.get("margin", self.default_margin)),
                "separation": float(spec.get("separation", 0.015)),
                "regions": self._mesh_regions(name),
            }
        return guards

    def _point(self, configuration, entry: dict) -> np.ndarray:
        if entry["kind"] == "site":
            return configuration.data.site_xpos[entry["site_id"]].copy()
        if entry["kind"] == "geom":
            return configuration.data.geom_xpos[entry["geom_id"]].copy()
        body_id = entry["body_id"]
        rotation = configuration.data.xmat[body_id].reshape(3, 3)
        return configuration.data.xpos[body_id] + rotation @ entry["offset"]

    def _mesh_world(self, configuration, specification: dict) -> np.ndarray:
        body_id = specification["body_id"]
        rotation = configuration.data.xmat[body_id].reshape(3, 3)
        return configuration.data.xpos[body_id] + specification["proxies"] @ rotation.T

    @staticmethod
    def _contact_score(contact_scores: dict, region: str) -> float:
        aliases = {
            "left_foot": ("left_heel", "left_toe"),
            "right_foot": ("right_heel", "right_toe"),
            "left_hand": ("left_palm",),
            "right_hand": ("right_palm",),
        }
        names = aliases.get(region, (region,))
        return max((float(contact_scores.get(name, {}).get("score", 0.0)) for name in names), default=0.0)

    def _select_mesh_points(self, points: np.ndarray, hits: list[TerrainSurfaceHit], separation: float):
        order = sorted(range(len(points)), key=lambda index: (hits[index].signed_distance, index))
        selected: list[tuple[np.ndarray, TerrainSurfaceHit]] = []
        separation2 = separation * separation
        for index in order:
            point = points[index]
            if not selected or all(float(np.sum((point - old_point) ** 2)) >= separation2 for old_point, _ in selected):
                selected.append((point, hits[index]))
                if len(selected) >= self.active_points_per_shell:
                    break
        return selected

    def measure_all_candidates(self, configuration) -> dict[str, dict]:
        configuration.update()
        measurements = {}
        self.entry_hits, self.entry_points = {}, {}
        for entry in self.entries:
            point = self._point(configuration, entry)
            hit = self.terrain.nearest_surface(point)
            self.entry_points[entry["name"]] = point
            self.entry_hits[entry["name"]] = hit
            measurements[entry["name"]] = self._measurement(hit, entry["margin"], point)
        self.mesh_selected = {}
        for name, spec in self.mesh_guards.items():
            points = self._mesh_world(configuration, spec)
            hits = self.terrain.nearest_surface_batch(points)
            selected = self._select_mesh_points(points, hits, spec["separation"])
            self.mesh_selected[name] = selected
            closest_index = min(
                range(len(hits)),
                key=lambda index: (hits[index].signed_distance, hits[index].surface_id, index),
            )
            measurements[f"mesh:{name}"] = self._measurement(
                hits[closest_index], spec["margin"], points[closest_index]
            )
        self.all_measurements = measurements
        return measurements

    @staticmethod
    def _measurement(hit: TerrainSurfaceHit, margin: float, point: np.ndarray) -> dict:
        return {
            "signed_distance": float(hit.signed_distance),
            "margin": float(margin),
            "slack": float(hit.signed_distance - margin),
            "surface_id": hit.surface_id,
            "surface_normal": hit.normal.copy(),
            "point": np.asarray(point, dtype=float).copy(),
            "closest_point": hit.closest_point.copy(),
        }

    def prepare_active_set(self, configuration, dt: float, contact_scores: dict) -> list[int]:
        measurements = self.measure_all_candidates(configuration)
        valid_dt = np.isfinite(dt) and dt > 1e-8
        active = []
        for index, entry in enumerate(self.entries):
            distance = measurements[entry["name"]]["signed_distance"]
            previous = self.previous_distance[index]
            speed = (distance - previous) / dt if valid_dt and np.isfinite(previous) else 0.0
            predicted = distance + self.prediction_horizon * dt * min(speed, 0.0)
            contact = self._contact_score(contact_scores, entry["region"])
            immediate = distance <= entry["margin"] or predicted <= self.activate_distance
            if not self.adaptive or immediate or distance <= self.activate_distance or contact > self.contact_activation_score:
                enabled = True
                self.deactivate_counter[index] = 0
            elif self.active_mask[index] and distance <= self.deactivate_distance:
                enabled = True
                self.deactivate_counter[index] = 0
            elif self.active_mask[index]:
                self.deactivate_counter[index] += 1
                enabled = self.deactivate_counter[index] < self.deactivate_hold_steps
            else:
                enabled = False
            self.active_mask[index] = enabled
            self.previous_distance[index] = distance
            if enabled:
                active.append(index)

        for name, spec in self.mesh_guards.items():
            key = f"mesh:{name}"
            distance = measurements[key]["signed_distance"]
            previous = self.mesh_previous_distance[name]
            speed = (distance - previous) / dt if valid_dt and np.isfinite(previous) else 0.0
            predicted = distance + self.prediction_horizon * dt * min(speed, 0.0)
            contact = max((self._contact_score(contact_scores, region) for region in spec["regions"]), default=0.0)
            immediate = distance <= spec["margin"] or predicted <= self.activate_distance
            if not self.adaptive or immediate or distance <= self.activate_distance or contact > self.contact_activation_score:
                enabled = True
                self.mesh_deactivate_counter[name] = 0
            elif self.mesh_active_mask[name] and distance <= self.deactivate_distance:
                enabled = True
                self.mesh_deactivate_counter[name] = 0
            elif self.mesh_active_mask[name]:
                self.mesh_deactivate_counter[name] += 1
                enabled = self.mesh_deactivate_counter[name] < self.deactivate_hold_steps
            else:
                enabled = False
            self.mesh_active_mask[name] = enabled
            self.mesh_previous_distance[name] = distance
        self.active_indices = active
        self.active_count = len(active) + sum(
            len(self.mesh_selected[name]) for name, enabled in self.mesh_active_mask.items() if enabled
        )
        return active

    def force_activate_violations(self, configuration) -> list[str]:
        measurements = self.measure_all_candidates(configuration)
        forced = []
        for index, entry in enumerate(self.entries):
            if measurements[entry["name"]]["slack"] < 0.0 and not self.active_mask[index]:
                self.active_mask[index] = True
                self.deactivate_counter[index] = 0
                forced.append(entry["name"])
        for name in self.mesh_guards:
            if measurements[f"mesh:{name}"]["slack"] < 0.0 and not self.mesh_active_mask[name]:
                self.mesh_active_mask[name] = True
                self.mesh_deactivate_counter[name] = 0
                forced.append(f"mesh:{name}")
        self.active_indices = [index for index, enabled in enumerate(self.active_mask) if enabled]
        return forced

    def min_signed_slack(self, configuration) -> float:
        measurements = self.measure_all_candidates(configuration)
        return min((item["slack"] for item in measurements.values()), default=np.inf)

    def measure_current_slacks(self, configuration) -> dict[str, dict]:
        return self.measure_all_candidates(configuration)

    def _jacobian(self, configuration, entry: dict, point: np.ndarray) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        if entry["kind"] == "site":
            mj.mj_jacSite(self.model, configuration.data, jacp, jacr, entry["site_id"])
        else:
            mj.mj_jac(self.model, configuration.data, jacp, jacr, point, entry["body_id"])
        return jacp

    def compute_qp_inequalities(self, configuration, dt: float) -> Constraint:
        del dt
        rows, bounds = [], []
        for index in self.active_indices:
            entry = self.entries[index]
            name = entry["name"]
            point, hit = self.entry_points[name], self.entry_hits[name]
            jacobian = self._jacobian(configuration, entry, point)
            rows.append(-hit.normal @ jacobian)
            bounds.append(float(hit.signed_distance - entry["margin"]))
        for name, enabled in self.mesh_active_mask.items():
            if not enabled:
                continue
            spec = self.mesh_guards[name]
            for point, hit in self.mesh_selected[name]:
                jacp = np.zeros((3, self.model.nv), dtype=float)
                jacr = np.zeros((3, self.model.nv), dtype=float)
                mj.mj_jac(self.model, configuration.data, jacp, jacr, point, spec["body_id"])
                rows.append(-hit.normal @ jacp)
                bounds.append(float(hit.signed_distance - spec["margin"]))
        return Constraint(
            G=np.asarray(rows, dtype=float).reshape((-1, self.model.nv)),
            h=np.asarray(bounds, dtype=float).reshape((-1,)),
        )
