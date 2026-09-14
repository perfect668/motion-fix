"""Terrain non-penetration limit for the independent WholeBody V5 solver.

The limit owns only terrain proxy discovery, signed-distance active-set
selection, and MuJoCo Jacobian linearization.  Scene-object collision remains
owned by the separate scene collision limit; keeping those contracts apart is
important for diagnostics and for future terrain backends.
"""

from __future__ import annotations

import mujoco as mj
import numpy as np
from mink.limits.limit import Constraint, Limit


class TerrainNonPenetrationLimit(Limit):
    """Adaptive terrain signed-distance inequality for robot surface proxies."""

    def __init__(self, model, terrain, config):
        self.model = model
        self.terrain = terrain
        self.config = dict(config or {})
        self.margin = float(self.config.get(
            "margin", self.config.get("default_margin", 0.004)
        ))
        self.activate = float(self.config.get("activate_distance", 0.06))
        self.deactivate = float(self.config.get("deactivate_distance", 0.09))
        self.deactivate_hold_steps = max(1, int(self.config.get("deactivate_hold_steps", 3)))
        self.prediction_horizon = max(0.0, float(self.config.get("prediction_horizon", 2.0)))
        self.contact_activation_score = float(self.config.get("contact_activation_score", 0.15))
        self.adaptive = bool(self.config.get("adaptive_activation", True))
        self.shells = self._discover()
        self.active = []
        self.active_indices: list[int] = []
        self.all_measurements: list[dict] = []
        self.previous_distances: dict[int, float] = {}
        self.active_mask: dict[int, bool] = {}
        self.deactivate_counter: dict[int, int] = {}
        self.contact_scores: dict[str, float] = {}

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
            primitive_points = []
            for geom_id in range(self.model.ngeom):
                if int(self.model.geom_bodyid[geom_id]) != body_id:
                    continue
                geom_rotation = np.zeros((3, 3), dtype=float)
                mj.mju_quat2Mat(geom_rotation.reshape(-1), self.model.geom_quat[geom_id])
                # MuJoCo exposes geom_type as a NumPy scalar in recent
                # releases.  Normalize it before comparing enum values so
                # primitive foot collision spheres are not silently omitted
                # from the terrain hard-constraint proxy set.
                geom_type = int(self.model.geom_type[geom_id])
                if geom_type == int(mj.mjtGeom.mjGEOM_MESH):
                    mesh_id = int(self.model.geom_dataid[geom_id])
                    start = int(self.model.mesh_vertadr[mesh_id])
                    count = int(self.model.mesh_vertnum[mesh_id])
                    verts.append(
                        np.asarray(self.model.mesh_vert[start : start + count]) @ geom_rotation.T
                        + self.model.geom_pos[geom_id]
                    )
                elif self.model.geom_contype[geom_id] and geom_type in (
                    int(mj.mjtGeom.mjGEOM_SPHERE),
                    int(mj.mjtGeom.mjGEOM_CAPSULE),
                ):
                    # A primitive collision geom has no mesh vertices.  Add
                    # its lowest local surface point so the terrain margin
                    # constrains the same physical foot/knee proxy used by
                    # MuJoCo scene collision.
                    radius = float(self.model.geom_size[geom_id, 0])
                    half_length = (
                        float(self.model.geom_size[geom_id, 1])
                        if geom_type == mj.mjtGeom.mjGEOM_CAPSULE
                        else 0.0
                    )
                    primitive_points.append(
                        (
                            np.asarray(self.model.geom_pos[geom_id], dtype=float)
                            + geom_rotation @ np.array([0.0, 0.0, -(half_length + radius)])
                        )[None, :]
                    )
            if verts or primitive_points:
                mesh_points = np.concatenate(verts) if verts else np.empty((0, 3))
                required = np.concatenate(primitive_points) if primitive_points else np.empty((0, 3))
                # Keep primitive support points at the front of the
                # deterministic sample stream; a stride over a dense visual
                # mesh must not discard the actual foot collision spheres.
                raw_shells.append((body_id, np.concatenate((required, mesh_points))))

        # The proxy budget is deterministic and distributed over every
        # articulated shell.  Terrain distance is still evaluated on the
        # actual terrain query surface, not on a world-Z approximation.
        budget = max(1, int(self.config.get("mesh_proxy_points", 96)))
        per_shell = max(4, int(np.ceil(budget / max(1, len(raw_shells)))))
        for body_id, samples in raw_shells:
            primitive_count = 0
            for geom_id in range(self.model.ngeom):
                if int(self.model.geom_bodyid[geom_id]) != body_id or not self.model.geom_contype[geom_id]:
                    continue
                if int(self.model.geom_type[geom_id]) in (
                    int(mj.mjtGeom.mjGEOM_SPHERE),
                    int(mj.mjtGeom.mjGEOM_CAPSULE),
                ):
                    primitive_count += 1
            # Sample visual mesh points independently of the primitive rows;
            # otherwise the stride can duplicate a sphere/capsule point and
            # make candidate indices depend on the shell's geometry density.
            primitive = samples[:primitive_count]
            visual = samples[primitive_count:]
            stride = max(1, int(np.ceil(max(len(visual), 1) / max(per_shell - primitive_count, 1))))
            selected_visual = visual[::stride][:max(per_shell - primitive_count, 0)]
            selected = np.concatenate((primitive, selected_visual))
            shells.append((body_id, selected))
        return shells

    def _world_points(self, configuration):
        configuration.update()
        for body_id, local in self.shells:
            rotation = configuration.data.xmat[body_id].reshape(3, 3)
            yield body_id, configuration.data.xpos[body_id] + local @ rotation.T

    def _region(self, body_id: int) -> str:
        name = (mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, int(body_id)) or "").upper()
        side = "left" if "_L_" in name or name.endswith("_L_LINK") else "right" if "_R_" in name or name.endswith("_R_LINK") else ""
        if side and "ANKLE" in name:
            return f"{side}_foot"
        if side and ("KNEE" in name or "SHIN" in name):
            return f"{side}_leg"
        return "body"

    def set_contact_scores(self, scores: dict[str, float] | None) -> None:
        self.contact_scores = {str(key): float(value) for key, value in (scores or {}).items()}

    def prepare_active_set(self, configuration, dt=0.0, contact_scores=None):
        if contact_scores is not None:
            self.set_contact_scores(contact_scores)
        active = []
        active_indices = []
        measurements = []
        point_index = 0
        for body_id, points in self._world_points(configuration):
            hits = self.terrain.nearest_surface_batch(points)
            for point, hit in zip(points, hits):
                distance = float(hit.signed_distance)
                previous = self.previous_distances.get(point_index)
                velocity = (
                    0.0
                    if previous is None or not np.isfinite(dt) or dt <= 0.0
                    else (distance - previous) / float(dt)
                )
                predicted = distance + self.prediction_horizon * max(float(dt), 0.0) * min(velocity, 0.0)
                region = self._region(body_id)
                if region == "left_foot":
                    contact_score = max(self.contact_scores.get("left_heel", 0.0), self.contact_scores.get("left_toe", 0.0))
                elif region == "right_foot":
                    contact_score = max(self.contact_scores.get("right_heel", 0.0), self.contact_scores.get("right_toe", 0.0))
                else:
                    contact_score = self.contact_scores.get(region, 0.0)
                forced = distance <= self.margin or predicted <= self.margin
                enter = distance <= self.activate or predicted <= self.activate or contact_score > self.contact_activation_score
                was_active = self.active_mask.get(point_index, False)
                if not self.adaptive:
                    is_active = True
                    self.deactivate_counter[point_index] = 0
                elif forced or enter:
                    is_active = True
                    self.deactivate_counter[point_index] = 0
                elif was_active and distance <= self.deactivate:
                    is_active = True
                    self.deactivate_counter[point_index] = 0
                elif was_active:
                    count = self.deactivate_counter.get(point_index, 0) + 1
                    self.deactivate_counter[point_index] = count
                    is_active = count < self.deactivate_hold_steps
                else:
                    is_active = False
                self.previous_distances[point_index] = distance
                self.active_mask[point_index] = is_active
                measurements.append({
                    "index": point_index,
                    "body_id": int(body_id),
                    "point": point.copy(),
                    "hit": hit,
                    "signed_distance": distance,
                    "margin": self.margin,
                    "slack": distance - self.margin,
                    "normal_velocity": velocity,
                    "predicted_distance": predicted,
                    "region": region,
                    "active": is_active,
                })
                if is_active:
                    active.append((body_id, point.copy(), hit))
                    active_indices.append(point_index)
                point_index += 1
        self.active = active
        self.active_indices = active_indices
        self.all_measurements = measurements

    def force_activate_violations(self, configuration) -> list[dict]:
        """Activate every missed proxy whose complete-set slack is negative."""
        self.prepare_active_set(configuration, 0.0)
        missed = [
            item for item in self.all_measurements
            if item["slack"] < 0.0 and not item["active"]
        ]
        if not missed:
            return []
        by_index = {item["index"]: item for item in self.all_measurements}
        for item in missed:
            index = int(item["index"])
            self.active_mask[index] = True
            self.deactivate_counter[index] = 0
            value = by_index[index]
            self.active.append((value["body_id"], value["point"].copy(), value["hit"]))
            self.active_indices.append(index)
        return missed

    def measure_all(self, configuration):
        """Measure every proxy for final validation and diagnostics."""
        values = []
        point_index = 0
        for body_id, points in self._world_points(configuration):
            for point, hit in zip(points, self.terrain.nearest_surface_batch(points)):
                values.append(
                    {
                        "body_id": int(body_id),
                        "index": point_index,
                        "point": point.copy(),
                        "signed_distance": float(hit.signed_distance),
                        "margin": float(self.margin),
                        "slack": float(hit.signed_distance - self.margin),
                        "surface_id": str(hit.surface_id),
                        "surface_normal": np.asarray(hit.normal).copy(),
                    }
                )
                point_index += 1
        return values

    def violating_measurements(self, configuration=None, tolerance: float = 0.0) -> list[dict]:
        """Return safety-margin violations from the complete proxy set.

        ``active`` is an optimization set and may intentionally omit distant
        proxies.  Final safety decisions must never depend on that set; this
        helper refreshes measurements when requested and always inspects all
        candidates.
        """
        if configuration is not None:
            self.prepare_active_set(configuration, 0.0)
        return [
            item for item in self.all_measurements
            if float(item.get("slack", 0.0)) < -float(tolerance)
        ]

    def compute_qp_inequalities(self, configuration, dt):
        del dt
        rows, bounds = [], []
        for body_id, point, hit in self.active:
            jacobian_position = np.zeros((3, self.model.nv))
            jacobian_rotation = np.zeros_like(jacobian_position)
            mj.mj_jac(
                self.model,
                configuration.data,
                jacobian_position,
                jacobian_rotation,
                point,
                body_id,
            )
            rows.append(-np.asarray(hit.normal, dtype=float) @ jacobian_position)
            bounds.append(float(hit.signed_distance - self.margin))
        # Keep the constraint matrix dimension explicit even for an empty
        # adaptive active set; Mink can then concatenate limits without
        # guessing the number of velocity variables.
        if not rows:
            return Constraint(
                G=np.empty((0, self.model.nv), dtype=float),
                h=np.empty((0,), dtype=float),
            )
        return Constraint(G=np.asarray(rows), h=np.asarray(bounds))


__all__ = ["TerrainNonPenetrationLimit"]
