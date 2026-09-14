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
        self.margin = float(self.config.get("margin", 0.004))
        self.activate = float(self.config.get("activate_distance", 0.06))
        self.shells = self._discover()
        self.active = []

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
            stride = max(1, int(np.ceil(len(samples) / per_shell)))
            selected = samples[::stride][:per_shell]
            # The first rows are primitive support points.  Retain them even
            # when the visual mesh requires a large stride.
            primitive_count = 0
            for geom_id in range(self.model.ngeom):
                if int(self.model.geom_bodyid[geom_id]) != body_id or not self.model.geom_contype[geom_id]:
                    continue
                if int(self.model.geom_type[geom_id]) in (
                    int(mj.mjtGeom.mjGEOM_SPHERE),
                    int(mj.mjtGeom.mjGEOM_CAPSULE),
                ):
                    primitive_count += 1
            if primitive_count:
                selected = np.concatenate((samples[:primitive_count], selected))
                selected = selected[:max(per_shell, primitive_count)]
            shells.append((body_id, selected))
        return shells

    def _world_points(self, configuration):
        configuration.update()
        for body_id, local in self.shells:
            rotation = configuration.data.xmat[body_id].reshape(3, 3)
            yield body_id, configuration.data.xpos[body_id] + local @ rotation.T

    def prepare_active_set(self, configuration, dt=0.0):
        del dt
        active = []
        for body_id, points in self._world_points(configuration):
            hits = self.terrain.nearest_surface_batch(points)
            for point, hit in zip(points, hits):
                if float(hit.signed_distance) <= self.activate:
                    active.append((body_id, point.copy(), hit))
        self.active = active

    def measure_all(self, configuration):
        """Measure every proxy for final validation and diagnostics."""
        values = []
        for body_id, points in self._world_points(configuration):
            for point, hit in zip(points, self.terrain.nearest_surface_batch(points)):
                values.append(
                    {
                        "body_id": int(body_id),
                        "point": point.copy(),
                        "signed_distance": float(hit.signed_distance),
                        "margin": float(self.margin),
                        "slack": float(hit.signed_distance - self.margin),
                        "surface_id": str(hit.surface_id),
                        "surface_normal": np.asarray(hit.normal).copy(),
                    }
                )
        return values

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
        return Constraint(G=np.asarray(rows), h=np.asarray(bounds)) if rows else Constraint()


__all__ = ["TerrainNonPenetrationLimit"]
