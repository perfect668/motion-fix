"""Format-oriented scene loading and deterministic triangle surface queries.

The V5 source-contact layer does not use MuJoCo collision distances as a
proxy for human contact.  It owns a lightweight triangle query instead.  The
query uses a KD-tree only to select candidate triangles; all returned closest
points, normals and support tests are evaluated on actual triangles.
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..core.schemas import FloorPolicy, PoseTrajectory, SceneAsset, SceneModel, SceneReference
from ..scene_asset_loader import load_scene_asset
from ..terrain_geometry import BoxPrimitive, TerrainField, TerrainSurfaceHit


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    length = np.linalg.norm(vector, axis=-1, keepdims=True)
    result = np.asarray(vector, dtype=float) / np.maximum(length, 1e-12)
    if fallback is not None:
        result = np.where(length > 1e-12, result, np.asarray(fallback, dtype=float))
    return result


def _closest_points_on_triangles(point: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Vectorized closest point routine from *Real-Time Collision Detection*."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    ap = point - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    closest = np.empty_like(a)
    mask_a = (d1 <= 0.0) & (d2 <= 0.0)
    closest[mask_a] = a[mask_a]
    bp = point - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    mask_b = (d3 >= 0.0) & (d4 <= d3)
    closest[mask_b] = b[mask_b]
    vc = d1 * d4 - d3 * d2
    mask_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0) & ~mask_a & ~mask_b
    v = d1 / np.maximum(d1 - d3, 1e-12)
    closest[mask_ab] = a[mask_ab] + v[mask_ab, None] * ab[mask_ab]
    cp = point - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    mask_c = (d6 >= 0.0) & (d5 <= d6)
    closest[mask_c] = c[mask_c]
    vb = d5 * d2 - d1 * d6
    mask_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0) & ~mask_a & ~mask_b & ~mask_ab & ~mask_c
    w = d2 / np.maximum(d2 - d6, 1e-12)
    closest[mask_ac] = a[mask_ac] + w[mask_ac, None] * ac[mask_ac]
    va = d3 * d6 - d5 * d4
    mask_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    mask_bc &= ~(mask_a | mask_b | mask_ab | mask_c | mask_ac)
    w = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-12)
    closest[mask_bc] = b[mask_bc] + w[mask_bc, None] * (c[mask_bc] - b[mask_bc])
    mask_face = ~(mask_a | mask_b | mask_ab | mask_c | mask_ac | mask_bc)
    denominator = 1.0 / np.maximum(va + vb + vc, 1e-12)
    v = vb * denominator
    w = vc * denominator
    closest[mask_face] = a[mask_face] + ab[mask_face] * v[mask_face, None] + ac[mask_face] * w[mask_face, None]
    return closest


def _barycentric(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    a, b, c = triangle
    v0, v1, v2 = b - a, c - a, np.asarray(point, dtype=float) - a
    d00, d01 = float(v0 @ v0), float(v0 @ v1)
    d11, d20, d21 = float(v1 @ v1), float(v2 @ v0), float(v2 @ v1)
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    return np.array([1.0 - v - w, v, w])


def _patch_ids(faces: np.ndarray, normals: np.ndarray, asset_id: str) -> np.ndarray:
    """Merge connected, approximately coplanar triangles into surface patches."""
    parent = np.arange(len(faces), dtype=int)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first

    edge_owner: dict[tuple[int, int], int] = {}
    for triangle_index, face in enumerate(faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(first), int(second))))
            other = edge_owner.get(edge)
            if other is None:
                edge_owner[edge] = triangle_index
            elif float(normals[other] @ normals[triangle_index]) >= 0.985:
                union(other, triangle_index)
    roots = np.asarray([find(index) for index in range(len(faces))], dtype=int)
    unique = {root: number for number, root in enumerate(sorted(set(roots.tolist())))}
    return np.asarray([f"{asset_id}:patch_{unique[root]:04d}" for root in roots], dtype=object)


class _TriangleBVH:
    """Deterministic AABB hierarchy for exact closest-triangle queries."""

    def __init__(self, triangles: np.ndarray, leaf_size: int = 64):
        self.triangles = triangles
        self.leaf_size = int(leaf_size)
        self.nodes = []
        self._build(np.arange(len(triangles), dtype=np.int64))

    def _build(self, indices: np.ndarray) -> int:
        triangles = self.triangles[indices]
        lower, upper = triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))
        node_index = len(self.nodes)
        self.nodes.append((lower, upper, None, None, None))
        if len(indices) <= self.leaf_size:
            self.nodes[node_index] = (lower, upper, indices.copy(), None, None)
            return node_index
        centroids = triangles.mean(axis=1)
        axis = int(np.argmax(upper - lower))
        order = np.argsort(centroids[:, axis], kind="stable")
        middle = max(1, len(indices) // 2)
        left = self._build(indices[order[:middle]])
        right = self._build(indices[order[middle:]])
        self.nodes[node_index] = (lower, upper, None, left, right)
        return node_index

    @staticmethod
    def _bbox_distance(point: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
        delta = np.maximum(np.maximum(lower - point, 0.0), point - upper)
        return float(delta @ delta)

    def nearest(self, point: np.ndarray) -> tuple[int, np.ndarray, float]:
        point = np.asarray(point, dtype=float).reshape(3)
        queue = [(self._bbox_distance(point, self.nodes[0][0], self.nodes[0][1]), 0)]
        best_index, best_point, best_distance = -1, None, np.inf
        while queue:
            lower_bound, node_index = heapq.heappop(queue)
            if lower_bound > best_distance + 1e-15:
                continue
            lower, upper, indices, left, right = self.nodes[node_index]
            if indices is not None:
                closest = _closest_points_on_triangles(point, self.triangles[indices])
                distances = np.sum((closest - point) ** 2, axis=1)
                local = int(np.argmin(distances))
                if float(distances[local]) < best_distance:
                    best_distance = float(distances[local])
                    best_index = int(indices[local])
                    best_point = closest[local].copy()
                continue
            for child in (left, right):
                if child is None:
                    continue
                child_node = self.nodes[child]
                bound = self._bbox_distance(point, child_node[0], child_node[1])
                if bound <= best_distance + 1e-15:
                    heapq.heappush(queue, (bound, child))
        if best_index < 0 or best_point is None:
            raise RuntimeError("Triangle BVH query returned no triangle")
        return best_index, best_point, best_distance


class MeshSceneField:
    """Triangle-accurate query for static meshes plus an optional floor."""

    is_mesh_scene = True

    def __init__(self, vertices, faces, asset_id: str, *, floor_z=0.0, support_normal_min_z=0.6, asset_inverse=None):
        self.vertices = np.asarray(vertices, dtype=float).reshape((-1, 3))
        self.faces = np.asarray(faces, dtype=np.int64).reshape((-1, 3))
        self.triangles = self.vertices[self.faces]
        normals = np.cross(self.triangles[:, 1] - self.triangles[:, 0], self.triangles[:, 2] - self.triangles[:, 0])
        self.normals = _unit(normals, np.array([0.0, 0.0, 1.0]))
        self.centroids = self.triangles.mean(axis=1)
        self.tree = cKDTree(self.centroids)
        self._xy_lower = self.triangles[:, :, :2].min(axis=1)
        self._xy_upper = self.triangles[:, :, :2].max(axis=1)
        self.patch_ids = _patch_ids(self.faces, self.normals, asset_id)
        self.floor_z = floor_z
        self.floor_id = "floor"
        self.support_normal_min_z = float(support_normal_min_z)
        self._support_mask = self.normals[:, 2] > self.support_normal_min_z
        self.boxes: list[BoxPrimitive] = []
        self.asset_id = asset_id
        self.asset_inverse = None if asset_inverse is None else np.asarray(asset_inverse, dtype=float).reshape(4, 4)
        # Small/medium meshes are queried exactly.  Large assets retain a
        # deterministic BVH-like centroid shortlist, but never use a fixed
        # 48-triangle approximation for ordinary props or stairs.
        self._candidate_count = len(self.triangles) if len(self.triangles) <= 8192 else min(512, len(self.triangles))
        self._bvh = _TriangleBVH(self.triangles) if len(self.triangles) > 8192 else None

    def _floor_hit(self, point: np.ndarray) -> TerrainSurfaceHit | None:
        if self.floor_z is None:
            return None
        return TerrainSurfaceHit(
            float(point[2] - self.floor_z),
            np.array([point[0], point[1], self.floor_z]),
            np.array([0.0, 0.0, 1.0]), self.floor_id, "floor", True,
        )

    def _candidate_indices(self, point: np.ndarray, count: int | None = None) -> np.ndarray:
        count = self._candidate_count if count is None else min(int(count), len(self.triangles))
        _, indices = self.tree.query(point, k=max(1, count))
        return np.atleast_1d(indices).astype(int)

    def _nearest_mesh_hit(self, point: np.ndarray) -> TerrainSurfaceHit:
        if self._bvh is not None:
            index, closest_point, _ = self._bvh.nearest(point)
        else:
            indices = self._candidate_indices(point)
            triangles = self.triangles[indices]
            closest = _closest_points_on_triangles(point, triangles)
            distances = np.linalg.norm(closest - point, axis=1)
            local = int(np.argmin(distances))
            index = int(indices[local])
            closest_point = closest[local]
        normal = self.normals[index].copy()
        # Meshes can be open or have inconsistent winding.  Orient the local
        # contact normal toward free space at the query point for a stable
        # normal residual, while collision signs remain MuJoCo's authority.
        if float((point - closest_point) @ normal) < 0.0:
            normal *= -1.0
        local_anchor = closest_point.copy()
        local_normal = normal.copy()
        if self.asset_inverse is not None:
            local_anchor = (np.r_[local_anchor, 1.0] @ self.asset_inverse.T)[:3]
            asset_linear = np.linalg.inv(self.asset_inverse[:3, :3])
            local_normal = _unit(asset_linear.T @ normal, np.array([0.0, 0.0, 1.0]))
        return TerrainSurfaceHit(
            float((point - closest_point) @ normal), closest_point, normal,
            str(self.patch_ids[index]), "mesh", bool(normal[2] > self.support_normal_min_z),
            triangle_id=index,
            # The exact branch returns ``closest_point`` directly from the
            # BVH; do not reference the centroid-shortlist locals here.  This
            # keeps large-mesh queries on the same barycentric contract as the
            # ordinary branch.
            barycentric=_barycentric(closest_point, self.triangles[index]),
            asset_local_anchor=local_anchor,
            asset_local_normal=local_normal,
        )

    def nearest_surface(self, point: np.ndarray) -> TerrainSurfaceHit:
        point = np.asarray(point, dtype=float).reshape(3)
        mesh = self._nearest_mesh_hit(point)
        floor = self._floor_hit(point)
        return mesh if floor is None or abs(mesh.signed_distance) <= abs(floor.signed_distance) else floor

    def nearest_contact_surface(self, point: np.ndarray) -> TerrainSurfaceHit:
        """Nearest geometric surface for desired-contact detection.

        Open prop meshes do not provide a globally reliable solid sign; use
        Euclidean proximity for contact evidence while retaining the oriented
        signed hit for diagnostics and collision validation.
        """
        point = np.asarray(point, dtype=float).reshape(3)
        mesh = self._nearest_mesh_hit(point)
        floor = self._floor_hit(point)
        if floor is None:
            return mesh
        mesh_distance = float(np.linalg.norm(point - mesh.closest_point))
        floor_distance = float(np.linalg.norm(point - floor.closest_point))
        return mesh if (mesh_distance, str(mesh.surface_id)) <= (floor_distance, str(floor.surface_id)) else floor

    def nearest_surface_batch(self, points: np.ndarray) -> list[TerrainSurfaceHit]:
        return [self.nearest_surface(point) for point in np.asarray(points, dtype=float).reshape((-1, 3))]

    def raycast(self, origin: np.ndarray, direction: np.ndarray, max_distance: float = np.inf):
        origin = np.asarray(origin, dtype=float).reshape(3)
        direction = np.asarray(direction, dtype=float).reshape(3)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        edge1 = self.triangles[:, 1] - self.triangles[:, 0]
        edge2 = self.triangles[:, 2] - self.triangles[:, 0]
        pvec = np.cross(direction[None, :], edge2)
        det = np.einsum("ij,ij->i", edge1, pvec)
        valid = np.abs(det) > 1e-10
        inv_det = np.zeros_like(det); inv_det[valid] = 1.0 / det[valid]
        tvec = origin[None, :] - self.triangles[:, 0]
        u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
        qvec = np.cross(tvec, edge1)
        v = np.einsum("ij,ij->i", direction[None, :], qvec) * inv_det
        distance = np.einsum("ij,ij->i", edge2, qvec) * inv_det
        mask = valid & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1.0 + 1e-8) & (distance >= -1e-8) & (distance <= float(max_distance))
        if not np.any(mask):
            return None
        index = int(np.flatnonzero(mask)[np.argmin(distance[mask])])
        point = origin + max(0.0, float(distance[index])) * direction
        normal = self.normals[index].copy()
        if float(normal @ direction) > 0.0:
            normal *= -1.0
        local_anchor = point.copy(); local_normal = normal.copy()
        if self.asset_inverse is not None:
            local_anchor = (np.r_[point, 1.0] @ self.asset_inverse.T)[:3]
            asset_linear = np.linalg.inv(self.asset_inverse[:3, :3])
            local_normal = _unit(asset_linear.T @ normal, np.array([0.0, 0.0, 1.0]))
        return TerrainSurfaceHit(float(distance[index]), point, normal, str(self.patch_ids[index]), "mesh", bool(normal[2] > self.support_normal_min_z), triangle_id=index, barycentric=_barycentric(point, self.triangles[index]), asset_local_anchor=local_anchor, asset_local_normal=local_normal)

    def support_surface(self, point: np.ndarray) -> TerrainSurfaceHit:
        point = np.asarray(point, dtype=float).reshape(3)
        # Exact vertical ray query.  A centroid shortlist is insufficient for
        # large triangles: a foot can be inside a triangle while its centroid
        # is metres away.  Filter all upward-facing triangle XY bounds in a
        # vectorized pass, then verify the projected point against the exact
        # triangle.
        tolerance = 2e-7
        indices = np.flatnonzero(
            self._support_mask
            & (self._xy_lower[:, 0] <= point[0] + tolerance)
            & (self._xy_upper[:, 0] >= point[0] - tolerance)
            & (self._xy_lower[:, 1] <= point[1] + tolerance)
            & (self._xy_upper[:, 1] >= point[1] - tolerance)
        )
        candidates: list[tuple[float, int, np.ndarray]] = []
        for index in indices:
            normal = self.normals[index]
            if normal[2] <= self.support_normal_min_z:
                continue
            triangle = self.triangles[index]
            plane_z = triangle[0, 2] - (normal[0] * (point[0] - triangle[0, 0]) + normal[1] * (point[1] - triangle[0, 1])) / normal[2]
            if plane_z > point[2] + 0.03:
                continue
            projected = np.array([point[0], point[1], plane_z])
            closest = _closest_points_on_triangles(projected, triangle[None])[0]
            if np.linalg.norm(closest - projected) <= 2e-4:
                candidates.append((float(point[2] - plane_z), int(index), projected))
        if candidates:
            _, index, projected = min(candidates, key=lambda item: (item[0], str(self.patch_ids[item[1]])))
            local_anchor = projected.copy()
            local_normal = self.normals[index].copy()
            if self.asset_inverse is not None:
                local_anchor = (np.r_[local_anchor, 1.0] @ self.asset_inverse.T)[:3]
                asset_linear = np.linalg.inv(self.asset_inverse[:3, :3])
                local_normal = _unit(asset_linear.T @ self.normals[index], np.array([0.0, 0.0, 1.0]))
            hit = TerrainSurfaceHit(
                float(point[2] - projected[2]), projected, self.normals[index].copy(),
                str(self.patch_ids[index]), "mesh", True,
                triangle_id=index,
                barycentric=_barycentric(projected, self.triangles[index]),
                asset_local_anchor=local_anchor,
                asset_local_normal=local_normal,
            )
        else:
            hit = self._nearest_mesh_hit(point)
            if not hit.supportable and self._floor_hit(point) is not None:
                floor_hit = self._floor_hit(point)
                if floor_hit is not None and floor_hit.closest_point[2] <= point[2] + 0.03:
                    return floor_hit
        floor = self._floor_hit(point)
        if floor is not None and floor.closest_point[2] <= point[2] + 0.03:
            # Highest support under the foot wins; deterministic tie uses the
            # surface identifier.
            if not hit.supportable or floor.closest_point[2] > hit.closest_point[2] + 1e-9:
                return floor
        return hit

    def support_surface_batch(self, points: np.ndarray) -> list[TerrainSurfaceHit]:
        return [self.support_surface(point) for point in np.asarray(points, dtype=float).reshape((-1, 3))]

    def to_spec(self) -> dict:
        return {
            "floor_z": self.floor_z,
            "floor_id": self.floor_id,
            "surface_type": "triangle_mesh",
            "asset_id": self.asset_id,
            "triangle_count": int(len(self.triangles)),
            "surface_patch_count": int(len(set(self.patch_ids.tolist()))),
        }


class CompositeSceneField:
    """Deterministic union of independently loaded scene surface fields."""

    is_mesh_scene = False

    def __init__(self, fields):
        self.fields = tuple(fields)
        if not self.fields:
            raise ValueError("CompositeSceneField requires at least one field")
        floors = [field.floor_z for field in self.fields if getattr(field, "floor_z", None) is not None]
        self.floor_z = min(floors) if floors else None
        self.floor_id = "floor"
        self.boxes = [box for field in self.fields for box in getattr(field, "boxes", [])]

    def nearest_surface(self, point):
        return min((field.nearest_surface(point) for field in self.fields),
                   key=lambda hit: (float(hit.signed_distance), str(hit.surface_id)))

    def nearest_contact_surface(self, point):
        candidates = []
        for field in self.fields:
            query = getattr(field, "nearest_contact_surface", field.nearest_surface)
            candidates.append(query(point))
        return min(candidates, key=lambda hit: (abs(float(hit.signed_distance)), str(hit.surface_id)))

    def nearest_surface_batch(self, points):
        return [self.nearest_surface(point) for point in np.asarray(points, dtype=float).reshape((-1, 3))]

    def support_surface(self, point):
        candidates = []
        for field in self.fields:
            hit = field.support_surface(point)
            if hit.supportable:
                candidates.append(hit)
        if not candidates:
            return self.nearest_surface(point)
        # Highest support below the query wins; ID breaks ties at edges.
        return min(candidates, key=lambda hit: (-float(hit.closest_point[2]), str(hit.surface_id)))

    def support_surface_batch(self, points):
        return [self.support_surface(point) for point in np.asarray(points, dtype=float).reshape((-1, 3))]

    def raycast(self, origin, direction, max_distance=np.inf):
        candidates = []
        for field in self.fields:
            hit = getattr(field, "raycast", lambda *_args, **_kwargs: None)(origin, direction, max_distance)
            if hit is not None:
                candidates.append((float(np.linalg.norm(np.asarray(hit.closest_point) - np.asarray(origin))), hit))
        return min(candidates, key=lambda item: (item[0], str(item[1].surface_id)))[1] if candidates else None

    def to_spec(self):
        return {"surface_type": "composite", "field_count": len(self.fields),
                "fields": [field.to_spec() for field in self.fields]}


class DynamicSceneField:
    """Time-indexed query assembled from asset-local triangle meshes."""

    is_mesh_scene = True

    def __init__(self, assets, *, floor_z=0.0, support_normal_min_z=0.6):
        self.assets = tuple(assets)
        if not self.assets:
            raise ValueError("DynamicSceneField requires assets")
        self.floor_z = floor_z
        self.floor_id = "floor"
        self.support_normal_min_z = float(support_normal_min_z)
        self.boxes = []
        self._time = None
        self._field = None
        self._poses = None

    def set_time(self, timestamp: float) -> None:
        timestamp = float(timestamp)
        poses = tuple(np.asarray(asset.pose_at(timestamp), dtype=float) for asset, _ in self.assets)
        if self._poses is not None and all(np.allclose(current, previous, atol=1e-10) for current, previous in zip(poses, self._poses)):
            # A dynamic trajectory may repeat a pose for many frames.  Keep
            # the exact triangle/BVH field instead of rebuilding it every
            # time; querying still uses the current timestamp transparently.
            self._time = timestamp
            return
        fields = []
        for (asset, loaded), pose in zip(self.assets, poses):
            vertices = (np.c_[loaded.vertices, np.ones(len(loaded.vertices))] @ pose.T)[:, :3]
            fields.append(MeshSceneField(
                vertices, loaded.faces, asset.asset_id, floor_z=None,
                support_normal_min_z=self.support_normal_min_z,
                asset_inverse=np.linalg.inv(pose),
            ))
        if self.floor_z is not None:
            fields.append(TerrainField(floor_z=self.floor_z))
        self._field = fields[0] if len(fields) == 1 else CompositeSceneField(fields)
        self._time = timestamp
        self._poses = poses

    def _current(self):
        if self._field is None:
            self.set_time(0.0)
        return self._field

    def nearest_surface(self, point):
        return self._current().nearest_surface(point)

    def nearest_contact_surface(self, point):
        field = self._current()
        query = getattr(field, "nearest_contact_surface", field.nearest_surface)
        return query(point)

    def nearest_surface_batch(self, points):
        return self._current().nearest_surface_batch(points)

    def support_surface(self, point):
        return self._current().support_surface(point)

    def support_surface_batch(self, points):
        return self._current().support_surface_batch(points)

    def raycast(self, origin, direction, max_distance=np.inf):
        return self._current().raycast(origin, direction, max_distance)

    def to_spec(self):
        return {"surface_type": "dynamic", "asset_ids": [asset.asset_id for asset, _ in self.assets],
                "floor_z": self.floor_z}


def load_scene_model(reference: SceneReference | None, *, floor_policy: FloorPolicy = FloorPolicy.AUTO, floor_height: float | None = 0.0) -> SceneModel:
    if reference is None:
        return SceneModel(floor_policy=floor_policy, floor_height=floor_height)
    if reference.path is None or not reference.path.is_file():
        raise FileNotFoundError(f"Scene asset does not exist: {reference.path}")
    specs = reference.metadata.get("assets") if isinstance(reference.metadata, dict) else None
    if not specs:
        specs = [{"asset_id": reference.asset_id, "path": str(reference.path),
                  "pose": reference.pose if reference.pose is not None else np.eye(4),
                  "pose_trajectory": reference.pose_trajectory,
                  "unit_scale": reference.unit_scale, **reference.metadata}]
    assets = []
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict) or not spec.get("path"):
            raise ValueError(f"Scene asset specification {index} lacks path")
        asset_path = Path(spec["path"]).expanduser()
        if not asset_path.is_absolute():
            asset_path = reference.path.parent / asset_path
        trajectory = spec.get("pose_trajectory")
        if isinstance(trajectory, dict):
            trajectory = PoseTrajectory(
                trajectory["timestamps"], trajectory["translations"], trajectory.get("rotations_wxyz")
            )
        assets.append(SceneAsset(
            str(spec.get("asset_id", f"{reference.asset_id}_{index}")), asset_path,
            static=bool(spec.get("static", True)),
            collision_enabled=bool(spec.get("collision_enabled", True)),
            pose=np.asarray(spec.get("pose") if spec.get("pose") is not None else np.eye(4), dtype=float),
            pose_trajectory=trajectory,
            unit_scale=spec.get("unit_scale"), metadata=dict(spec),
        ))
    return SceneModel(
        assets=tuple(assets),
        floor_policy=floor_policy, floor_height=floor_height,
        metadata={"format": reference.format},
    )


def terrain_from_scene(scene: SceneModel) -> TerrainField:
    floor = scene.floor_height if scene.floor_policy != FloorPolicy.DISABLED else None
    boxes: list[BoxPrimitive] = []
    mesh_assets: list[tuple[SceneAsset, object]] = []
    for asset in scene.assets:
        if asset.path.suffix.lower() == ".json":
            spec = json.loads(asset.path.read_text(encoding="utf-8"))
            for index, item in enumerate(spec.get("primitives", spec.get("boxes", []))):
                boxes.append(BoxPrimitive(str(item.get("surface_id", f"{asset.asset_id}:{index}")), item["center"], item["half_extents"], item.get("rotation", np.eye(3)), item.get("type", "box")))
        elif asset.path.suffix.lower() == ".urdf":
            primitive = TerrainField.from_file(asset.path, floor_z=None)
            boxes.extend(primitive.boxes)
        elif asset.path.suffix.lower() in {".usd", ".usda", ".usdc", ".obj"}:
            loaded = load_scene_asset(asset.path, {"object_id": asset.asset_id, "sample_count": 2048})
            mesh_assets.append((asset, loaded))
        else:
            raise ValueError(f"Unsupported scene loader format: {asset.path.suffix}")
    if mesh_assets:
        if any(asset.pose_trajectory is not None for asset, _ in mesh_assets):
            return DynamicSceneField(mesh_assets, floor_z=floor, support_normal_min_z=0.6)
        fields = []
        for asset, loaded in mesh_assets:
            pose = np.asarray(asset.pose, dtype=float)
            vertices = (np.c_[loaded.vertices, np.ones(len(loaded.vertices))] @ pose.T)[:, :3]
            fields.append(MeshSceneField(
                vertices, loaded.faces, asset.asset_id, floor_z=None,
                asset_inverse=np.linalg.inv(np.asarray(asset.pose, dtype=float)),
            ))
        if floor is not None:
            fields.append(TerrainField(floor_z=floor))
        return fields[0] if len(fields) == 1 else CompositeSceneField(fields)
    return TerrainField(boxes, floor_z=floor)
