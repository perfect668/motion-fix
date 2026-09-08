"""Format-oriented scene loading and deterministic triangle surface queries.

The V5 source-contact layer does not use MuJoCo collision distances as a
proxy for human contact.  It owns a lightweight triangle query instead.  The
query uses a KD-tree only to select candidate triangles; all returned closest
points, normals and support tests are evaluated on actual triangles.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..core.schemas import FloorPolicy, SceneAsset, SceneModel, SceneReference
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


class MeshSceneField:
    """Triangle-accurate query for static meshes plus an optional floor."""

    is_mesh_scene = True

    def __init__(self, vertices, faces, asset_id: str, *, floor_z=0.0, support_normal_min_z=0.6):
        self.vertices = np.asarray(vertices, dtype=float).reshape((-1, 3))
        self.faces = np.asarray(faces, dtype=np.int64).reshape((-1, 3))
        self.triangles = self.vertices[self.faces]
        normals = np.cross(self.triangles[:, 1] - self.triangles[:, 0], self.triangles[:, 2] - self.triangles[:, 0])
        self.normals = _unit(normals, np.array([0.0, 0.0, 1.0]))
        self.centroids = self.triangles.mean(axis=1)
        self.tree = cKDTree(self.centroids)
        self.patch_ids = _patch_ids(self.faces, self.normals, asset_id)
        self.floor_z = floor_z
        self.floor_id = "floor"
        self.support_normal_min_z = float(support_normal_min_z)
        self.boxes: list[BoxPrimitive] = []
        self.asset_id = asset_id
        self._candidate_count = min(48, len(self.triangles))

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
        indices = self._candidate_indices(point)
        triangles = self.triangles[indices]
        closest = _closest_points_on_triangles(point, triangles)
        distances = np.linalg.norm(closest - point, axis=1)
        local = int(np.argmin(distances))
        index = int(indices[local])
        normal = self.normals[index].copy()
        # Meshes can be open or have inconsistent winding.  Orient the local
        # contact normal toward free space at the query point for a stable
        # normal residual, while collision signs remain MuJoCo's authority.
        if float((point - closest[local]) @ normal) < 0.0:
            normal *= -1.0
        return TerrainSurfaceHit(
            float((point - closest[local]) @ normal), closest[local], normal,
            str(self.patch_ids[index]), "mesh", bool(normal[2] > self.support_normal_min_z),
        )

    def nearest_surface(self, point: np.ndarray) -> TerrainSurfaceHit:
        point = np.asarray(point, dtype=float).reshape(3)
        mesh = self._nearest_mesh_hit(point)
        floor = self._floor_hit(point)
        return mesh if floor is None or abs(mesh.signed_distance) <= abs(floor.signed_distance) else floor

    def nearest_surface_batch(self, points: np.ndarray) -> list[TerrainSurfaceHit]:
        return [self.nearest_surface(point) for point in np.asarray(points, dtype=float).reshape((-1, 3))]

    def support_surface(self, point: np.ndarray) -> TerrainSurfaceHit:
        point = np.asarray(point, dtype=float).reshape(3)
        # Search a wide local neighbourhood for upward triangles below the
        # foot.  Vertical projection must land inside the finite triangle; a
        # side wall is never a valid support surface.
        indices = self._candidate_indices(point, min(128, len(self.triangles)))
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
            hit = TerrainSurfaceHit(
                float(point[2] - projected[2]), projected, self.normals[index].copy(),
                str(self.patch_ids[index]), "mesh", True,
            )
        else:
            hit = self._nearest_mesh_hit(point)
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


def load_scene_model(reference: SceneReference | None, *, floor_policy: FloorPolicy = FloorPolicy.AUTO, floor_height: float | None = 0.0) -> SceneModel:
    if reference is None:
        return SceneModel(floor_policy=floor_policy, floor_height=floor_height)
    if reference.path is None or not reference.path.is_file():
        raise FileNotFoundError(f"Scene asset does not exist: {reference.path}")
    return SceneModel(
        assets=(SceneAsset(reference.asset_id, reference.path, pose=reference.pose if reference.pose is not None else np.eye(4), unit_scale=reference.unit_scale, metadata=reference.metadata),),
        floor_policy=floor_policy, floor_height=floor_height,
        metadata={"format": reference.format},
    )


def terrain_from_scene(scene: SceneModel) -> TerrainField:
    floor = scene.floor_height if scene.floor_policy != FloorPolicy.DISABLED else None
    boxes: list[BoxPrimitive] = []
    mesh_assets: list[tuple[np.ndarray, np.ndarray, str]] = []
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
            pose = np.asarray(asset.pose, dtype=float)
            vertices = (np.c_[loaded.vertices, np.ones(len(loaded.vertices))] @ pose.T)[:, :3]
            mesh_assets.append((vertices, loaded.faces, asset.asset_id))
        else:
            raise ValueError(f"Unsupported scene loader format: {asset.path.suffix}")
    if mesh_assets:
        if len(mesh_assets) != 1:
            raise ValueError("V5 mesh query currently requires one merged scene asset")
        vertices, faces, asset_id = mesh_assets[0]
        return MeshSceneField(vertices, faces, asset_id, floor_z=floor)
    return TerrainField(boxes, floor_z=floor)
