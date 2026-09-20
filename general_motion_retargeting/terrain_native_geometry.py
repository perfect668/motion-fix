"""Support-patch extraction and terrain interaction sampling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _unit(value: np.ndarray, fallback=(0.0, 0.0, 1.0)) -> np.ndarray:
    value = np.asarray(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        return np.asarray(fallback, dtype=float)
    return value / norm


def _point_in_triangle_xy(point_xy: np.ndarray, triangle: np.ndarray, tolerance: float = 1e-8) -> bool:
    a, b, c = np.asarray(triangle, dtype=float)[:, :2]
    v0, v1, v2 = c - a, b - a, np.asarray(point_xy, dtype=float) - a
    den = float(v0[0] * v1[1] - v1[0] * v0[1])
    if abs(den) < 1e-12:
        return False
    u = float((v2[0] * v1[1] - v1[0] * v2[1]) / den)
    v = float((v0[0] * v2[1] - v2[0] * v0[1]) / den)
    return u >= -tolerance and v >= -tolerance and u + v <= 1.0 + tolerance


@dataclass(frozen=True)
class SupportPatch:
    patch_id: str
    face_indices: np.ndarray
    triangles: np.ndarray
    normal: np.ndarray
    center: np.ndarray
    xy_min: np.ndarray
    xy_max: np.ndarray

    def plane_point(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).reshape(2)
        normal = _unit(self.normal)
        if abs(float(normal[2])) < 1e-8:
            return self.center.copy()
        z = float(self.center[2] - (
            normal[0] * (xy[0] - self.center[0])
            + normal[1] * (xy[1] - self.center[1])
        ) / normal[2])
        return np.array([xy[0], xy[1], z], dtype=float)

    def contains_xy(self, xy: np.ndarray, edge_margin: float = 0.0) -> bool:
        xy = np.asarray(xy, dtype=float).reshape(2)
        if np.any(xy < self.xy_min - edge_margin) or np.any(xy > self.xy_max + edge_margin):
            return False
        if any(_point_in_triangle_xy(xy, tri) for tri in self.triangles):
            return True
        if edge_margin <= 0.0:
            return False
        for tri in self.triangles:
            for a, b in zip(tri[:, :2], np.roll(tri[:, :2], -1, axis=0)):
                ab = b - a
                t = float(np.clip(((xy - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0))
                if np.linalg.norm(xy - (a + t * ab)) <= edge_margin:
                    return True
        return False


class TerrainPatchMap:
    """Connected upward-facing terrain patches in solver/world coordinates."""

    def __init__(self, object_id: str, vertices: np.ndarray, faces: np.ndarray, patches: list[SupportPatch]):
        self.object_id = str(object_id)
        self.vertices = np.asarray(vertices, dtype=float).reshape((-1, 3))
        self.faces = np.asarray(faces, dtype=int).reshape((-1, 3))
        self.triangles = self.vertices[self.faces]
        self.patches = list(patches)
        self.by_id = {patch.patch_id: patch for patch in self.patches}

    @classmethod
    def from_mesh(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        object_pose: np.ndarray,
        object_id: str,
        config: dict[str, Any] | None = None,
    ) -> "TerrainPatchMap":
        cfg = config or {}
        support_normal_min_z = float(cfg.get("support_normal_min_z", 0.65))
        normal_angle_deg = float(cfg.get("patch_normal_angle_deg", 8.0))
        plane_tolerance = float(cfg.get("patch_plane_tolerance", 0.008))
        pose = np.asarray(object_pose, dtype=float).reshape(4, 4)
        local = np.asarray(vertices, dtype=float).reshape((-1, 3))
        world = (np.c_[local, np.ones(len(local))] @ pose.T)[:, :3]
        faces = np.asarray(faces, dtype=int).reshape((-1, 3))
        triangles = world[faces]
        raw_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        raw_normals /= np.maximum(np.linalg.norm(raw_normals, axis=1, keepdims=True), 1e-12)
        normals = raw_normals.copy()
        normals[normals[:, 2] < 0.0] *= -1.0
        support = np.flatnonzero(normals[:, 2] >= support_normal_min_z)

        parent = {int(index): int(index) for index in support}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        edge_to_faces: dict[tuple[int, int], list[int]] = {}
        support_set = set(int(i) for i in support)
        for face_index in support:
            face = faces[int(face_index)]
            for u, v in zip(face, np.roll(face, -1)):
                edge_to_faces.setdefault(tuple(sorted((int(u), int(v)))), []).append(int(face_index))

        cos_limit = float(np.cos(np.deg2rad(normal_angle_deg)))
        centers = triangles.mean(axis=1)
        for connected in edge_to_faces.values():
            connected = [i for i in connected if i in support_set]
            for i in range(len(connected)):
                for j in range(i + 1, len(connected)):
                    a, b = connected[i], connected[j]
                    if float(normals[a] @ normals[b]) < cos_limit:
                        continue
                    delta = centers[b] - centers[a]
                    if max(abs(float(normals[a] @ delta)), abs(float(normals[b] @ delta))) > plane_tolerance:
                        continue
                    union(a, b)

        groups: dict[int, list[int]] = {}
        for index in support:
            groups.setdefault(find(int(index)), []).append(int(index))

        patches: list[SupportPatch] = []
        for ordinal, indices in enumerate(sorted(groups.values(), key=lambda g: (float(centers[g, 2].mean()), min(g)))):
            tris = triangles[np.asarray(indices, dtype=int)]
            normal = _unit(normals[np.asarray(indices, dtype=int)].mean(axis=0))
            center = tris.reshape(-1, 3).mean(axis=0)
            xy = tris[:, :, :2].reshape(-1, 2)
            patches.append(SupportPatch(
                patch_id=f"{object_id}:support_{ordinal:03d}",
                face_indices=np.asarray(indices, dtype=int),
                triangles=tris.copy(),
                normal=normal,
                center=center,
                xy_min=xy.min(axis=0),
                xy_max=xy.max(axis=0),
            ))
        if not patches:
            raise ValueError("Terrain-native retargeting found no upward support patches in scene mesh")
        return cls(object_id, world, faces, patches)

    def add_horizontal_patch(
        self,
        patch_id: str,
        z: float,
        xy_min: np.ndarray,
        xy_max: np.ndarray,
    ) -> None:
        """Add an analytic floor/support plane to the same semantic patch map."""
        lo = np.asarray(xy_min, dtype=float).reshape(2)
        hi = np.asarray(xy_max, dtype=float).reshape(2)
        corners = np.array([
            [lo[0], lo[1], z], [hi[0], lo[1], z],
            [hi[0], hi[1], z], [lo[0], hi[1], z],
        ], dtype=float)
        triangles = corners[np.array([[0, 1, 2], [0, 2, 3]], dtype=int)]
        patch = SupportPatch(
            patch_id=str(patch_id),
            face_indices=np.empty(0, dtype=int),
            triangles=triangles,
            normal=np.array([0.0, 0.0, 1.0]),
            center=corners.mean(axis=0),
            xy_min=lo,
            xy_max=hi,
        )
        self.patches.append(patch)
        self.by_id[patch.patch_id] = patch

    def support_at(
        self,
        point: np.ndarray,
        *,
        edge_margin: float = 0.02,
        above_tolerance: float = 0.03,
        max_gap: float | None = None,
    ) -> tuple[SupportPatch, np.ndarray, float] | None:
        point = np.asarray(point, dtype=float).reshape(3)
        candidates = []
        for patch in self.patches:
            if not patch.contains_xy(point[:2], edge_margin=edge_margin):
                continue
            surface = patch.plane_point(point[:2])
            gap = float(patch.normal @ (point - surface))
            if gap < -above_tolerance:
                continue
            if max_gap is not None and gap > max_gap:
                continue
            candidates.append((float(surface[2]), patch.patch_id, patch, surface, gap))
        if not candidates:
            return None
        _, _, patch, surface, gap = max(candidates, key=lambda item: (item[0], item[1]))
        return patch, surface, gap

    def interaction_samples(self, references: np.ndarray, count: int) -> np.ndarray:
        """Bias interaction samples toward support patches, then nearby full geometry."""
        count = max(0, int(count))
        if count == 0:
            return np.empty((0, 3), dtype=float)
        refs = np.asarray(references, dtype=float).reshape((-1, 3))
        support_candidates = []
        for patch in self.patches:
            if patch.patch_id == "floor":
                continue
            support_candidates.append(patch.center)
            support_candidates.extend(patch.triangles.reshape(-1, 3))
        support = np.unique(np.round(np.asarray(support_candidates), 9), axis=0)
        general = np.unique(np.round(self.triangles.reshape(-1, 3), 9), axis=0)

        def nearest_order(points: np.ndarray) -> np.ndarray:
            if len(points) == 0:
                return np.empty(0, dtype=int)
            if len(refs) == 0:
                return np.arange(len(points), dtype=int)
            d2 = np.min(np.sum((points[:, None, :] - refs[None, :, :]) ** 2, axis=-1), axis=1)
            return np.argsort(d2, kind="stable")

        support_n = min(len(support), max(1, int(round(count * 0.75))))
        selected = list(support[nearest_order(support)[:support_n]])
        remaining = count - len(selected)
        if remaining > 0:
            selected.extend(general[nearest_order(general)[:remaining]])
        return np.asarray(selected[:count], dtype=float).reshape((-1, 3))

    def summary(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "patch_count": len(self.patches),
            "patches": [
                {
                    "patch_id": patch.patch_id,
                    "face_count": int(len(patch.face_indices)),
                    "normal": patch.normal.tolist(),
                    "center": patch.center.tolist(),
                    "xy_min": patch.xy_min.tolist(),
                    "xy_max": patch.xy_max.tolist(),
                }
                for patch in self.patches
            ],
        }
