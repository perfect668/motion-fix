"""Geometry-agnostic scene objects used by WholeBody Omni retargeting.

Visual/interactions samples are deliberately independent from collision
representations.  Collision pieces may be primitives or pre-decomposed convex
meshes and all pieces retain the same logical object id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class SceneObject:
    object_id: str
    surface_samples: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    collision_representation: list[dict[str, Any]] = field(default_factory=list)
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    surface_groups: dict[str, np.ndarray] = field(default_factory=dict)
    supportable_groups: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.surface_samples = np.asarray(self.surface_samples, dtype=float).reshape((-1, 3))
        self.pose = np.asarray(self.pose, dtype=float).reshape(4, 4)
        if not np.all(np.isfinite(self.pose)) or not np.allclose(self.pose[3], [0, 0, 0, 1]):
            raise ValueError("SceneObject.pose must be a finite homogeneous transform")

    def transformed_samples(self) -> np.ndarray:
        if len(self.surface_samples) == 0:
            return self.surface_samples.copy()
        homogeneous = np.c_[self.surface_samples, np.ones(len(self.surface_samples))]
        return (homogeneous @ self.pose.T)[:, :3]

    def update_pose(self, pose: np.ndarray) -> None:
        pose = np.asarray(pose, dtype=float).reshape(4, 4)
        if not np.all(np.isfinite(pose)):
            raise ValueError("SceneObject pose contains NaN or Inf")
        self.pose = pose


@dataclass
class SceneGeometry:
    objects: list[SceneObject] = field(default_factory=list)
    floor_z: float | None = 0.0

    def __post_init__(self) -> None:
        ids = [obj.object_id for obj in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate scene object ids: {ids}")

    def interaction_samples(self, max_points: int = 256) -> np.ndarray:
        points = [obj.transformed_samples() for obj in self.objects if len(obj.surface_samples)]
        if self.floor_z is not None:
            points.append(np.array([[0.0, 0.0, float(self.floor_z)]]))
        if not points:
            return np.empty((0, 3))
        merged = np.concatenate(points)
        if len(merged) <= max_points:
            return merged
        # Deterministic farthest-point downsampling keeps object outlines.
        chosen = [int(np.argmax(np.sum((merged - merged.mean(0)) ** 2, axis=1)))]
        nearest = np.sum((merged - merged[chosen[0]]) ** 2, axis=1)
        for _ in range(1, max_points):
            index = int(np.argmax(nearest))
            chosen.append(index)
            nearest = np.minimum(nearest, np.sum((merged - merged[index]) ** 2, axis=1))
        return merged[np.asarray(chosen)]

    def to_spec(self) -> dict[str, Any]:
        return {
            "floor_z": self.floor_z,
            "objects": [
                {
                    "object_id": obj.object_id,
                    "surface_samples": obj.surface_samples.tolist(),
                    "collision_representation": obj.collision_representation,
                    "pose": obj.pose.tolist(),
                    "surface_groups": {k: np.asarray(v).tolist() for k, v in obj.surface_groups.items()},
                    "supportable_groups": sorted(obj.supportable_groups),
                }
                for obj in self.objects
            ],
        }

    @classmethod
    def from_mesh_file(
        cls,
        path: str | Path,
        object_id: str,
        *,
        sample_count: int = 128,
        pose: np.ndarray | None = None,
        collision_representation: list[dict[str, Any]] | None = None,
    ) -> "SceneGeometry":
        """Load a visual mesh through trimesh for interaction sampling.

        Collision pieces remain an explicit input; no automatic convex
        decomposition is performed during retargeting.
        """
        try:
            import trimesh
        except ImportError as error:
            raise RuntimeError("Mesh scene loading requires the optional trimesh package") from error
        mesh = trimesh.load_mesh(str(path), process=False)
        if hasattr(mesh, "geometry"):
            meshes = list(mesh.geometry.values())
            vertices = np.concatenate([np.asarray(item.vertices) for item in meshes])
            faces = []
            offset = 0
            for item in meshes:
                faces.append(np.asarray(item.faces) + offset)
                offset += len(item.vertices)
            faces = np.concatenate(faces)
        else:
            vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        samples = sample_mesh_surface(vertices, faces, sample_count)
        return cls([SceneObject(
            object_id=object_id,
            surface_samples=samples,
            collision_representation=list(collision_representation or []),
            pose=np.eye(4) if pose is None else pose,
        )])


def sample_mesh_surface(vertices: np.ndarray, faces: np.ndarray, count: int = 128) -> np.ndarray:
    """Deterministically sample triangle surfaces proportional to area."""
    vertices = np.asarray(vertices, dtype=float).reshape((-1, 3))
    faces = np.asarray(faces, dtype=int).reshape((-1, 3))
    if len(vertices) == 0 or len(faces) == 0 or count <= 0:
        return np.empty((0, 3))
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    valid = areas > 1e-12
    triangles, areas = triangles[valid], areas[valid]
    if len(triangles) == 0:
        return vertices[: min(len(vertices), count)].copy()
    quotas = np.maximum(1, np.floor(count * areas / areas.sum()).astype(int))
    while quotas.sum() < count:
        quotas[int(np.argmax(areas / quotas))] += 1
    samples = []
    for tri, quota in zip(triangles, quotas):
        for step in range(int(quota)):
            u = (step + 0.5) / quota
            v = ((step * 0.61803398875) % quota + 0.5) / quota
            if u + v > 1.0:
                u, v = 1.0 - u, 1.0 - v
            samples.append(tri[0] + u * (tri[1] - tri[0]) + v * (tri[2] - tri[0]))
    return np.asarray(samples[:count], dtype=float)
