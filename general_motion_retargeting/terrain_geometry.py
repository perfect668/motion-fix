"""Deterministic terrain primitives and shared source-to-solver transforms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero vector")
    return vector / norm


@dataclass(frozen=True)
class SceneTransform:
    rotation: np.ndarray
    scale: float
    translation: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        translation = np.asarray(self.translation, dtype=float).reshape(3)
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("SceneTransform.rotation must be orthonormal")
        if np.linalg.det(rotation) < 0.999:
            raise ValueError("SceneTransform.rotation must be a proper rotation")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("SceneTransform.scale must be positive")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "scale", float(self.scale))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return self.scale * np.einsum("ij,...j->...i", self.rotation, points) + self.translation

    def transform_normals(self, normals: np.ndarray) -> np.ndarray:
        normals = np.einsum("ij,...j->...i", self.rotation, np.asarray(normals, dtype=float))
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals / np.maximum(norms, 1e-12)

    def transform_box(self, box: "BoxPrimitive") -> "BoxPrimitive":
        return BoxPrimitive(
            surface_id=box.surface_id,
            center=self.transform_points(box.center),
            half_extents=self.scale * box.half_extents,
            rotation=self.rotation @ box.rotation,
            surface_type=box.surface_type,
        )

    def inverse(self) -> "SceneTransform":
        inverse_rotation = self.rotation.T
        inverse_scale = 1.0 / self.scale
        return SceneTransform(
            rotation=inverse_rotation,
            scale=inverse_scale,
            translation=-inverse_scale * (inverse_rotation @ self.translation),
        )

    def to_dict(self) -> dict:
        return {
            "rotation": self.rotation.tolist(),
            "scale": self.scale,
            "translation": self.translation.tolist(),
        }


@dataclass(frozen=True)
class TerrainSurfaceHit:
    signed_distance: float
    closest_point: np.ndarray
    normal: np.ndarray
    surface_id: str
    surface_type: str
    supportable: bool


@dataclass(frozen=True)
class BoxPrimitive:
    surface_id: str
    center: np.ndarray
    half_extents: np.ndarray
    rotation: np.ndarray
    surface_type: str = "box"

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float).reshape(3))
        half_extents = np.asarray(self.half_extents, dtype=float).reshape(3)
        if np.any(half_extents <= 0.0):
            raise ValueError(f"Box {self.surface_id} half extents must be positive")
        object.__setattr__(self, "half_extents", half_extents)
        rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError(f"Box {self.surface_id} rotation must be orthonormal")
        object.__setattr__(self, "rotation", rotation)

    def to_dict(self) -> dict:
        return {
            "type": self.surface_type,
            "surface_id": self.surface_id,
            "center": self.center.tolist(),
            "half_extents": self.half_extents.tolist(),
            "rotation": self.rotation.tolist(),
        }


class TerrainField:
    """Solid union of a horizontal floor and deterministic box primitives."""

    _FACE_ORDER = ((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1))

    def __init__(
        self,
        boxes: list[BoxPrimitive] | None = None,
        *,
        floor_z: float | None = 0.0,
        floor_id: str = "floor",
        support_normal_min_z: float = 0.6,
    ) -> None:
        self.boxes = sorted(list(boxes or []), key=lambda item: item.surface_id)
        self.floor_z = None if floor_z is None else float(floor_z)
        self.floor_id = str(floor_id)
        self.support_normal_min_z = float(support_normal_min_z)
        # Cached arrays keep the hot batch query out of the Python box loop.
        self._box_centers = np.asarray([b.center for b in self.boxes], dtype=float).reshape((-1, 3))
        self._box_half_extents = np.asarray([b.half_extents for b in self.boxes], dtype=float).reshape((-1, 3))
        self._box_rotations = np.asarray([b.rotation for b in self.boxes], dtype=float).reshape((-1, 3, 3))

    @staticmethod
    def _box_hit(box: BoxPrimitive, point: np.ndarray, support_min_z: float) -> TerrainSurfaceHit:
        point = np.asarray(point, dtype=float).reshape(3)
        local = box.rotation.T @ (point - box.center)
        delta = np.abs(local) - box.half_extents
        outside = np.maximum(delta, 0.0)
        outside_norm = float(np.linalg.norm(outside))
        if outside_norm > 1e-12:
            closest_local = np.clip(local, -box.half_extents, box.half_extents)
            closest = box.center + box.rotation @ closest_local
            normal = _unit(point - closest)
            signed_distance = outside_norm
            face_axis = int(np.argmax(np.abs(box.rotation.T @ normal)))
            face_sign = 1 if (box.rotation.T @ normal)[face_axis] >= 0.0 else -1
        else:
            clearances = box.half_extents - np.abs(local)
            candidates = []
            for order, (axis, sign) in enumerate(TerrainField._FACE_ORDER):
                distance = float(box.half_extents[axis] - sign * local[axis])
                candidates.append((distance, order, axis, sign))
            depth, _, face_axis, face_sign = min(candidates)
            closest_local = local.copy()
            closest_local[face_axis] = face_sign * box.half_extents[face_axis]
            closest = box.center + box.rotation @ closest_local
            normal = box.rotation[:, face_axis] * face_sign
            signed_distance = -float(depth)
        face_name = ("x", "y", "z")[face_axis] + ("+" if face_sign > 0 else "-")
        normal = _unit(normal)
        return TerrainSurfaceHit(
            signed_distance=signed_distance,
            closest_point=closest,
            normal=normal,
            surface_id=f"{box.surface_id}:{face_name}",
            surface_type=box.surface_type,
            supportable=bool(normal[2] > support_min_z),
        )

    def nearest_surface(self, point: np.ndarray) -> TerrainSurfaceHit:
        point = np.asarray(point, dtype=float).reshape(3)
        hits = [self._box_hit(box, point, self.support_normal_min_z) for box in self.boxes]
        if self.floor_z is not None:
            distance = float(point[2] - self.floor_z)
            hits.append(TerrainSurfaceHit(
                signed_distance=distance,
                closest_point=np.array([point[0], point[1], self.floor_z]),
                normal=np.array([0.0, 0.0, 1.0]),
                surface_id=self.floor_id,
                surface_type="floor",
                supportable=True,
            ))
        if not hits:
            raise ValueError("TerrainField has no surfaces")
        # SDF of a solid union is the minimum component SDF.  surface_id is a
        # deterministic secondary key for exact edge/corner ties.
        return min(hits, key=lambda hit: (hit.signed_distance, hit.surface_id))

    def nearest_surface_batch(self, points: np.ndarray) -> list[TerrainSurfaceHit]:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected (N,3) points, got {points.shape}")
        if not len(points):
            return []
        if not self.boxes:
            if self.floor_z is None:
                raise ValueError("TerrainField has no surfaces")
            return [TerrainSurfaceHit(float(p[2] - self.floor_z),
                                      np.array([p[0], p[1], self.floor_z]),
                                      np.array([0., 0., 1.]), self.floor_id,
                                      "floor", True) for p in points]

        # Batched OBB SDF.  Only the winning box is materialized into a
        # TerrainSurfaceHit below; this removes the O(N*B) dataclass and
        # normalization overhead from every IK substep.
        local = np.einsum("bji,nbj->nbi", self._box_rotations,
                          points[:, None, :] - self._box_centers[None, :, :])
        delta = np.abs(local) - self._box_half_extents[None, :, :]
        outside = np.maximum(delta, 0.0)
        sdf = np.linalg.norm(outside, axis=-1)
        inside = np.max(delta, axis=-1) <= 0.0
        sdf[inside] = np.max(delta, axis=-1)[inside]
        box_index = np.argmin(sdf, axis=1)
        box_distance = sdf[np.arange(len(points)), box_index]

        hits: list[TerrainSurfaceHit] = []
        for i, point in enumerate(points):
            box = self.boxes[int(box_index[i])]
            candidate = self._box_hit(box, point, self.support_normal_min_z)
            if self.floor_z is not None:
                floor_distance = float(point[2] - self.floor_z)
                floor_hit = TerrainSurfaceHit(
                    floor_distance, np.array([point[0], point[1], self.floor_z]),
                    np.array([0., 0., 1.]), self.floor_id, "floor", True)
                if (floor_distance, self.floor_id) < (candidate.signed_distance, candidate.surface_id):
                    candidate = floor_hit
            hits.append(candidate)
        return hits

    def nearest_surface_batch_arrays(self, points: np.ndarray) -> dict[str, np.ndarray]:
        """Vectorized SDF query for hot IK loops.

        The returned arrays intentionally contain only numeric data.  Callers
        that need the richer ``TerrainSurfaceHit`` object can materialize it
        for the small active subset after this query.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected (N,3) points, got {points.shape}")
        n = len(points)
        if n == 0:
            return {
                "signed_distance": np.empty(0),
                "closest_point": np.empty((0, 3)),
                "normal": np.empty((0, 3)),
                "surface_id": np.empty(0, dtype=object),
                "supportable": np.empty(0, dtype=bool),
            }
        # The scalar path is used only for the floor-only case and preserves
        # the exact floor semantics without allocating box arrays.
        if not self.boxes:
            if self.floor_z is None:
                raise ValueError("TerrainField has no surfaces")
            closest = points.copy()
            closest[:, 2] = self.floor_z
            return {
                "signed_distance": points[:, 2] - self.floor_z,
                "closest_point": closest,
                "normal": np.tile([0.0, 0.0, 1.0], (n, 1)),
                "surface_id": np.full(n, self.floor_id, dtype=object),
                "supportable": np.ones(n, dtype=bool),
            }
        local = np.einsum(
            "bji,nbj->nbi", self._box_rotations,
            points[:, None, :] - self._box_centers[None, :, :],
        )
        delta = np.abs(local) - self._box_half_extents[None, :, :]
        outside = np.maximum(delta, 0.0)
        sdf = np.linalg.norm(outside, axis=-1)
        inside = np.max(delta, axis=-1) <= 0.0
        sdf[inside] = np.max(delta, axis=-1)[inside]
        winner = np.argmin(sdf, axis=1)
        distance = sdf[np.arange(n), winner]
        closest = np.empty((n, 3), dtype=float)
        normals = np.empty((n, 3), dtype=float)
        surface_ids = np.empty(n, dtype=object)
        supportable = np.zeros(n, dtype=bool)
        # Only one scalar face query per point is needed after the vectorized
        # winner selection; no point-box dataclass allocation is performed.
        for box_index, box in enumerate(self.boxes):
            indices = np.flatnonzero(winner == box_index)
            for index in indices:
                hit = self._box_hit(box, points[index], self.support_normal_min_z)
                closest[index] = hit.closest_point
                normals[index] = hit.normal
                surface_ids[index] = hit.surface_id
                supportable[index] = hit.supportable
        if self.floor_z is not None:
            floor_distance = points[:, 2] - self.floor_z
            use_floor = (floor_distance, np.full(n, self.floor_id, dtype=object))
            # Stable lexicographic tie-break is equivalent to nearest_surface.
            replace = floor_distance < distance
            replace |= (floor_distance == distance) & (np.asarray(surface_ids, dtype=str) > self.floor_id)
            if np.any(replace):
                closest[replace] = points[replace]
                closest[replace, 2] = self.floor_z
                normals[replace] = [0.0, 0.0, 1.0]
                surface_ids[replace] = self.floor_id
                supportable[replace] = True
                distance[replace] = floor_distance[replace]
        return {
            "signed_distance": distance,
            "closest_point": closest,
            "normal": normals,
            "surface_id": surface_ids,
            "supportable": supportable,
        }

    def support_surface(self, point: np.ndarray) -> TerrainSurfaceHit:
        point = np.asarray(point, dtype=float).reshape(3)
        hits: list[TerrainSurfaceHit] = []
        if self.floor_z is not None and self.floor_z <= point[2] + 1e-9:
            hits.append(TerrainSurfaceHit(
                signed_distance=float(point[2] - self.floor_z),
                closest_point=np.array([point[0], point[1], self.floor_z]),
                normal=np.array([0.0, 0.0, 1.0]),
                surface_id=self.floor_id,
                surface_type="floor",
                supportable=True,
            ))
        ray = np.array([0.0, 0.0, -1.0])
        for box in self.boxes:
            for axis, sign in self._FACE_ORDER:
                normal = box.rotation[:, axis] * sign
                if normal[2] <= self.support_normal_min_z:
                    continue
                face_center = box.center + normal * box.half_extents[axis]
                denominator = float(normal @ ray)
                if abs(denominator) < 1e-10:
                    continue
                t = float(normal @ (face_center - point) / denominator)
                if t < -1e-9:
                    # A noisy contact marker may already be slightly inside a
                    # support solid.  Keep the upward face as its support when
                    # the point itself lies inside the box; otherwise a point
                    # just below a platform would incorrectly fall through to
                    # the global floor.
                    local_point = box.rotation.T @ (point - box.center)
                    horizontal_axes = [index for index in range(3) if index != axis]
                    inside = (
                        abs(local_point[axis]) <= box.half_extents[axis] + 1e-8
                        and all(abs(local_point[index]) <= box.half_extents[index] + 1e-8 for index in horizontal_axes)
                    )
                    if not inside:
                        continue
                    closest = point + t * ray
                else:
                    closest = point + max(t, 0.0) * ray
                local = box.rotation.T @ (closest - box.center)
                other_axes = [index for index in range(3) if index != axis]
                if any(abs(local[index]) > box.half_extents[index] + 1e-8 for index in other_axes):
                    continue
                face_name = ("x", "y", "z")[axis] + ("+" if sign > 0 else "-")
                hits.append(TerrainSurfaceHit(
                    signed_distance=float(normal @ (point - closest)),
                    closest_point=closest,
                    normal=normal,
                    surface_id=f"{box.surface_id}:{face_name}",
                    surface_type=box.surface_type,
                    supportable=True,
                ))
        if not hits:
            return self.nearest_surface(point)
        # Highest valid support wins; stable surface id resolves coplanar ties.
        return min(hits, key=lambda hit: (-hit.closest_point[2], hit.surface_id))

    def transform(self, scene_transform: SceneTransform) -> "TerrainField":
        floor_z = None
        if self.floor_z is not None:
            normal = scene_transform.transform_normals(np.array([0.0, 0.0, 1.0]))
            if normal[2] < 1.0 - 1e-6:
                raise ValueError("A tilted infinite floor is not representable by floor_z")
            floor_z = float(scene_transform.transform_points(np.array([0.0, 0.0, self.floor_z]))[2])
        return TerrainField(
            [scene_transform.transform_box(box) for box in self.boxes],
            floor_z=floor_z,
            floor_id=self.floor_id,
            support_normal_min_z=self.support_normal_min_z,
        )

    def to_spec(self) -> dict:
        return {
            "floor_z": self.floor_z,
            "floor_id": self.floor_id,
            "support_normal_min_z": self.support_normal_min_z,
            "primitives": [box.to_dict() for box in self.boxes],
        }

    @classmethod
    def from_spec(cls, spec: dict) -> "TerrainField":
        boxes = []
        for index, item in enumerate(spec.get("primitives", [])):
            primitive_type = item.get("type", "box")
            if primitive_type not in {"box", "aabb", "obb"}:
                raise ValueError(f"Unsupported terrain primitive: {primitive_type}")
            boxes.append(BoxPrimitive(
                surface_id=str(item.get("surface_id", f"box_{index}")),
                center=item["center"],
                half_extents=item["half_extents"],
                rotation=item.get("rotation", np.eye(3)),
                surface_type=primitive_type,
            ))
        return cls(
            boxes,
            floor_z=spec.get("floor_z", 0.0),
            floor_id=spec.get("floor_id", "floor"),
            support_normal_min_z=spec.get("support_normal_min_z", 0.6),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "TerrainField":
        path = Path(path)
        if path.suffix.lower() == ".json":
            return cls.from_spec(json.loads(path.read_text()))
        if path.suffix.lower() == ".urdf":
            return cls.from_holosoma_multi_boxes(path)
        if path.suffix.lower() == ".obj":
            return cls.from_multi_boxes_obj(path)
        raise ValueError(f"Unsupported terrain file: {path}")

    @classmethod
    def from_multi_boxes_obj(cls, obj_path: str | Path, *, floor_z: float = 0.0) -> "TerrainField":
        """Recover box OBBs from the connected components of a source OBJ."""
        obj_path = Path(obj_path)
        vertices: list[list[float]] = []
        faces: list[list[int]] = []
        with obj_path.open("r", encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                if line.startswith("v "):
                    fields = line.split()
                    vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
                elif line.startswith("f "):
                    faces.append([int(field.split("/")[0]) - 1 for field in line.split()[1:]])
        if not vertices or not faces:
            raise ValueError(f"No box mesh geometry found in {obj_path}")
        vertices_array = np.asarray(vertices, dtype=float)
        adjacency = [set() for _ in vertices]
        for face in faces:
            for first in face:
                adjacency[first].update(second for second in face if second != first)
        components, unseen = [], set(range(len(vertices)))
        while unseen:
            seed = min(unseen)
            stack, component = [seed], set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                unseen.discard(current)
                stack.extend(sorted(adjacency[current] - component, reverse=True))
            components.append(sorted(component))
        boxes = []
        for component_index, indices in enumerate(components):
            points = vertices_array[indices]
            if len(points) != 8:
                raise ValueError(f"Expected 8 vertices for box component {component_index}, got {len(points)}")
            z_min, z_max = float(points[:, 2].min()), float(points[:, 2].max())
            bottom = points[np.isclose(points[:, 2], z_min)]
            if len(bottom) != 4:
                raise ValueError(f"Box component {component_index} has {len(bottom)} bottom vertices")
            origin = bottom[np.lexsort((bottom[:, 1], bottom[:, 0]))[0]]
            horizontal = [point - origin for point in bottom if np.linalg.norm(point - origin) > 1e-9]
            horizontal.sort(key=lambda vector: (float(np.linalg.norm(vector)), *vector.tolist()))
            first = horizontal[0]
            second = next(
                vector for vector in horizontal[1:]
                if abs(float(np.dot(first[:2], vector[:2])))
                < 1e-6 * np.linalg.norm(first[:2]) * np.linalg.norm(vector[:2]) + 1e-9
            )
            x_axis = _unit(np.array([first[0], first[1], 0.0]))
            y_axis = _unit(np.array([second[0], second[1], 0.0]))
            if np.linalg.det(np.column_stack((x_axis, y_axis, [0.0, 0.0, 1.0]))) < 0.0:
                y_axis = -y_axis
            rotation = np.column_stack((x_axis, y_axis, [0.0, 0.0, 1.0]))
            center = points.mean(axis=0)
            local = (points - center) @ rotation
            half_extents = np.max(np.abs(local), axis=0)
            boxes.append(BoxPrimitive(
                surface_id=f"box_{component_index + 1}",
                center=center,
                half_extents=half_extents,
                rotation=rotation,
                surface_type="obb",
            ))
        return cls(boxes, floor_z=floor_z)

    @classmethod
    def from_holosoma_multi_boxes(cls, urdf_path: str | Path, *, floor_z: float = 0.0) -> "TerrainField":
        urdf_path = Path(urdf_path)
        root = ET.parse(urdf_path).getroot()
        boxes: list[BoxPrimitive] = []
        for link in root.findall("link"):
            collision = link.find("collision")
            if collision is None:
                continue
            mesh = collision.find("geometry/mesh")
            if mesh is None or "filename" not in mesh.attrib:
                continue
            mesh_path = (urdf_path.parent / mesh.attrib["filename"]).resolve()
            scale = np.fromstring(mesh.attrib.get("scale", "1 1 1"), sep=" ")
            if scale.size != 3 or not np.allclose(scale, scale[0], atol=1e-9):
                raise ValueError(f"Terrain requires one uniform scene scale, got {scale} in {mesh_path}")
            vertices = []
            with mesh_path.open("r", encoding="utf-8", errors="ignore") as stream:
                for line in stream:
                    if line.startswith("v "):
                        fields = line.split()
                        vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            if not vertices:
                raise ValueError(f"No vertices found in {mesh_path}")
            vertices = np.asarray(vertices, dtype=float) * float(scale[0])
            lower, upper = vertices.min(axis=0), vertices.max(axis=0)
            boxes.append(BoxPrimitive(
                surface_id=link.attrib["name"],
                center=0.5 * (lower + upper),
                half_extents=0.5 * (upper - lower),
                rotation=np.eye(3),
                surface_type="aabb",
            ))
        if not boxes:
            raise ValueError(f"No collision mesh boxes found in {urdf_path}")
        return cls(boxes, floor_z=floor_z)
