"""Prepared scene geometry shared by V5 query, interaction and MuJoCo paths.

The important invariant is that a scene asset is decoded, unit-normalized and
assigned its object pose once.  Every downstream consumer receives the same
``SceneMesh`` instance: triangle contact queries, interaction samples, CoACD
collision decomposition and MuJoCo visual meshes cannot silently choose
different scale metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json

import numpy as np

from ..core.schemas import SceneModel
from ..scene_asset_loader import SceneMesh, load_scene_asset
from ..terrain_geometry import BoxPrimitive, TerrainField
from .loader import CompositeSceneField, DynamicSceneField, MeshSceneField


_MESH_SUFFIXES = {".obj", ".usd", ".usda", ".usdc"}


def _asset_metadata(asset, sample_count: int) -> dict:
    return {
        "object_id": str(asset.asset_id),
        "unit_scale": 1.0 if asset.unit_scale is None else float(asset.unit_scale),
        "asset_scale_baked": bool(asset.asset_scale_baked),
        "asset_space": str(asset.asset_space),
        "sample_count": int(sample_count),
    }


def _pose_box(asset, box: BoxPrimitive) -> BoxPrimitive:
    """Map a local box primitive through its asset pose once.

    Meshes have always received ``SceneAsset.pose``; JSON/URDF primitives
    must obey the same world contract.  A sheared affine transform cannot be
    represented by this OBB data type, so it is rejected rather than silently
    producing a visual/query mismatch.
    """
    pose = np.asarray(asset.pose, dtype=float).reshape(4, 4)
    extent_axes = pose[:3, :3] @ np.asarray(box.rotation, dtype=float).reshape(3, 3)
    extent_axes = extent_axes * np.asarray(box.half_extents, dtype=float).reshape(1, 3)
    extents = np.linalg.norm(extent_axes, axis=0)
    if np.any(extents <= 1e-12):
        raise ValueError(f"Scene box {box.surface_id!r} has a collapsed transformed extent")
    rotation = extent_axes / extents
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(
            f"Scene box {box.surface_id!r} has a sheared/non-orthogonal asset transform; "
            "use a mesh asset for this geometry"
        )
    center = pose[:3, :3] @ np.asarray(box.center, dtype=float) + pose[:3, 3]
    return BoxPrimitive(str(box.surface_id), center, extents, rotation, str(box.surface_type))


@dataclass(frozen=True)
class PreparedSceneGeometry:
    """The one decoded geometry representation for a resolved V5 scene."""

    scene: SceneModel
    mesh_assets: tuple[tuple[object, SceneMesh], ...]
    boxes: tuple[BoxPrimitive, ...]
    sample_count: int

    @classmethod
    def from_scene(cls, scene: SceneModel, *, sample_count: int = 2048) -> "PreparedSceneGeometry":
        meshes: list[tuple[object, SceneMesh]] = []
        boxes: list[BoxPrimitive] = []
        for asset in scene.assets:
            suffix = asset.path.suffix.lower()
            if suffix == ".json":
                spec = json.loads(asset.path.read_text(encoding="utf-8"))
                for index, item in enumerate(spec.get("primitives", spec.get("boxes", []))):
                    boxes.append(_pose_box(asset, BoxPrimitive(
                        str(item.get("surface_id", f"{asset.asset_id}:{index}")),
                        item["center"], item["half_extents"],
                        item.get("rotation", np.eye(3)), item.get("type", "box"),
                    )))
                continue
            if suffix == ".urdf":
                primitive = TerrainField.from_file(asset.path, floor_z=None)
                boxes.extend(_pose_box(asset, box) for box in primitive.boxes)
                continue
            if suffix not in _MESH_SUFFIXES:
                raise ValueError(f"Unsupported scene loader format: {asset.path.suffix}")
            mesh = load_scene_asset(asset.path, _asset_metadata(asset, sample_count))
            # Asset pose is a world transform and remains separate from the
            # normalized object-local vertices for moving scene bodies.
            mesh.object_pose = np.asarray(asset.pose, dtype=float).copy()
            meshes.append((asset, mesh))
        return cls(scene, tuple(meshes), tuple(boxes), int(sample_count))

    def with_scene(self, scene: SceneModel) -> "PreparedSceneGeometry":
        """Reuse decoded geometry after a floor-policy-only model update."""
        if len(scene.assets) != len(self.scene.assets) or any(
            left.path != right.path or not np.allclose(left.pose, right.pose)
            for left, right in zip(scene.assets, self.scene.assets)
        ):
            raise ValueError("Prepared geometry cannot be reused after asset transforms change")
        return replace(self, scene=scene)

    @property
    def meshes(self) -> tuple[SceneMesh, ...]:
        return tuple(mesh for _, mesh in self.mesh_assets)

    def terrain_field(self):
        """Build a query field directly from these prepared meshes."""
        floor = self.scene.floor_height if self.scene.floor_policy.value != "disabled" else None
        if self.mesh_assets:
            if any(asset.pose_trajectory is not None for asset, _ in self.mesh_assets):
                return DynamicSceneField(self.mesh_assets, floor_z=floor, support_normal_min_z=0.6)
            fields = []
            for asset, mesh in self.mesh_assets:
                pose = np.asarray(asset.pose, dtype=float)
                vertices = (np.c_[mesh.vertices, np.ones(len(mesh.vertices))] @ pose.T)[:, :3]
                fields.append(MeshSceneField(
                    vertices, mesh.faces, asset.asset_id, floor_z=None,
                    asset_inverse=np.linalg.inv(pose),
                ))
            if self.boxes:
                fields.append(TerrainField(list(self.boxes), floor_z=None))
            if floor is not None:
                fields.append(TerrainField(floor_z=floor))
            return fields[0] if len(fields) == 1 else CompositeSceneField(fields)
        return TerrainField(list(self.boxes), floor_z=floor)

    def world_vertices(self, *, timestamp: float = 0.0) -> dict[str, np.ndarray]:
        """Return final world vertices for alignment and floor diagnostics."""
        values = {}
        for asset, mesh in self.mesh_assets:
            pose = np.asarray(asset.pose_at(timestamp), dtype=float)
            values[str(asset.asset_id)] = (
                np.c_[mesh.vertices, np.ones(len(mesh.vertices))] @ pose.T
            )[:, :3]
        return values

    def lower_extent(self, quantile: float = 0.005) -> tuple[float | None, list[dict]]:
        """Deterministically infer a floor candidate from prepared assets."""
        quantile = float(np.clip(quantile, 0.0, 0.25))
        records, values = [], []
        for asset_id, vertices in self.world_vertices().items():
            if not len(vertices):
                continue
            value = float(np.quantile(vertices[:, 2], quantile))
            values.append(value)
            records.append({
                "asset_id": asset_id,
                "quantile": value,
                "min": float(np.min(vertices[:, 2])),
                "max": float(np.max(vertices[:, 2])),
            })
        # Analytic box/URDF assets do not have mesh vertices, but their lower
        # support extent is just as authoritative for floor inference.  Keep
        # the same deterministic asset-level record shape for diagnostics.
        for box in self.boxes:
            z_extent = float(np.abs(np.asarray(box.rotation, dtype=float)[2]) @ np.asarray(box.half_extents, dtype=float))
            lower = float(np.asarray(box.center, dtype=float)[2] - z_extent)
            values.append(lower)
            records.append({
                "asset_id": str(box.surface_id),
                "quantile": lower,
                "min": lower,
                "max": float(np.asarray(box.center, dtype=float)[2] + z_extent),
                "surface_type": "box",
            })
        return (float(min(values)) if values else None), records
