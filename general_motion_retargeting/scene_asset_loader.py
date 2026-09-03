"""Scene mesh loading and deterministic cached collision decomposition.

The source mesh is retained for rendering, interaction sampling, and source
contact queries.  CoACD pieces are a separate collision-only representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .scene_geometry import SceneGeometry, SceneObject, sample_mesh_surface


SUPPORTED_SCENE_SUFFIXES = {".obj", ".usd", ".usda", ".usdc"}


@dataclass
class SceneMesh:
    """One logical scene object's unified, object-local triangle mesh."""

    vertices: np.ndarray
    faces: np.ndarray
    object_id: str
    source_path: Path
    source_transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    object_pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape((-1, 3))
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape((-1, 3))
        self.source_path = Path(self.source_path).resolve()
        self.source_transform = np.asarray(self.source_transform, dtype=float).reshape(4, 4)
        self.object_pose = np.asarray(self.object_pose, dtype=float).reshape(4, 4)
        if not len(self.vertices) or not len(self.faces):
            raise ValueError(f"Scene mesh is empty: {self.source_path}")
        if self.faces.min() < 0 or self.faces.max() >= len(self.vertices):
            raise ValueError(f"Scene mesh has invalid face indices: {self.source_path}")
        for name, transform in (("source_transform", self.source_transform), ("object_pose", self.object_pose)):
            if not np.all(np.isfinite(transform)) or not np.allclose(transform[3], [0, 0, 0, 1]):
                raise ValueError(f"SceneMesh.{name} must be a finite homogeneous transform")

    def to_scene_geometry(self, sample_count: int = 384) -> SceneGeometry:
        samples = sample_mesh_surface(self.vertices, self.faces, sample_count)
        raw_mesh = {
            "type": "source_mesh",
            "vertices": self.vertices,
            "faces": self.faces,
            "source_path": str(self.source_path),
        }
        return SceneGeometry(
            [
                SceneObject(
                    object_id=self.object_id,
                    surface_samples=samples,
                    pose=self.object_pose,
                    collision_representation=[raw_mesh],
                )
            ]
        )

    @property
    def objects(self) -> list[SceneObject]:
        """Compatibility view for the earlier SceneGeometry loader result."""
        return self.to_scene_geometry(int(self.metadata.get("sample_count", 384))).objects


def _triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    triangles: list[list[int]] = []
    cursor = 0
    for count_value in counts:
        count = int(count_value)
        face = indices[cursor : cursor + count]
        cursor += count
        if count < 3:
            continue
        for index in range(1, count - 1):
            triangles.append([int(face[0]), int(face[index]), int(face[index + 1])])
    return np.asarray(triangles, dtype=np.int64).reshape((-1, 3))


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError("OBJ scene loading requires trimesh") from error
    loaded = trimesh.load(str(path), force="scene", process=False)
    meshes = list(loaded.geometry.values()) if hasattr(loaded, "geometry") else [loaded]
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for mesh in meshes:
        current_vertices = np.asarray(mesh.vertices, dtype=float)
        current_faces = np.asarray(mesh.faces, dtype=np.int64)
        if not len(current_vertices) or not len(current_faces):
            continue
        vertices.append(current_vertices)
        faces.append(current_faces + offset)
        offset += len(current_vertices)
    if not vertices:
        raise ValueError(f"OBJ contains no triangle mesh: {path}")
    return np.concatenate(vertices), np.concatenate(faces), np.eye(4)


def _load_usd(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from pxr import Usd, UsdGeom
    except ImportError as error:
        raise RuntimeError("USD scene loading requires usd-core (the pxr package)") from error
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open USD scene asset: {path}")
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points_value = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
        counts_value = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
        indices_value = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
        if points_value is None or counts_value is None or indices_value is None:
            continue
        points = np.asarray(points_value, dtype=float)
        triangles = _triangulate(np.asarray(counts_value), np.asarray(indices_value))
        if not len(points) or not len(triangles):
            continue
        # Gf matrices expose row-vector semantics.  Multiplying [x y z 1]
        # directly by the returned matrix applies the full parent hierarchy.
        matrix = np.asarray(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            dtype=float,
        )
        transformed = (np.c_[points, np.ones(len(points))] @ matrix)[:, :3]
        vertices.append(transformed)
        faces.append(triangles + offset)
        offset += len(points)
    if not vertices:
        raise ValueError(f"USD contains no usable UsdGeom.Mesh: {path}")
    # Prim hierarchy transforms have already been baked into one unified mesh.
    return np.concatenate(vertices), np.concatenate(faces), np.eye(4)


def _metadata_pose(metadata: dict[str, Any]) -> np.ndarray:
    if "pose" in metadata:
        return np.asarray(metadata["pose"], dtype=float).reshape(4, 4)
    pose = np.eye(4)
    rotation = metadata.get("rotation", metadata.get("obj_R"))
    translation = metadata.get("position", metadata.get("translation", metadata.get("obj_t")))
    scale = metadata.get("scale", metadata.get("obj_scale", 1.0))
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=float)
        if rotation.ndim == 3:
            rotation = rotation[0]
        pose[:3, :3] = rotation.reshape(3, 3)
    scale_array = np.asarray(scale, dtype=float).reshape(-1)
    if len(scale_array) == 1:
        pose[:3, :3] *= float(scale_array[0])
    elif len(scale_array) == 3:
        pose[:3, :3] = pose[:3, :3] @ np.diag(scale_array)
    else:
        raise ValueError(f"Object scale must have 1 or 3 values, got shape {np.shape(scale)}")
    if translation is not None:
        translation = np.asarray(translation, dtype=float)
        if translation.ndim == 2:
            translation = translation[0]
        pose[:3, 3] = translation.reshape(3)
    return pose


def load_scene_asset(
    path: str | Path,
    metadata: dict[str, Any] | None = None,
    sample_count: int = 384,
) -> SceneMesh:
    """Load every mesh prim/part from an OBJ or USD into one :class:`SceneMesh`.

    Metadata contains the logical object pose and is deliberately not baked
    into the vertices.  This keeps source interaction geometry reusable when
    an object pose changes between sequences or frames.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Scene asset does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SCENE_SUFFIXES:
        raise ValueError(f"Unsupported scene asset format {suffix!r}: {source}")
    vertices, faces, source_transform = (
        _load_usd(source) if suffix in {".usd", ".usda", ".usdc"} else _load_obj(source)
    )
    metadata = dict(metadata or {})
    metadata.setdefault("sample_count", int(sample_count))
    # USD prim hierarchy transforms are baked into vertices by _load_usd, but
    # the logical object pose/scale is still applied exactly once below.
    metadata.setdefault("asset_space", "unspecified")
    metadata.setdefault("asset_scale_baked", False)
    return SceneMesh(
        vertices=vertices,
        faces=faces,
        object_id=str(metadata.get("object_id", source.stem)),
        source_path=source,
        source_transform=source_transform,
        object_pose=_metadata_pose(metadata),
        metadata=metadata,
    )


def _mesh_digest(scene_mesh: SceneMesh, parameters: dict[str, Any]) -> tuple[str, str]:
    source_sha = hashlib.sha256(scene_mesh.source_path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    digest.update(source_sha.encode("ascii"))
    # Include extracted geometry because a USD may reference external assets
    # whose bytes are not part of the root layer.
    digest.update(np.ascontiguousarray(scene_mesh.vertices, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(scene_mesh.faces, dtype="<i8").tobytes())
    digest.update(json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return source_sha, digest.hexdigest()


def decompose_cached(
    scene_mesh: SceneMesh | SceneGeometry,
    source_path: str | Path | None = None,
    cache_root: str | Path = ".cache/scene_collision",
    resolution: int = 2_000,
    *,
    threshold: float = 0.05,
    max_convex_hull: int = 32,
    preprocess_mode: str = "auto",
) -> tuple[dict[str, Any], Path]:
    """Create or reuse deterministic CoACD convex pieces for ``scene_mesh``."""

    if isinstance(scene_mesh, SceneGeometry):
        if len(scene_mesh.objects) != 1 or not scene_mesh.objects[0].collision_representation:
            raise ValueError("SceneGeometry decomposition requires one object with a source mesh")
        obj = scene_mesh.objects[0]
        raw = obj.collision_representation[0]
        resolved_source = Path(source_path or raw.get("source_path", ""))
        if not resolved_source.is_file():
            raise FileNotFoundError("A valid source_path is required for SceneGeometry decomposition")
        scene_mesh = SceneMesh(raw["vertices"], raw["faces"], obj.object_id, resolved_source, object_pose=obj.pose)
    elif source_path is not None and Path(source_path).resolve() != scene_mesh.source_path:
        raise ValueError("source_path does not match SceneMesh.source_path")

    parameters = {
        "algorithm": "coacd",
        "resolution": int(resolution),
        "threshold": float(threshold),
        "max_convex_hull": int(max_convex_hull),
        "preprocess_mode": str(preprocess_mode),
    }
    source_sha, cache_key = _mesh_digest(scene_mesh, parameters)
    root = Path(cache_root).expanduser().resolve() / cache_key
    manifest_path = root / "collision_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        missing = [piece for piece in manifest.get("pieces", []) if not (root / piece).is_file()]
        if not missing and (root / manifest.get("visual_mesh", "source_visual.obj")).is_file():
            manifest["cache_hit"] = True
            return manifest, root
    # Reuse a valid decomposition made by an older V4 build when it has the
    # same source mesh hash.  New parameterized caches remain preferred above;
    # this avoids an expensive duplicate CoACD run after upgrading the loader.
    cache_root_path = Path(cache_root).expanduser().resolve()
    for legacy_manifest_path in sorted(cache_root_path.glob("*/collision_manifest.json")):
        try:
            legacy = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
            if legacy.get("source_sha256") != source_sha:
                continue
            legacy_root = legacy_manifest_path.parent
            pieces = legacy.get("pieces", [])
            visual = legacy.get("visual_mesh", "source_visual.obj")
            if pieces and all((legacy_root / p).is_file() for p in pieces) and (legacy_root / visual).is_file():
                legacy["cache_hit"] = True
                legacy["reused_for_parameters"] = parameters
                return legacy, legacy_root
        except (OSError, ValueError, TypeError):
            continue

    try:
        import coacd
        import trimesh
    except ImportError as error:
        raise RuntimeError(
            f"Complex scene object {scene_mesh.object_id!r} requires CoACD collision "
            "decomposition, but coacd/trimesh is unavailable"
        ) from error

    root.mkdir(parents=True, exist_ok=True)
    source_tri_mesh = trimesh.Trimesh(scene_mesh.vertices, scene_mesh.faces, process=False)
    visual_name = "source_visual.obj"
    source_tri_mesh.export(root / visual_name)
    try:
        result = coacd.run_coacd(
            coacd.Mesh(np.asarray(scene_mesh.vertices), np.asarray(scene_mesh.faces)),
            threshold=float(threshold),
            max_convex_hull=int(max_convex_hull),
            preprocess_mode=str(preprocess_mode),
            resolution=int(resolution),
        )
    except Exception as error:
        raise RuntimeError(f"CoACD failed for scene object {scene_mesh.object_id!r}: {error}") from error
    if not result:
        raise RuntimeError(f"CoACD produced no collision pieces for {scene_mesh.object_id!r}")
    pieces: list[str] = []
    for index, (vertices, faces) in enumerate(result):
        name = f"piece_{index:03d}.obj"
        trimesh.Trimesh(np.asarray(vertices), np.asarray(faces), process=False).export(root / name)
        pieces.append(name)
    manifest = {
        "version": 2,
        "object_id": scene_mesh.object_id,
        "source_mesh": str(scene_mesh.source_path),
        "source_sha256": source_sha,
        "cache_key": cache_key,
        "decomposition_parameters": parameters,
        "source_vertices": int(len(scene_mesh.vertices)),
        "source_faces": int(len(scene_mesh.faces)),
        "source_watertight": bool(source_tri_mesh.is_watertight),
        "visual_mesh": visual_name,
        "pieces": pieces,
        "piece_count": len(pieces),
        "cache_hit": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest, root
