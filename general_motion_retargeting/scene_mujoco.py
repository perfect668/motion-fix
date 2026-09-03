"""Build a temporary robot + visual/collision scene MuJoCo model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from .scene_asset_loader import SceneMesh, decompose_cached
from .scene_geometry import SceneGeometry


@dataclass(frozen=True)
class CombinedSceneModel:
    xml_path: Path
    manifest_path: Path
    robot_geom_ids: tuple[int, ...]
    scene_geom_ids: tuple[int, ...]
    scene_body_ids: tuple[int, ...]
    object_ids: tuple[str, ...]
    visual_geom_count: int
    collision_geom_count: int


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _numbers(values: Any) -> str:
    return " ".join(f"{float(value):.12g}" for value in np.asarray(values).reshape(-1))


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    # MuJoCo quaternion order is w, x, y, z.
    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
                               (matrix[0, 2] - matrix[2, 0]) / scale,
                               (matrix[1, 0] - matrix[0, 1]) / scale])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array([(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                                   (matrix[0, 1] + matrix[1, 0]) / scale,
                                   (matrix[0, 2] + matrix[2, 0]) / scale])
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array([(matrix[0, 2] - matrix[2, 0]) / scale,
                                   (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                                   (matrix[1, 2] + matrix[2, 1]) / scale])
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array([(matrix[1, 0] - matrix[0, 1]) / scale,
                                   (matrix[0, 2] + matrix[2, 0]) / scale,
                                   (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale])
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def _pose_components(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose = np.asarray(pose, dtype=float).reshape(4, 4)
    linear = pose[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    if np.any(scale < 1e-12):
        raise ValueError("Scene object pose contains a zero scale axis")
    rotation = linear / scale
    # Reject shear because MJCF has no body-level shear representation.
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("Scene object pose contains shear; only rotation and axis scale are supported")
    if np.linalg.det(rotation) < 0:
        scale[0] *= -1
        rotation[:, 0] *= -1
    return pose[:3, 3], _rotation_to_quaternion(rotation), scale


def _absolutize_compiler_paths(root: ET.Element, robot_xml: Path) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    for attribute in ("meshdir", "texturedir", "assetdir"):
        value = compiler.get(attribute)
        if value and not Path(value).is_absolute():
            compiler.set(attribute, str((robot_xml.parent / value).resolve()))


def _mesh_from_scene_object(obj: Any) -> SceneMesh:
    raw_meshes = [item for item in obj.collision_representation if item.get("type", "source_mesh") == "source_mesh" and "vertices" in item]
    if not raw_meshes:
        raise ValueError(f"Scene object {obj.object_id!r} has no source triangle mesh")
    if len(raw_meshes) > 1:
        raise ValueError(f"Scene object {obj.object_id!r} has multiple source meshes; load it as one SceneMesh")
    raw = raw_meshes[0]
    source = raw.get("source_path")
    if not source:
        raise ValueError(f"Scene object {obj.object_id!r} source mesh has no source_path")
    return SceneMesh(raw["vertices"], raw["faces"], obj.object_id, source, object_pose=obj.pose)


def _normalize_scene(scene: SceneMesh | SceneGeometry | dict | str | Path) -> tuple[list[SceneMesh], list[dict[str, Any]], float | None]:
    if isinstance(scene, SceneMesh):
        return [scene], [], 0.0
    if isinstance(scene, SceneGeometry):
        return [_mesh_from_scene_object(obj) for obj in scene.objects], [], scene.floor_z
    spec = json.loads(Path(scene).read_text(encoding="utf-8")) if not isinstance(scene, dict) else scene
    return [], list(spec.get("objects", [])), spec.get("floor_z")


def _add_primitive(body: ET.Element, object_id: str, collision: dict[str, Any], obj: dict[str, Any]) -> int:
    geometry_type = collision.get("type", "box")
    if geometry_type not in {"box", "sphere", "capsule", "cylinder"}:
        return 0
    attributes = {
        "name": f"scene_{object_id}_collision_000",
        "type": geometry_type,
        "contype": "1",
        "conaffinity": "1",
        "group": "3",
        "rgba": "0.25 0.55 0.85 0.25",
    }
    if geometry_type == "box":
        attributes["size"] = _numbers(collision.get("half_extents", [0.2, 0.2, 0.2]))
    elif geometry_type == "sphere":
        attributes["size"] = _numbers([collision.get("radius", 0.2)])
    else:
        attributes["size"] = _numbers([collision.get("radius", 0.1), collision.get("half_height", 0.3)])
    attributes["pos"] = _numbers(collision.get("position", obj.get("position", [0, 0, 0])))
    if "quaternion" in collision:
        attributes["quat"] = _numbers(collision["quaternion"])
    ET.SubElement(body, "geom", **attributes)
    return 1


def build_scene_model(
    robot_xml: str | Path,
    scene_spec: SceneMesh | SceneGeometry | str | Path | dict,
    output_xml: str | Path,
    *,
    cache_root: str | Path = ".cache/scene_collision",
    decomposition_config: dict[str, Any] | None = None,
    show_collision: bool = False,
    return_info: bool = False,
) -> Path | CombinedSceneModel:
    """Build and compile a combined robot and scene MJCF.

    ``SceneMesh``/``SceneGeometry`` inputs automatically receive a cached CoACD
    representation.  The legacy primitive/manifest dictionary input remains
    supported.  The original mesh is visual-only; convex pieces are
    collision-only.
    """

    robot_xml = Path(robot_xml).expanduser().resolve()
    if not robot_xml.is_file():
        raise FileNotFoundError(f"Robot XML does not exist: {robot_xml}")
    root = ET.parse(robot_xml).getroot()
    _absolutize_compiler_paths(root, robot_xml)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    meshes, legacy_objects, floor_z = _normalize_scene(scene_spec)
    decomposition_config = dict(decomposition_config or {})
    generated_objects: list[dict[str, Any]] = []
    for mesh in meshes:
        manifest, cache_dir = decompose_cached(mesh, cache_root=cache_root, **decomposition_config)
        generated_objects.append({
            "object_id": mesh.object_id,
            "pose": mesh.object_pose,
            "scale": np.ones(3),
            "visual_mesh": cache_dir / manifest["visual_mesh"],
            "collision_manifest": cache_dir / "collision_manifest.json",
            "collision_cache": cache_dir,
            "source_asset": mesh.source_path,
            "source_vertices": len(mesh.vertices),
            "source_faces": len(mesh.faces),
            "interaction_points": len(
                mesh.to_scene_geometry(int(mesh.metadata.get("sample_count", 384))).objects[0].surface_samples
            ),
            "collision_cache_hit": bool(manifest.get("cache_hit", False)),
            "source_watertight": manifest.get("source_watertight"),
            "decomposition_parameters": manifest.get("decomposition_parameters", {}),
        })

    visual_count = 0
    collision_count = 0
    build_objects: list[dict[str, Any]] = []
    for generated in generated_objects:
        object_id = _safe_name(str(generated["object_id"]))
        position, quaternion, pose_scale = _pose_components(generated["pose"])
        body = ET.SubElement(
            worldbody,
            "body",
            name=f"scene_{object_id}",
            pos=_numbers(position),
            quat=_numbers(quaternion),
            mocap="true",
        )
        visual_mesh_name = f"scene_{object_id}_visual_mesh"
        ET.SubElement(asset, "mesh", name=visual_mesh_name, file=str(generated["visual_mesh"]), scale=_numbers(pose_scale))
        ET.SubElement(body, "geom", name=f"scene_{object_id}_visual", type="mesh", mesh=visual_mesh_name,
                      contype="0", conaffinity="0", group="2", rgba="0.62 0.48 0.30 1")
        visual_count += 1
        manifest = json.loads(Path(generated["collision_manifest"]).read_text(encoding="utf-8"))
        collision_names: list[str] = []
        for index, piece in enumerate(manifest.get("pieces", [])):
            piece_path = (Path(generated["collision_manifest"]).parent / piece).resolve()
            mesh_name = f"scene_{object_id}_collision_mesh_{index:03d}"
            geom_name = f"scene_{object_id}_collision_{index:03d}"
            ET.SubElement(asset, "mesh", name=mesh_name, file=str(piece_path), scale=_numbers(pose_scale))
            ET.SubElement(body, "geom", name=geom_name, type="mesh", mesh=mesh_name, contype="1",
                          conaffinity="1", group="3", rgba="0.15 0.65 0.95 0.22" if show_collision else "0 0 0 0")
            collision_names.append(geom_name)
            collision_count += 1
        build_objects.append({
            **{key: _json_value(value) for key, value in generated.items() if key != "pose"},
            "pose": np.asarray(generated["pose"]).tolist(),
            "body_name": f"scene_{object_id}",
            "visual_geom_name": f"scene_{object_id}_visual",
            "collision_geom_names": collision_names,
            "convex_pieces": len(collision_names),
        })

    # Backward-compatible path for primitive or pre-decomposed specifications.
    for obj in legacy_objects:
        object_id = _safe_name(str(obj["object_id"]))
        position = obj.get("position", [0, 0, 0])
        quaternion = obj.get("quaternion", [1, 0, 0, 0])
        body = ET.SubElement(worldbody, "body", name=f"scene_{object_id}", pos=_numbers(position), quat=_numbers(quaternion), mocap="true")
        collision = obj.get("collision", {})
        count = _add_primitive(body, object_id, collision, obj)
        collision_names = [f"scene_{object_id}_collision_000"] if count else []
        if not count and collision.get("type") in {"convex_decomposition", "decomposed_mesh"}:
            manifest_path = Path(collision["manifest"]).expanduser().resolve()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for index, piece in enumerate(manifest.get("pieces", [])):
                mesh_name = f"scene_{object_id}_collision_mesh_{index:03d}"
                geom_name = f"scene_{object_id}_collision_{index:03d}"
                ET.SubElement(asset, "mesh", name=mesh_name, file=str((manifest_path.parent / piece).resolve()))
                ET.SubElement(body, "geom", name=geom_name, type="mesh", mesh=mesh_name, contype="1", conaffinity="1", group="3")
                collision_names.append(geom_name)
            count = len(collision_names)
        if not count:
            raise ValueError(f"Unsupported scene collision type: {collision.get('type', 'missing')}")
        collision_count += count
        build_objects.append({"object_id": obj["object_id"], "body_name": f"scene_{object_id}", "collision_geom_names": collision_names, "convex_pieces": count})

    if not collision_count and (meshes or legacy_objects):
        raise RuntimeError("Scene objects were provided but no MuJoCo collision geoms were generated")
    output = Path(output_xml).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)

    # Compilation is an assertion that the scene truly entered MuJoCo, and
    # gives the caller stable geom/body registrations for collision limits.
    try:
        import mujoco as mj
        model = mj.MjModel.from_xml_path(str(output))
    except Exception as error:
        raise RuntimeError(f"Combined robot + scene MuJoCo model failed to compile: {output}: {error}") from error
    scene_body_ids = tuple(
        body_id for body_id in range(model.nbody)
        if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) or "").startswith("scene_")
    )
    scene_body_set = set(scene_body_ids)
    scene_geom_ids = tuple(geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) in scene_body_set)
    robot_geom_ids = tuple(
        geom_id for geom_id in range(model.ngeom)
        if geom_id not in scene_geom_ids and int(model.geom_bodyid[geom_id]) != 0
    )
    collision_scene_ids = tuple(geom_id for geom_id in scene_geom_ids if int(model.geom_contype[geom_id]) != 0)
    if (meshes or legacy_objects) and not collision_scene_ids:
        raise RuntimeError("Combined MuJoCo model compiled but contains no collidable scene geoms")

    sidecar = output.with_suffix(".scene.json")
    payload = {
        "version": 1,
        "robot_xml": str(robot_xml),
        "combined_xml": str(output),
        "floor_z": floor_z,
        "objects": build_objects,
        "object_ids": [str(item["object_id"]) for item in build_objects],
        "robot_geom_ids": list(robot_geom_ids),
        "scene_geom_ids": list(scene_geom_ids),
        "scene_collision_geom_ids": list(collision_scene_ids),
        "visual_geom_count": visual_count,
        "collision_geom_count": len(collision_scene_ids),
    }
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    info = CombinedSceneModel(
        xml_path=output,
        manifest_path=sidecar,
        robot_geom_ids=robot_geom_ids,
        scene_geom_ids=scene_geom_ids,
        scene_body_ids=scene_body_ids,
        object_ids=tuple(payload["object_ids"]),
        visual_geom_count=visual_count,
        collision_geom_count=len(collision_scene_ids),
    )
    return info if return_info else output
