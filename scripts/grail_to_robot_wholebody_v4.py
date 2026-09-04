"""End-to-end GRAIL -> NE01 WholeBody V4 conversion.

The scene is resolved from GRAIL metadata.  A missing object asset or a
failed convex decomposition is a hard error; the motion is never silently
converted without its interaction object.
"""
from __future__ import annotations

import copy
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _closest_point_triangle(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Closest point on one triangle (Ericson region test)."""
    a, b, c = np.asarray(triangle, dtype=float)
    ab, ac, ap = b - a, c - a, np.asarray(point, dtype=float) - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy()
    bp = np.asarray(point, dtype=float) - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return b.copy()
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + (d1 / max(d1 - d3, 1e-12)) * ab
    cp = np.asarray(point, dtype=float) - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return c.copy()
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + (d2 / max(d2 - d6, 1e-12)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return b + ((d4 - d3) / max((d4 - d3) + (d5 - d6), 1e-12)) * (c - b)
    denominator = max(va + vb + vc, 1e-12)
    v, w = vb / denominator, vc / denominator
    return a + ab * v + ac * w


def _closest_mesh_surface(point: np.ndarray, triangles: np.ndarray, centers: np.ndarray, normals: np.ndarray, candidate_count: int = 64):
    count = min(int(candidate_count), len(triangles))
    candidates = np.argsort(np.sum((centers - point[None, :]) ** 2, axis=1), kind="stable")[:count]
    best_index, best_point, best_distance = int(candidates[0]), None, float("inf")
    for index in candidates:
        closest = _closest_point_triangle(point, triangles[int(index)])
        distance = float(np.sum((closest - point) ** 2))
        if distance < best_distance:
            best_index, best_point, best_distance = int(index), closest, distance
    normal = normals[best_index].copy()
    if float(normal @ (point - best_point)) < 0.0:
        normal = -normal
    return best_index, best_point, normal, float(np.sqrt(best_distance))


def _asset_path(record: dict, motion: Path, override: Path | None = None) -> Path:
    if override is not None:
        if not override.is_file():
            raise FileNotFoundError(f"GRAIL object asset does not exist: {override}")
        return override.resolve()
    raw = str(record.get("object_path", ""))
    candidates = []
    if raw:
        raw_path = Path(raw).expanduser()
        candidates.append(raw_path if raw_path.is_absolute() else motion.parent / raw_path)
    # GRAIL reconstruction names are also used by the generated USD assets.
    candidates.append(motion.parent.parent / "object_usd" / f"{motion.stem}.usd")
    # Keep the error actionable without embedding developer-machine paths.
    unique_candidates = list(dict.fromkeys(str(path) for path in candidates))
    for candidate_name in unique_candidates:
        candidate = Path(candidate_name)
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "GRAIL metadata declares an object, but no mesh/USD asset was found. "
        f"object_path={raw!r}; checked: {unique_candidates}"
    )


def _object_pose(record: dict) -> np.ndarray:
    obj = record.get("obj_data") or {}
    R = np.asarray(obj.get("obj_R"), dtype=float)
    t = np.asarray(obj.get("obj_t"), dtype=float)
    scale = np.asarray(obj.get("obj_scale", np.ones((3, 1))), dtype=float).reshape(-1)
    if R.ndim == 3:
        R = R[0]
    if t.ndim == 2:
        t = t[0]
    if len(scale) == 1:
        scale = np.repeat(scale, 3)
    if R.shape != (3, 3) or t.shape != (3,) or len(scale) != 3:
        raise ValueError("Invalid GRAIL obj_R/obj_t/obj_scale metadata")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3):
        raise ValueError("GRAIL object rotation is not orthonormal")
    pose = np.eye(4)
    pose[:3, :3] = R @ np.diag(scale)
    pose[:3, 3] = t
    return pose


def main() -> None:
    import grail_to_robot_wholebody_v3 as impl
    from general_motion_retargeting.scene_asset_loader import decompose_cached, load_scene_asset
    from general_motion_retargeting.scene_mujoco import build_scene_model
    from general_motion_retargeting.wholebody_omni_gmr_v4 import WholeBodyOmniGMRV4

    # Parse only enough metadata to construct the combined MuJoCo model.  The
    # validated V3 adapter remains responsible for SMPL-X conversion/output.
    parser = impl.argparse.ArgumentParser(add_help=False)
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--save_path", required=True, type=Path)
    parser.add_argument("--scene_cache", type=Path, default=ROOT / ".cache" / "scene_collision")
    parser.add_argument("--object_asset", type=Path, default=None,
                        help="Optional resolved USD/OBJ asset when metadata object_path is dataset-relative")
    known, _ = parser.parse_known_args()
    with known.motion.open("rb") as stream:
        record = pickle.load(stream)
    if not record.get("object_path") and not record.get("obj_data"):
        raise ValueError("WholeBody V4 requires scene metadata for GRAIL input")
    asset = _asset_path(record, known.motion, known.object_asset)
    pose = _object_pose(record)
    # The robot solver uses the same height normalization as the GRAIL
    # adapter.  Apply that transform to the object pose too; otherwise the
    # chair remains in source coordinates while the robot is scaled/translated.
    raw_config = json.loads((ROOT / "general_motion_retargeting/ik_configs/holosoma_to_ne01_wholebody_omni_gmr_v4.json").read_text())
    parent = json.loads((ROOT / "general_motion_retargeting/ik_configs" / raw_config["extends"]).read_text())
    scene_cfg = {**parent.get("scene", {}), **raw_config.get("scene", {})}
    human = record["human_data"]
    tmp = work_tmp = known.save_path.parent / f".{known.save_path.stem}_height.npz"
    work_tmp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(tmp, root_orient=np.asarray(human["poses"], dtype=np.float32)[:, :3], pose_body=np.asarray(human["poses"], dtype=np.float32)[:, 3:66], trans=np.asarray(human["trans"], dtype=np.float32), betas=np.asarray(human.get("betas", np.zeros(10)), dtype=np.float32).reshape(-1)[:10], gender=np.asarray(str(human.get("gender", "neutral"))), mocap_frame_rate=np.asarray(float(human.get("mocap_frame_rate", 50.0))))
    try:
        from general_motion_retargeting.utils.smpl import load_smplx_file
        _, _, _, human_height = load_smplx_file(tmp, ROOT / "assets" / "body_models")
    finally:
        tmp.unlink(missing_ok=True)
    from general_motion_retargeting.terrain_geometry import SceneTransform
    scene_scale_multiplier = float(scene_cfg.get("scene_scale", 1.0))
    raw_reference = scene_cfg.get("source_reference_height")
    reference_height = human_height if raw_reference is None else float(raw_reference)
    resolved_scale = (scene_scale_multiplier
                      * float(scene_cfg.get("robot_height", 1.316))
                      / max(reference_height, 1e-9))
    scene_scale = resolved_scale
    scene_transform = SceneTransform(np.asarray(scene_cfg.get("rotation", np.eye(3))), scene_scale, np.asarray(scene_cfg.get("translation", [0, 0, 0])))
    scene_pose = np.eye(4)
    # Motion and object originate in the same GRAIL reconstruction world, so
    # the complete similarity transform must be shared by both.  This keeps
    # visual, interaction and collision geometry metrically aligned.
    scene_pose[:3, :3] = scene_transform.scale * scene_transform.rotation @ pose[:3, :3]
    # Object pose is a static reconstruction-world pose.  Human root
    # translation changes through the sequence (approach, sit, stand) and
    # must not be baked into the object transform.
    scene_pose[:3, 3] = scene_transform.transform_points(pose[:3, 3])
    scene_mesh = load_scene_asset(asset, {
        "object_id": asset.stem,
        # GRAIL's generated object_usd stores the reconstructed object mesh;
        # obj_R/obj_t/obj_scale from metadata are the sole logical pose.
        "asset_space": str(record.get("asset_space", "object_local")),
        "asset_scale_baked": bool(record.get("asset_scale_baked", False)),
    })
    scene_mesh.object_pose = scene_pose
    scene = scene_mesh.to_scene_geometry(sample_count=4096)
    manifest, cache_dir = decompose_cached(scene, asset, cache_root=known.scene_cache)
    scene_spec = {
        "objects": [{
            "object_id": scene.objects[0].object_id,
            "pose": pose.tolist(),
            "collision": {"type": "convex_decomposition", "manifest": str(cache_dir / "collision_manifest.json")},
            "visual": {"path": str(asset)},
        }],
    }
    work = known.save_path.parent
    work.mkdir(parents=True, exist_ok=True)
    spec_path = work / f".{known.save_path.stem}_scene.json"
    spec_path.write_text(json.dumps(scene_spec, indent=2))
    config_path = ROOT / "general_motion_retargeting/ik_configs/holosoma_to_ne01_wholebody_omni_gmr_v4.json"
    raw_config = json.loads(config_path.read_text())
    parent_config = json.loads((config_path.parent / raw_config["extends"]).read_text()) if raw_config.get("extends") else {}
    base = _deep_merge(parent_config, {k: v for k, v in raw_config.items() if k != "extends"})
    robot_xml = Path(base["robot_xml"])
    if not robot_xml.is_absolute():
        robot_xml = (config_path.parents[2] / robot_xml).resolve()
    combined_xml = work / f".{known.save_path.stem}_combined.xml"
    combined_info = build_scene_model(
        robot_xml, scene_mesh, combined_xml, cache_root=known.scene_cache,
        return_info=True,
    )
    effective = copy.deepcopy(base)
    effective["robot_xml"] = str(combined_xml)
    effective.setdefault("scene", {})["scene_scale"] = resolved_scale
    effective["scene"].pop("source_reference_height", None)
    effective["scene_collision"] = {**effective.get("scene_collision", {}), "backend": "mujoco", "scene_body_prefix": "scene_"}
    effective_path = work / f".{known.save_path.stem}_v4_config.json"
    effective_path.write_text(json.dumps(effective, indent=2))

    # Feed visual scene samples into the Omni interaction pool as well as the
    # collision model.  The adapter's terrain sampler remains the source of
    # floor samples; object samples are deterministic and never replace human
    # semantic targets.
    original_pool_builder = impl.sample_terrain_surface_pool
    # Keep the V3 semantic interaction pool unchanged until a source-contact
    # channel explicitly selects chair surface samples.  Feeding every chair
    # vertex to the global Laplacian makes unrelated leg targets compete with
    # the human pose and was the source of the V4 limb distortion.
    impl.RETARGETER_CLASS = WholeBodyOmniGMRV4
    impl.DEFAULT_CONFIG = effective_path
    def _object_contact_schedule(schedule, source_frames, config):
        object_id = scene.objects[0].object_id
        threshold = float(config.get("terrain_contact", {}).get("object_contact_distance", 0.05))
        static_speed = float(config.get("terrain_contact", {}).get("static_tangent_speed", 0.08))
        # Keep contact normals tied to the same source mesh used by visual and
        # collision construction.  Centroid lookup is deterministic and
        # avoids hard-coding +Z for a vertical chair back.
        triangles = scene_mesh.vertices[scene_mesh.faces]
        centers = np.mean(triangles, axis=1)
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        linear = scene_mesh.object_pose[:3, :3]
        transformed_centers = (np.c_[centers, np.ones(len(centers))] @ scene_mesh.object_pose.T)[:, :3]
        transformed_triangles = (np.c_[triangles.reshape(-1, 3), np.ones(len(triangles) * 3)] @ scene_mesh.object_pose.T)[:, :3].reshape((-1, 3, 3))
        transformed_normals = normals @ np.linalg.inv(linear)
        transformed_normals /= np.maximum(np.linalg.norm(transformed_normals, axis=1, keepdims=True), 1e-12)
        for frame, record_frame in zip(source_frames, schedule):
            # Feet use only upward-facing mesh faces as support surfaces;
            # knees, shins, hands and torso may contact any nearby face.
            for channel in (
                "left_heel", "right_heel", "left_toe", "right_toe",
                "left_palm", "right_palm", "left_knee", "right_knee",
                "left_shin", "right_shin", "left_butt", "right_butt",
                "lower_back", "upper_back",
            ):
                item = record_frame.get("contacts", {}).get(channel)
                if not item or not np.all(np.isfinite(item.get("human_point_solver", [np.nan] * 3))):
                    continue
                point = np.asarray(item["human_point_solver"], dtype=float)
                if channel.endswith(("heel", "toe")):
                    support = np.flatnonzero(transformed_normals[:, 2] > 0.6)
                    if len(support) == 0:
                        continue
                    index, surface_point, normal, distance = _closest_mesh_surface(
                        point, transformed_triangles[support], transformed_centers[support],
                        transformed_normals[support], candidate_count=64,
                    )
                    index = int(support[index])
                else:
                    index, surface_point, normal, distance = _closest_mesh_surface(
                        point, transformed_triangles, transformed_centers,
                        transformed_normals, candidate_count=64,
                    )
                if distance > threshold:
                    continue
                score = float(np.clip(1.0 - distance / max(threshold, 1e-9), 0.0, 1.0))
                tangent_speed = float(item.get("tangential_speed", 0.0))
                state = "STATIC" if tangent_speed < static_speed else "SLIDING"
                item.update({
                    "score": max(float(item.get("score", 0.0)), score),
                    "state": state,
                    "source_state": state,
                    "object_id": object_id,
                    "surface_id": f"{object_id}:face_{index:05d}",
                    "surface_type": "mesh",
                    "surface_point_solver": surface_point.copy(),
                    "surface_normal_solver": normal.copy(),
                    "signed_distance": distance,
                    "normal_error": distance,
                    "tangent_error": tangent_speed,
                })
        return schedule
    impl.SCENE_CONTACT_POSTPROCESS = _object_contact_schedule
    try:
        # The shared adapter intentionally knows only its stable public CLI.
        # Consume V4-only preprocessing flags before delegating.
        old_argv = sys.argv[:]
        sys.argv = [sys.argv[0]]
        skip = False
        for token in old_argv[1:]:
            if skip:
                skip = False
                continue
            if token == "--scene_cache":
                skip = True
                continue
            if token == "--object_asset":
                skip = True
                continue
            sys.argv.append(token)
        impl.main()
        # The shared adapter normalizes robot XY to the first frame. Keep the
        # visual/collision scene in exactly the same frame before viewing.
        out_pkl = known.save_path.with_suffix(".pkl")
        if out_pkl.is_file():
            with out_pkl.open("rb") as stream:
                payload = pickle.load(stream)
            from general_motion_retargeting.scene_diagnostics import summarize_scene_diagnostics
            summary = summarize_scene_diagnostics(payload.get("terrain_diagnostics", []), payload.get("contact_schedule", []))
            summary["chair_asset"] = str(asset)
            summary["interaction_scene_points"] = int(len(scene.objects[0].surface_samples))
            summary["mujoco_scene_geom_count"] = int(len(getattr(combined_info, "scene_geom_ids", ())))
            summary["coacd_piece_count"] = int(len(manifest.get("pieces", [])))
            summary["scene_scale"] = float(scene_transform.scale)
            summary["human_height"] = float(human_height)
            summary["reference_height"] = float(reference_height)
            summary["scene_scale_multiplier"] = float(scene_scale_multiplier)
            summary["resolved_scale"] = float(resolved_scale)
            summary["obj_scale"] = np.asarray(record.get("obj_data", {}).get("obj_scale", [1, 1, 1]), dtype=float).reshape(-1).tolist()
            summary["asset_space"] = str(scene_mesh.metadata.get("asset_space", "unknown"))
            summary["asset_scale_baked"] = bool(scene_mesh.metadata.get("asset_scale_baked", False))
            summary["raw_object_aabb"] = {"min": np.min(scene_mesh.vertices, axis=0).tolist(), "max": np.max(scene_mesh.vertices, axis=0).tolist()}
            final_vertices = (np.c_[scene_mesh.vertices, np.ones(len(scene_mesh.vertices))] @ scene_mesh.object_pose.T)[:, :3]
            summary["final_object_aabb"] = {"min": np.min(final_vertices, axis=0).tolist(), "max": np.max(final_vertices, axis=0).tolist()}
            out_pkl.with_name(out_pkl.stem + ".scene_summary.json").write_text(json.dumps(summary, indent=2))
            q = np.asarray(payload.get("qpos"), dtype=float)
            if q.ndim == 2 and len(q):
                import xml.etree.ElementTree as ET
                xml_path = Path(payload["robot_xml"])
                root_xml = ET.parse(xml_path).getroot()
                # Preserve the source-world origin.  Applying an XY shift to
                # qpos alone detaches the chair from the robot and invalidates
                # interaction/collision alignment.
                shift = np.zeros(2, dtype=float)
                for body in root_xml.findall("./worldbody/body"):
                    if str(body.get("name", "")).startswith("scene_"):
                        pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
                        if pos.size == 3:
                            pos[:2] -= shift
                            body.set("pos", " ".join(f"{v:.12g}" for v in pos))
                ET.indent(root_xml, space="  ")
                ET.ElementTree(root_xml).write(xml_path, encoding="utf-8", xml_declaration=True)
    finally:
        sys.argv = old_argv
        impl.sample_terrain_surface_pool = original_pool_builder
        impl.RETARGETER_CLASS = impl.WholeBodyOmniGMRV3
        impl.SCENE_CONTACT_POSTPROCESS = None
        # Keep combined XML/spec/cache for reproducibility and visualization;
        # only the temporary effective config is disposable.
        effective_path.unlink(missing_ok=True)
    print(f"Scene asset: {asset}")
    print(f"Convex pieces: {len(manifest.get('pieces', []))}; cache: {cache_dir}")
    if hasattr(combined_info, "visual_geom_count"):
        print(f"MuJoCo scene visual geoms: {combined_info.visual_geom_count}; collision geoms: {combined_info.collision_geom_count}")
    print(f"Combined MuJoCo model: {combined_xml}")


if __name__ == "__main__":
    main()
