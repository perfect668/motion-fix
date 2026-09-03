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


def _asset_path(record: dict, motion: Path, override: Path | None = None) -> Path:
    if override is not None:
        if not override.is_file():
            raise FileNotFoundError(f"GRAIL object asset does not exist: {override}")
        return override.resolve()
    raw = str(record.get("object_path", ""))
    candidates = []
    if raw:
        candidates.append(Path(raw))
        candidates.append(Path("/home/user/datasets/grail_200/data") / motion.parent.parent.name / "object_usd" / (motion.stem + ".usd"))
    # GRAIL reconstruction names are also used by the generated USD assets.
    candidates.extend([
        motion.parent.parent / "object_usd" / f"{motion.stem}.usd",
        Path("/home/user/datasets/grail_200/data/sitting/object_usd") / f"{motion.stem}.usd",
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "GRAIL metadata declares an object, but no mesh/USD asset was found. "
        f"object_path={raw!r}; checked {len(candidates)} paths"
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
    scene_transform = SceneTransform(np.asarray(scene_cfg.get("rotation", np.eye(3))), float(scene_cfg["robot_height"]) / float(human_height), np.asarray(scene_cfg.get("translation", [0, 0, 0])))
    scene_pose = np.eye(4)
    # GRAIL USD assets are already expressed in metric object units; their
    # recorded obj_scale is the object-size calibration.  Applying the human
    # height scale a second time shrinks the chair by ~0.79 and caused the
    # visibly undersized scene.  Rotate/translate into solver coordinates,
    # while preserving the metadata object scale.
    scene_pose[:3, :3] = scene_transform.rotation @ pose[:3, :3]
    # Object pose is a static reconstruction-world pose.  Human root
    # translation changes through the sequence (approach, sit, stand) and
    # must not be baked into the object transform.
    source_root = np.asarray(human["trans"], dtype=float)[0]
    scene_pose[:3, 3] = scene_transform.transform_points(pose[:3, 3])
    scene_mesh = load_scene_asset(asset, {"object_id": asset.stem})
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
    impl.WholeBodyOmniGMRV3 = WholeBodyOmniGMRV4
    impl.DEFAULT_CONFIG = effective_path
    def _object_contact_schedule(schedule, source_frames, config):
        samples = scene.objects[0].transformed_samples()
        object_id = scene.objects[0].object_id
        threshold = float(config.get("terrain_contact", {}).get("object_contact_distance", 0.18))
        for frame, record_frame in zip(source_frames, schedule):
            for channel in ("left_butt", "right_butt", "lower_back", "upper_back", "left_palm", "right_palm"):
                item = record_frame.get("contacts", {}).get(channel)
                if not item or not np.all(np.isfinite(item.get("human_point_solver", [np.nan] * 3))):
                    continue
                point = np.asarray(item["human_point_solver"], dtype=float)
                index = int(np.argmin(np.linalg.norm(samples - point[None, :], axis=1)))
                distance = float(np.linalg.norm(samples[index] - point))
                if distance > threshold:
                    continue
                item.update({"score": float(np.clip(1.0 - distance / threshold, 0.0, 1.0)), "state": "STATIC", "source_state": "STATIC", "object_id": object_id, "surface_id": f"{object_id}:sample_{index:04d}", "surface_type": "mesh", "surface_point_solver": samples[index].copy(), "surface_normal_solver": np.array([0., 0., 1.]), "signed_distance": distance, "normal_error": distance, "tangent_error": float(item.get("tangential_speed", 0.0))})
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
            out_pkl.with_name(out_pkl.stem + ".scene_summary.json").write_text(json.dumps(summary, indent=2))
            q = np.asarray(payload.get("qpos"), dtype=float)
            if q.ndim == 2 and len(q):
                import xml.etree.ElementTree as ET
                xml_path = Path(payload["robot_xml"])
                root_xml = ET.parse(xml_path).getroot()
                # Object metadata and the reconstructed human already share
                # GRAIL's scene origin.  qpos XY normalization is a robot
                # trajectory convention, not a source-world translation;
                # applying human_data.trans here would move the chair away.
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
