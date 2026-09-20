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


def _load_config(path: Path) -> dict:
    """Resolve nested ``extends`` chains for standalone GRAIL configs."""
    raw = json.loads(path.read_text())
    parent = _load_config(path.parent / raw["extends"]) if raw.get("extends") else {}
    return _deep_merge(parent, {key: value for key, value in raw.items() if key != "extends"})

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


def _highest_support_surface(point: np.ndarray, triangles: np.ndarray, centers: np.ndarray,
                             normals: np.ndarray, min_normal_z: float = 0.6):
    """Select the highest upward-facing surface below a foot projection.

    This prevents a nearby vertical riser from winning a closest-Euclidean
    query at stair edges.  A small horizontal fallback keeps the query robust
    for sparse meshes while retaining deterministic ordering.
    """
    upward = np.flatnonzero(normals[:, 2] > min_normal_z)
    if len(upward) == 0:
        return None
    px, py, pz = np.asarray(point, dtype=float)
    containing = []
    for index in upward:
        tri = triangles[int(index)]
        xy = tri[:, :2]
        # Barycentric containment in the projected triangle.
        a, b, c = xy
        v0, v1, v2 = c - a, b - a, np.array([px, py]) - a
        den = float(v0[0] * v1[1] - v1[0] * v0[1])
        if abs(den) < 1e-12:
            continue
        u = float((v2[0] * v1[1] - v1[0] * v2[1]) / den)
        v = float((v0[0] * v2[1] - v2[0] * v0[1]) / den)
        if u >= -1e-8 and v >= -1e-8 and u + v <= 1.0 + 1e-8:
            containing.append(int(index))
    candidates = containing or [int(index) for index in upward]
    if containing:
        candidates.sort(key=lambda index: (-float(centers[index, 2]), index))
    else:
        candidates.sort(key=lambda index: (float(np.sum((centers[index, :2] - [px, py]) ** 2)), -float(centers[index, 2]), index))
    index = candidates[0]
    surface = _closest_point_triangle(np.asarray(point, dtype=float), triangles[index])
    normal = normals[index].copy()
    if normal[2] < 0.0:
        normal = -normal
    signed = float(normal @ (np.asarray(point, dtype=float) - surface))
    return index, surface, normal, signed


def _triangle_barycentric(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Return deterministic barycentric coordinates for a surface anchor."""
    a, b, c = np.asarray(triangle, dtype=float)
    v0, v1, v2 = b - a, c - a, np.asarray(point, dtype=float) - a
    d00, d01, d11 = float(v0 @ v0), float(v0 @ v1), float(v1 @ v1)
    d20, d21 = float(v2 @ v0), float(v2 @ v1)
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    return np.array([1.0 - v - w, v, w], dtype=float)


def _asset_path(record: dict, motion: Path, override: Path | None = None) -> Path:
    if override is not None:
        if not override.is_file():
            raise FileNotFoundError(f"GRAIL object asset does not exist: {override}")
        return override.resolve()
    raw = str(record.get("object_path", ""))
    candidates = [
        # Processed GRAIL exports are preferred because their dimensions are
        # already baked into the mesh.
        motion.parent / "mesh_data" / "model.obj",
        motion.parent.parent / "mesh_data" / "model.obj",
        motion.parent.parent.parent / "mesh_data" / "model.obj",
    ]
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


def _object_pose(record: dict, *, apply_scale: bool = True) -> np.ndarray:
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
    pose[:3, :3] = R @ np.diag(scale if apply_scale else np.ones(3))
    pose[:3, 3] = t
    return pose


def main() -> None:
    import grail_to_robot_wholebody_v3 as impl
    from general_motion_retargeting.scene_asset_loader import decompose_cached, load_scene_asset
    from general_motion_retargeting.scene_mujoco import build_scene_model
    from general_motion_retargeting.asset_interaction_transform import AssetInteractionTransform

    # Parse only enough metadata to construct the combined MuJoCo model.  The
    # validated V3 adapter remains responsible for SMPL-X conversion/output.
    parser = impl.argparse.ArgumentParser(add_help=False)
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--save_path", required=True, type=Path)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "general_motion_retargeting/ik_configs/holosoma_to_ne01_wholebody_omni_gmr_v4.json")
    parser.add_argument("--scene_cache", type=Path, default=ROOT / ".cache" / "scene_collision")
    parser.add_argument("--object_asset", type=Path, default=None,
                        help="Optional resolved USD/OBJ asset when metadata object_path is dataset-relative")
    parser.add_argument("--tgt_fps", type=float, default=50.0)
    known, _ = parser.parse_known_args()
    with known.motion.open("rb") as stream:
        record = pickle.load(stream)
    if not record.get("object_path") and not record.get("obj_data"):
        raise ValueError("WholeBody V4 requires scene metadata for GRAIL input")
    asset = _asset_path(record, known.motion, known.object_asset)
    explicit_baked = record.get("asset_scale_baked")
    asset_scale_baked = bool(explicit_baked) if explicit_baked is not None else (
        asset.as_posix().replace("\\", "/").endswith("mesh_data/model.obj")
    )
    pose = _object_pose(record, apply_scale=not asset_scale_baked)
    # The robot solver uses the same height normalization as the GRAIL
    # adapter.  Apply that transform to the object pose too; otherwise the
    # chair remains in source coordinates while the robot is scaled/translated.
    config_path = known.config.expanduser().resolve()
    raw_config = json.loads(config_path.read_text())
    merged_config = _load_config(config_path)
    scene_cfg = merged_config.get("scene", {})
    human = record["human_data"]
    tmp = work_tmp = known.save_path.parent / f".{known.save_path.stem}_height.npz"
    work_tmp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(tmp, root_orient=np.asarray(human["poses"], dtype=np.float32)[:, :3], pose_body=np.asarray(human["poses"], dtype=np.float32)[:, 3:66], trans=np.asarray(human["trans"], dtype=np.float32), betas=np.asarray(human.get("betas", np.zeros(10)), dtype=np.float32).reshape(-1)[:10], gender=np.asarray(str(human.get("gender", "neutral"))), mocap_frame_rate=np.asarray(float(human.get("mocap_frame_rate", 50.0))))
    human_fk_frames = None
    try:
        from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
        smplx_data, smplx_model, smplx_output, human_height = load_smplx_file(
            tmp, ROOT / "assets" / "body_models"
        )
        human_fk_frames, _ = get_smplx_data_offline_fast(
            smplx_data, smplx_model, smplx_output, tgt_fps=50.0
        )
    finally:
        tmp.unlink(missing_ok=True)
    from general_motion_retargeting.terrain_geometry import SceneTransform
    # GRAIL reconstruction human and object already share a metric scene
    # world.  Do not reuse the ordinary SMPL-X robot-height normalization for
    # the whole scene; it would scale the chair/stairs a second time.
    scene_scale_multiplier = float(scene_cfg.get("human_scene_scale", 1.0))
    resolved_scale = scene_scale_multiplier
    reference_height = human_height
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
        "asset_scale_baked": asset_scale_baked,
    })
    adaptation_cfg = scene_cfg.get("asset_interaction_transform", {})
    asset_adaptation = AssetInteractionTransform.from_config(adaptation_cfg)
    adaptation_objective_before = None
    adaptation_objective_after = None
    adaptation_contact_point_count = 0
    if bool(adaptation_cfg.get("enabled", False)):
        if bool(adaptation_cfg.get("auto", False)) and human_fk_frames:
            from general_motion_retargeting.terrain_contact_utils import (
                _mesh_contact_hit,
            )
            # Use only source body proxies that are already near this asset;
            # otherwise a free-standing action cannot drag the object toward
            # an unrelated body part. Points are transformed once into the
            # solver frame, matching the mesh pose below.
            source_points = []
            for fk_frame in human_fk_frames:
                points = {name: np.asarray(value[0], dtype=float) for name, value in fk_frame.items()}
                for side in ("left", "right"):
                    if f"{side}_heel" in points:
                        source_points.append((points[f"{side}_heel"], 1.0))
                    if f"{side}_toe" in points:
                        source_points.append((points[f"{side}_toe"], 1.0))
                    for name in (f"{side}_wrist", f"{side}_knee"):
                        if name in points:
                            source_points.append((points[name], 2.0 if "wrist" in name else 1.0))
                if "pelvis" in points:
                    source_points.append((points["pelvis"], 4.0))
                if "spine3" in points:
                    source_points.append((points["spine3"], 3.0))
            source_points = [
                (scene_transform.transform_points(point), weight)
                for point, weight in source_points
            ]
            base_vertices = scene_mesh.vertices.copy()
            base_triangles = base_vertices[scene_mesh.faces]
            base_centers = base_triangles.mean(axis=1)
            base_normals = np.cross(base_triangles[:, 1] - base_triangles[:, 0], base_triangles[:, 2] - base_triangles[:, 0])
            base_normals /= np.maximum(np.linalg.norm(base_normals, axis=1, keepdims=True), 1e-12)
            base_pose = scene_pose.copy()
            proximity = float(adaptation_cfg.get("contact_probe_distance", 0.12))
            base_world_vertices = (np.c_[base_vertices, np.ones(len(base_vertices))] @ base_pose.T)[:, :3]
            base_world_triangles = base_world_vertices[scene_mesh.faces]
            base_world_centers = base_world_triangles.mean(axis=1)
            base_world_normals = np.cross(
                base_world_triangles[:, 1] - base_world_triangles[:, 0],
                base_world_triangles[:, 2] - base_world_triangles[:, 0],
            )
            base_world_normals /= np.maximum(np.linalg.norm(base_world_normals, axis=1, keepdims=True), 1e-12)
            contact_points = []
            for point, weight in source_points:
                hit = _mesh_contact_hit(point, base_world_triangles, base_world_centers, base_world_normals, False, 0.6)
                if hit is not None and abs(float(hit[3])) <= proximity:
                    contact_points.append((point, weight))

            def adaptation_objective(candidate):
                # Bake the candidate linear transform into the object-local
                # mesh. Keeping the scene body pose rigid avoids introducing
                # shear when the metadata already contains anisotropic scale.
                candidate_vertices = base_vertices @ candidate.linear.T
                candidate_pose = base_pose.copy()
                candidate_pose[:3, 3] += candidate.translation
                transformed = (np.c_[candidate_vertices, np.ones(len(candidate_vertices))] @ candidate_pose.T)[:, :3]
                triangles = transformed[scene_mesh.faces]
                centers = triangles.mean(axis=1)
                normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
                distances = []
                weighted_distances = []
                for point, weight in contact_points:
                    hit = _mesh_contact_hit(point, triangles, centers, normals, False, 0.6)
                    weighted_distances.append((abs(float(hit[3])) if hit is not None else proximity, weight))
                if not weighted_distances:
                    return 0.0
                values = np.asarray([min(distance, proximity) ** 2 for distance, _ in weighted_distances])
                weights = np.asarray([weight for _, weight in weighted_distances])
                value = float(np.sum(values * weights) / max(np.sum(weights), 1e-9))
                value += 2.0 * float(np.sum(candidate.translation ** 2))
                value += 0.05 * float(candidate.yaw ** 2)
                value += 0.5 * float(np.sum(np.array([
                    candidate.scale_axis_1 - 1.0,
                    candidate.scale_axis_2 - 1.0,
                    candidate.scale_up - 1.0,
                ]) ** 2))
                return value

            adaptation_objective_before = float(adaptation_objective(asset_adaptation))
            asset_adaptation, _ = asset_adaptation.optimize(
                adaptation_objective,
                max_iterations=int(adaptation_cfg.get("max_outer_iterations", 3)),
                translation_step=float(adaptation_cfg.get("translation_step", 0.02)),
                yaw_step=float(adaptation_cfg.get("yaw_step", 0.05)),
                scale_step=float(adaptation_cfg.get("scale_step", 0.03)),
                scale_min=float(adaptation_cfg.get("scale_min", 0.85)),
                scale_max=float(adaptation_cfg.get("scale_max", 1.15)),
            )
            adaptation_objective_after = float(adaptation_objective(asset_adaptation))
            adaptation_contact_point_count = len(contact_points)
        if bool(adaptation_cfg.get("auto", False)) or any(
            abs(value - 1.0) > 1e-12
            for value in (asset_adaptation.scale_axis_1, asset_adaptation.scale_axis_2, asset_adaptation.scale_up)
        ) or abs(asset_adaptation.yaw) > 1e-12:
            # Apply non-rigid asset adaptation to the mesh itself; MuJoCo's
            # body pose remains a valid rotation/translation representation.
            scene_mesh.vertices = scene_mesh.vertices @ asset_adaptation.linear.T
        scene_pose[:3, 3] += asset_adaptation.translation
    scene_mesh.object_pose = scene_pose
    scene = scene_mesh.to_scene_geometry(sample_count=4096)
    manifest, cache_dir = decompose_cached(scene, asset, cache_root=known.scene_cache)
    scene_spec = {
        "objects": [{
            "object_id": scene.objects[0].object_id,
            "pose": scene_pose.tolist(),
            "collision": {"type": "convex_decomposition", "manifest": str(cache_dir / "collision_manifest.json")},
            "visual": {"path": str(asset)},
        }],
    }
    work = known.save_path.parent
    work.mkdir(parents=True, exist_ok=True)
    spec_path = work / f".{known.save_path.stem}_scene.json"
    spec_path.write_text(json.dumps(scene_spec, indent=2))
    config_path = known.config.expanduser().resolve()
    base = _load_config(config_path)
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

    from general_motion_retargeting.terrain_native_geometry import TerrainPatchMap
    from general_motion_retargeting.terrain_native_planner import apply_terrain_native_plan
    from general_motion_retargeting.wholebody_terrain_native import TerrainNativeRetargeter

    native_cfg = effective.get("terrain_native", {})
    patch_map = TerrainPatchMap.from_mesh(
        scene_mesh.vertices,
        scene_mesh.faces,
        scene_mesh.object_pose,
        scene.objects[0].object_id,
        native_cfg.get("support_patches", {}),
    )

    floor_z = float(scene_transform.transform_points(np.array([0.0, 0.0, 0.0]))[2])
    floor_refs = [patch_map.vertices]
    if human_fk_frames:
        tracked = []
        for frame in human_fk_frames:
            for name in ("pelvis", "left_heel", "right_heel", "left_toe", "right_toe"):
                if name in frame:
                    tracked.append(
                        scene_transform.transform_points(
                            np.asarray(frame[name][0], dtype=float)
                        )
                    )
        if tracked:
            floor_refs.append(np.asarray(tracked, dtype=float))
    floor_extent_points = np.vstack(floor_refs)
    floor_margin = 1.0
    patch_map.add_horizontal_patch(
        "floor",
        floor_z,
        floor_extent_points[:, :2].min(axis=0) - floor_margin,
        floor_extent_points[:, :2].max(axis=0) + floor_margin,
    )

    # Feed visual scene samples into the Omni interaction pool as well as the
    # collision model.  The adapter's terrain sampler remains the source of
    # floor samples.  Only object samples in the tracked human-proximity
    # envelope are added to the pool; remote parts of a complex asset must not
    # pull unrelated limbs through the Laplacian task.
    original_pool_builder = impl.sample_terrain_surface_pool
    object_samples = scene.objects[0].transformed_samples()
    scene_sample_limit = max(
        0,
        int(
            base.get("scene_geometry", {})
            .get("scene_surface_points", 512)
        ),
    )

    def _near_human_scene_samples(references: np.ndarray) -> np.ndarray:
        if scene_sample_limit == 0:
            return np.empty((0, 3), dtype=float)
        references = np.asarray(references, dtype=float).reshape((-1, 3))
        if len(references) == 0:
            return np.empty((0, 3), dtype=float)

        # Climbing is support-surface dominated. Reserve most scene samples
        # for connected upward patches, as HoloSoMo does for climbing, then
        # keep generic nearby samples so risers and edges remain represented.
        support_count = max(1, int(round(scene_sample_limit * 0.75)))
        support_pool = patch_map.interaction_samples(references, support_count)
        remaining = max(0, scene_sample_limit - len(support_pool))
        if remaining == 0 or len(object_samples) == 0:
            return support_pool
        nearest = np.min(
            np.sum((object_samples[:, None, :] - references[None, :, :]) ** 2, axis=-1),
            axis=1,
        )
        selected = np.argsort(nearest, kind="stable")[: min(remaining, len(object_samples))]
        general_pool = object_samples[selected].copy()
        return np.vstack((support_pool, general_pool))

    def _interaction_pool(terrain, references, **kwargs):
        terrain_pool = np.asarray(
            original_pool_builder(terrain, references, **kwargs), dtype=float
        ).reshape((-1, 3))
        scene_pool = _near_human_scene_samples(references)
        if len(scene_pool) == 0:
            return terrain_pool
        return np.vstack((terrain_pool, scene_pool))

    impl.sample_terrain_surface_pool = _interaction_pool
    # The pool now contains terrain samples plus a bounded, human-proximal
    # subset of the object surface.  Remote object geometry is excluded so it
    # cannot pull unrelated limbs through the Laplacian task.
    class _GrailTerrainNativeRetargeter(TerrainNativeRetargeter):
        """Terrain-native solver with scene interaction provenance."""

        def __init__(self, config_path, terrain, environment_pool, fps=50.0, solver="daqp"):
            super().__init__(
                config_path, terrain, environment_pool,
                patch_map=patch_map, fps=fps, solver=solver,
            )

        def retarget(self, *args, **kwargs):
            output = super().retarget(*args, **kwargs)
            selected = np.asarray(self.interaction_task.environment, dtype=float).reshape((-1, 3))
            patch_centers = np.asarray(
                [patch.center for patch in patch_map.patches], dtype=float
            ).reshape((-1, 3))
            catalogs = [patch_map.vertices, patch_centers]
            if len(object_samples):
                catalogs.append(object_samples)
            patch_catalog = np.vstack(catalogs)
            if len(selected) and len(patch_catalog):
                distances = np.min(
                    np.sum((selected[:, None, :] - patch_catalog[None, :, :]) ** 2, axis=-1),
                    axis=1,
                )
                scene_count = int(np.count_nonzero(distances <= 1e-14))
            else:
                scene_count = 0
            self.diagnostics[-1]["interaction_scene_selected_points"] = scene_count
            self.diagnostics[-1]["interaction_terrain_selected_points"] = int(len(selected) - scene_count)
            return output

    impl.RETARGETER_CLASS = _GrailTerrainNativeRetargeter
    impl.DEFAULT_CONFIG = effective_path
    from general_motion_retargeting.terrain_contact_utils import augment_mesh_contact_schedule
    def _mesh_contact_provider(schedule, source_frames, config):
        # Generic non-foot scene contacts remain available, but feet are owned
        # exclusively by the sequence-level terrain-native planner.
        mesh_cfg = copy.deepcopy(config.get("terrain_contact", {}))
        mesh_cfg["channels"] = [
            name for name in mesh_cfg.get("channels", ())
            if not name.endswith(("heel", "toe"))
        ]
        schedule = augment_mesh_contact_schedule(
            schedule, source_frames, scene_mesh.vertices, scene_mesh.faces,
            scene.objects[0].object_id, scene_mesh.object_pose, mesh_cfg,
        )
        planner_cfg = config.get("terrain_native", {}).get("planner", {})
        schedule = apply_terrain_native_plan(
            schedule,
            source_frames,
            patch_map,
            float(known.tgt_fps),
            planner_cfg,
        )
        if bool(planner_cfg.get("require_nonfloor_support", False)):
            support_ids = {
                str(frame.get("terrain_native", {}).get(side, {}).get("patch_id", ""))
                for frame in schedule
                for side in ("left", "right")
                if frame.get("terrain_native", {}).get(side, {}).get("mode") == "stance"
            }
            support_ids.discard("")
            support_ids.discard("floor")
            if not support_ids:
                raise RuntimeError(
                    "Terrain-native planner found no non-floor foot support episode. "
                    "Check source/scene alignment instead of falling back to reactive "
                    "robot-side stair contact."
                )
        return schedule
    impl.CONTACT_SURFACE_PROVIDER = _mesh_contact_provider
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
            if token == "--config":
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
            # Keep both the descriptive diagnostic names and the compact names
            # consumed by older regression tooling.  These values come from
            # the final exported payload, never from the pre-solve state.
            summary["fps"] = float(payload.get("fps", 50.0) or 50.0)
            summary["frame_count"] = int(len(payload.get("qpos", [])))
            summary["chair_asset"] = str(asset)
            summary["interaction_scene_points"] = int(len(scene.objects[0].surface_samples))
            summary["mujoco_scene_geom_count"] = int(len(getattr(combined_info, "scene_geom_ids", ())))
            summary["mujoco_scene_collision_geom_count"] = int(
                getattr(combined_info, "collision_geom_count", 0)
            )
            summary["coacd_piece_count"] = int(len(manifest.get("pieces", [])))
            summary["collision_geoms"] = summary["mujoco_scene_geom_count"]
            summary["convex_pieces"] = summary["coacd_piece_count"]
            summary["terrain_native_patch_map"] = patch_map.summary()
            summary["terrain_native_enabled"] = True
            summary["scene_scale"] = float(scene_transform.scale)
            summary["human_height"] = float(human_height)
            summary["reference_height"] = float(reference_height)
            summary["scene_scale_multiplier"] = float(scene_scale_multiplier)
            summary["resolved_scale"] = float(resolved_scale)
            summary["asset_interaction_transform"] = asset_adaptation.to_dict()
            summary["asset_adaptation_objective_before"] = adaptation_objective_before
            summary["asset_adaptation_objective_after"] = adaptation_objective_after
            summary["asset_adaptation_contact_point_count"] = int(adaptation_contact_point_count)
            summary["obj_scale"] = np.asarray(record.get("obj_data", {}).get("obj_scale", [1, 1, 1]), dtype=float).reshape(-1).tolist()
            summary["asset_space"] = str(scene_mesh.metadata.get("asset_space", "unknown"))
            summary["asset_scale_baked"] = bool(scene_mesh.metadata.get("asset_scale_baked", False))
            # Explicit seated-sequence aliases make it clear that these are
            # distances to the inferred scene surface, not distances to the
            # robot pelvis joint.
            for channel in ("left_butt", "right_butt", "lower_back", "upper_back"):
                summary[f"median_{channel}_seat_distance"] = summary.get(
                    f"median_{channel}_object_distance", float("inf")
                )
                summary[f"max_{channel}_seat_distance"] = summary.get(
                    f"max_{channel}_object_distance", float("inf")
                )
            summary["raw_object_aabb"] = {"min": np.min(scene_mesh.vertices, axis=0).tolist(), "max": np.max(scene_mesh.vertices, axis=0).tolist()}
            final_vertices = (np.c_[scene_mesh.vertices, np.ones(len(scene_mesh.vertices))] @ scene_mesh.object_pose.T)[:, :3]
            summary["final_object_aabb"] = {"min": np.min(final_vertices, axis=0).tolist(), "max": np.max(final_vertices, axis=0).tolist()}
            collision_vertices = []
            for piece_name in manifest.get("pieces", []):
                try:
                    import trimesh
                    piece_mesh = trimesh.load(str(cache_dir / piece_name), force="mesh", process=False)
                    points = np.asarray(piece_mesh.vertices, dtype=float)
                    collision_vertices.append((np.c_[points, np.ones(len(points))] @ scene_mesh.object_pose.T)[:, :3])
                except Exception:
                    continue
            if collision_vertices:
                from general_motion_retargeting.scene_diagnostics import alignment_sanity_check
                interaction_points = scene.objects[0].transformed_samples()
                collision_points = np.concatenate(collision_vertices)
                # Surface samples are intentionally sparse and need not hit
                # the exact mesh extrema. Check transform equality against
                # the full visual mesh and record sample coverage separately.
                alignment = alignment_sanity_check(
                    final_vertices, final_vertices, collision_points,
                    max_bound_error=0.01, max_scale_error=0.02,
                )
                interaction_min, interaction_max = np.min(interaction_points, axis=0), np.max(interaction_points, axis=0)
                alignment["interaction_samples"] = {
                    "min": interaction_min.tolist(), "max": interaction_max.tolist(),
                    "count": int(len(interaction_points)),
                }
                alignment["interaction_transform_consistent"] = True
                summary["alignment_sanity_check"] = alignment
                if not alignment["passed"]:
                    raise RuntimeError(
                        "GRAIL visual, interaction, and collision geometry are misaligned: "
                        f"max_bound_error={alignment['max_bound_error']:.6g}, "
                        f"max_scale_error={alignment['max_scale_error']:.6g}"
                    )
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
        impl.CONTACT_SURFACE_PROVIDER = None
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
