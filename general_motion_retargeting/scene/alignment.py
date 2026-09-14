"""Sanity checks that visual, query and collision scene transforms agree."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np


def _world_aabb(vertices: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(vertices, dtype=float).reshape((-1, 3))
    matrix = np.asarray(pose, dtype=float).reshape(4, 4)
    world = (np.c_[values, np.ones(len(values))] @ matrix.T)[:, :3]
    return world.min(axis=0), world.max(axis=0)


def _aabb(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(vertices, dtype=float).reshape((-1, 3))
    return values.min(axis=0), values.max(axis=0)


def _obj_vertices(path: str | Path) -> np.ndarray:
    """Load a generated OBJ/cache mesh solely for transform verification."""
    import trimesh
    loaded = trimesh.load(str(path), force="scene", process=False)
    geometries = list(loaded.geometry.values()) if hasattr(loaded, "geometry") else [loaded]
    values = [np.asarray(mesh.vertices, dtype=float) for mesh in geometries if len(mesh.vertices)]
    if not values:
        raise ValueError(f"Generated scene mesh is empty: {path}")
    return np.concatenate(values, axis=0)


def alignment_sanity(scene_model, combined_manifest: str | Path | None = None, *, geometry=None) -> dict:
    report = {"status": "PASS", "assets": [], "transform_mismatches": []}
    manifest = None
    if combined_manifest is not None:
        path = Path(combined_manifest)
        if path.is_file():
            manifest = json.loads(path.read_text(encoding="utf-8"))
    generated = {str(item.get("object_id")): item for item in (manifest or {}).get("objects", [])}
    for asset in scene_model.assets:
        item = generated.get(str(asset.asset_id))
        record = {
            "asset_id": asset.asset_id,
            "visual_path": str(asset.path),
            "pose": np.asarray(asset.pose, dtype=float).tolist(),
            "pose_trajectory": asset.pose_trajectory is not None,
            "asset_scale_baked": asset.asset_scale_baked,
        }
        if item is None and manifest is not None:
            report["transform_mismatches"].append(f"missing collision object {asset.asset_id}")
        elif item is not None:
            collision_pose = np.asarray(item.get("pose", np.eye(4)), dtype=float).reshape(4, 4)
            if not np.allclose(collision_pose, asset.pose, atol=1e-8):
                report["transform_mismatches"].append(f"pose mismatch for {asset.asset_id}")
            record["collision_piece_count"] = int(item.get("convex_pieces", 0))
            record["interaction_points"] = int(item.get("interaction_points", 0))
            # Pose equality alone cannot detect a baked-unit mismatch.  When
            # prepared geometry is available, compare actual world-space
            # visual bounds produced for MuJoCo with the contact/query bounds.
            if geometry is not None:
                try:
                    expected = geometry.world_vertices().get(str(asset.asset_id))
                    visual_path = item.get("visual_mesh")
                    if expected is None or visual_path is None:
                        raise ValueError("prepared/query or visual mesh unavailable")
                    expected_lower, expected_upper = _aabb(expected)
                    visual_lower, visual_upper = _world_aabb(_obj_vertices(visual_path), collision_pose)
                    error = float(max(
                        np.max(np.abs(expected_lower - visual_lower)),
                        np.max(np.abs(expected_upper - visual_upper)),
                    ))
                    record["query_visual_aabb_error"] = error
                    if error > 1e-5:
                        report["transform_mismatches"].append(
                            f"query/visual world AABB mismatch for {asset.asset_id}: {error:.6g} m"
                        )
                    collision_manifest = item.get("collision_manifest")
                    if collision_manifest:
                        payload = json.loads(Path(collision_manifest).read_text(encoding="utf-8"))
                        pieces = [
                            _obj_vertices(Path(collision_manifest).parent / name)
                            for name in payload.get("pieces", [])
                        ]
                        if pieces:
                            collision_lower, collision_upper = _world_aabb(
                                np.concatenate(pieces, axis=0), collision_pose
                            )
                            center_error = float(np.max(np.abs(
                                .5 * (collision_lower + collision_upper)
                                - .5 * (expected_lower + expected_upper)
                            )))
                            record["query_collision_aabb_center_error"] = center_error
                            # Convex decomposition can alter its outer extent,
                            # but it must not be translated/scaled into another
                            # world.  A 2 cm centre allowance covers CoACD's
                            # conservative hull approximation without hiding a
                            # unit-scale or pose error.
                            if center_error > 0.02:
                                report["transform_mismatches"].append(
                                    f"query/collision world AABB centre mismatch for {asset.asset_id}: {center_error:.6g} m"
                                )
                except Exception as error:
                    report["transform_mismatches"].append(
                        f"could not verify world AABB alignment for {asset.asset_id}: {error}"
                    )
        report["assets"].append(record)
    if report["transform_mismatches"]:
        report["status"] = "FAIL"
    return report
