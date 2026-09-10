"""Sanity checks that visual, query and collision scene transforms agree."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np


def alignment_sanity(scene_model, combined_manifest: str | Path | None = None) -> dict:
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
        report["assets"].append(record)
    if report["transform_mismatches"]:
        report["status"] = "FAIL"
    return report
