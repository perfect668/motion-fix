"""Recompute GRAIL object AABBs and verify the exported scene summary."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import numpy as np

from general_motion_retargeting.scene_asset_loader import load_scene_asset


def _pose(record: dict) -> np.ndarray:
    obj = record.get("obj_data", {})
    R = np.asarray(obj["obj_R"], dtype=float)
    t = np.asarray(obj["obj_t"], dtype=float)
    scale = np.asarray(obj.get("obj_scale", [1, 1, 1]), dtype=float).reshape(-1)
    if R.ndim == 3: R = R[0]
    if t.ndim == 2: t = t[0]
    if scale.size == 1: scale = np.repeat(scale, 3)
    pose = np.eye(4)
    pose[:3, :3] = R @ np.diag(scale)
    pose[:3, 3] = t
    return pose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--object_asset", required=True, type=Path)
    args = parser.parse_args()
    record = pickle.loads(args.motion.read_bytes())
    summary = json.loads(args.summary.read_text())
    mesh = load_scene_asset(args.object_asset, {"asset_space": "object_local", "asset_scale_baked": False})
    raw_min, raw_max = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    pose = _pose(record)
    final = (np.c_[mesh.vertices, np.ones(len(mesh.vertices))] @ pose.T)[:, :3]
    final_min, final_max = final.min(axis=0), final.max(axis=0)
    expected = {"raw_object_aabb": (raw_min, raw_max), "final_object_aabb": (final_min, final_max)}
    errors = {}
    for key, (lo, hi) in expected.items():
        observed = summary.get(key, {})
        errors[key] = max(
            float(np.max(np.abs(np.asarray(observed.get("min", [])) - lo))),
            float(np.max(np.abs(np.asarray(observed.get("max", [])) - hi))),
        ) if observed else float("inf")
    result = {"scene_scale": float(summary.get("scene_scale", np.nan)), "obj_scale": summary.get("obj_scale"), "aabb_max_abs_error": errors, "passed": all(value <= 1e-8 for value in errors.values())}
    print(json.dumps(result, indent=2))
    if not result["passed"] or not np.isclose(result["scene_scale"], 1.0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
