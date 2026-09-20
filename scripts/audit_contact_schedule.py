"""Audit exported contacts independently of their stored signed distances.

Run on trusted V4 PKL outputs. This checks contact geometry and reported
motion increments; it does not certify dynamic feasibility or replay IK.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def audit(payload: dict, contact_distance: float = 0.05) -> dict:
    schedule = payload.get("contact_schedule", [])
    channels = {}
    for frame_index, frame in enumerate(schedule):
        for name, item in frame.get("contacts", {}).items():
            if not item.get("object_id") or item.get("state", "NONE") == "NONE":
                continue
            point = np.asarray(item["human_point_solver"], dtype=float)
            surface = np.asarray(item["surface_point_solver"], dtype=float)
            distance = float(np.linalg.norm(point - surface))
            row = channels.setdefault(name, {"active_mesh_contacts": 0,
                "outside_distance_count": 0, "max_surface_distance_m": 0.0,
                "worst_frame": None})
            row["active_mesh_contacts"] += 1
            row["outside_distance_count"] += int(not np.isfinite(distance) or distance > contact_distance + 1e-6)
            if np.isfinite(distance) and distance > row["max_surface_distance_m"]:
                row["max_surface_distance_m"] = distance
                row["worst_frame"] = frame_index
    result = {"frames": len(schedule), "contact_distance_m": contact_distance,
              "channels": channels,
              "outside_distance_count": sum(c["outside_distance_count"] for c in channels.values())}
    qpos = np.asarray(payload.get("qpos", []), dtype=float)
    if qpos.ndim == 2 and len(qpos) > 1 and qpos.shape[1] >= 7:
        steps = np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1)
        result["max_root_translation_step_m"] = float(np.max(steps))
        result["max_root_translation_step_frame"] = int(np.argmax(steps) + 1)
        result["root_translation_step_p95_m"] = float(np.percentile(steps, 95))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--contact-distance", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.contact_distance <= 0:
        parser.error("--contact-distance must be positive")
    with args.motion.open("rb") as stream:
        result = audit(pickle.load(stream), args.contact_distance)
    report = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
