"""Inspect exported terrain contact and nonpenetration diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

from general_motion_retargeting.scene_diagnostics import summarize_scene_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--summary_output", type=Path, default=None,
                        help="Write a normalized V4 scene/contact summary JSON")
    args = parser.parse_args()
    with args.motion.open("rb") as stream:
        motion = pickle.load(stream)
    if args.summary_output is not None:
        schedule = motion.get("contact_schedule", [])
        if not schedule and isinstance(motion.get("contact_plan"), dict):
            schedule = motion["contact_plan"].get("frames", [])
        summary = summarize_scene_diagnostics(
            motion.get("terrain_diagnostics", motion.get("diagnostics", [])),
            schedule,
        )
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote normalized scene summary to {args.summary_output}")
    print(json.dumps(motion.get("contact_metrics", motion.get("validation", {}).get("contact_metrics", {})), indent=2))
    if args.frame is None:
        return
    frame = args.frame
    schedule = motion.get("contact_schedule", [])
    if not schedule and isinstance(motion.get("contact_plan"), dict):
        schedule = motion["contact_plan"].get("frames", [])
    diagnostics = motion.get("terrain_diagnostics", motion.get("diagnostics", []))
    if frame < 0 or frame >= min(len(schedule), len(diagnostics)):
        raise IndexError(f"Frame {frame} outside [0, {min(len(schedule), len(diagnostics)) - 1}]")
    print("\nContact channels:")
    for name, item in schedule[frame]["contacts"].items():
        print(
            f"{name:12s} score={item['score']:.3f} state={item['state']:7s} "
            f"surface={item['surface_id']:12s} d={item['signed_distance']:+.4f} "
            f"vn={item['normal_speed']:+.3f} vt={item['tangential_speed']:.3f}"
        )
    current = diagnostics[frame]
    print(
        f"\nIK passes={current.get('qp_iterations', current.get('passes', 0))} "
        f"active={current.get('active_constraints', current.get('active_terrain_constraints', 0))} "
        f"QP failures={current.get('qp_failures', [])} "
        f"min_terrain={current.get('minimum_terrain_distance', float('inf')):+.5f}"
    )
    print(
        "Scene collision: "
        f"candidate_pairs={current.get('scene_collision_candidate_pairs', 0)} "
        f"active_pairs={current.get('scene_collision_active_pairs', 0)} "
        f"min_distance={current.get('minimum_scene_distance', float('inf')):+.5f} "
        f"max_penetration={current.get('maximum_penetration', current.get('maximum_scene_penetration', 0.0)):.5f} "
        f"query_time={current.get('collision_query_time', 0.0) * 1e3:.2f}ms "
        f"qp_time={current.get('qp_solve_time', 0.0) * 1e3:.2f}ms"
    )
    if current.get("slacks"):
        print("Closest candidates:")
        for name, item in sorted(current["slacks"].items(), key=lambda pair: pair[1]["slack"])[:20]:
            print(
                f"{name:42s} d={item['signed_distance']:+.5f} margin={item['margin']:.4f} "
                f"slack={item['slack']:+.5f} surface={item['surface_id']}"
            )


if __name__ == "__main__":
    main()
