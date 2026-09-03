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
        summary = summarize_scene_diagnostics(
            motion.get("terrain_diagnostics", motion.get("diagnostics", [])),
            motion.get("contact_schedule", []),
        )
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote normalized scene summary to {args.summary_output}")
    print(json.dumps(motion.get("contact_metrics", {}), indent=2))
    if args.frame is None:
        return
    frame = args.frame
    schedule = motion.get("contact_schedule", [])
    diagnostics = motion.get("terrain_diagnostics", [])
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
        f"\nIK passes={current['passes']} active={current['active_constraints']} "
        f"QP failures={current['qp_failures']} min_slack={current['min_slack_after']:+.5f}"
    )
    print("Closest candidates:")
    for name, item in sorted(current["slacks"].items(), key=lambda pair: pair[1]["slack"])[:20]:
        print(
            f"{name:42s} d={item['signed_distance']:+.5f} margin={item['margin']:.4f} "
            f"slack={item['slack']:+.5f} surface={item['surface_id']}"
        )


if __name__ == "__main__":
    main()
