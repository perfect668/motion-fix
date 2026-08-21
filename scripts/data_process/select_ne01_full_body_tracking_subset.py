#!/usr/bin/env python3
"""Create a duration-capped, category-balanced NE01 robot-motion subset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
from collections import defaultdict
from pathlib import Path

from motion_semantics import describe_motion, has_external_support_dependency


TARGET_RATIOS = {
    "locomotion_walk": 0.25,
    "locomotion_jog_run": 0.25,
    "jump": 0.15,
    "turn_transition": 0.10,
    "dance": 0.10,
    "exercise_sport": 0.05,
    "kick_throw_stoop": 0.05,
    "upper_body_gesture": 0.03,
    "idle_stance": 0.02,
}

EXCLUDED_CATEGORIES = {"object_manipulation_carry", "ground_low_posture"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-hours", type=float, default=20.0)
    parser.add_argument("--max-hours", type=float, default=20.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_candidate(path: Path, root: Path):
    with path.open("rb") as stream:
        motion = pickle.load(stream)
    fps = float(motion.get("fps", 50.0))
    frames = len(motion["root_pos"])
    relative_file = path.relative_to(root)
    semantics = describe_motion(relative_file)
    if (
        semantics.category in EXCLUDED_CATEGORIES
        or semantics.external_support_dependency
        or has_external_support_dependency(relative_file)
        or fps <= 0
        or frames < 2
    ):
        return None
    return {
        "relative_file": str(relative_file),
        "category": semantics.category,
        "motion_family": semantics.motion_family,
        "duration_sec": (frames - 1) / fps,
        "frames": frames,
        "fps": fps,
    }


def order_pool(rows):
    # Prefer distinct motion families before duplicate takes of the same action.
    return sorted(rows, key=lambda row: (row["motion_family"], row["duration_sec"], row["relative_file"]))


def choose(rows, target_seconds, max_seconds):
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    selected = []
    selected_paths = set()
    family_counts = defaultdict(int)

    def add_from_pool(pool, budget):
        nonlocal selected
        pending = list(pool)
        while pending and budget > 0:
            pending.sort(key=lambda row: (family_counts[row["motion_family"]], row["duration_sec"], row["relative_file"]))
            candidate = next(
                (
                    row
                    for row in pending
                    if row["relative_file"] not in selected_paths
                    and sum(item["duration_sec"] for item in selected) + row["duration_sec"] <= max_seconds
                ),
                None,
            )
            if candidate is None:
                return
            selected.append(candidate)
            selected_paths.add(candidate["relative_file"])
            family_counts[candidate["motion_family"]] += 1
            budget -= candidate["duration_sec"]
            pending.remove(candidate)

    for category, ratio in TARGET_RATIOS.items():
        add_from_pool(order_pool(by_category[category]), target_seconds * ratio)

    remaining = [row for row in rows if row["relative_file"] not in selected_paths]
    add_from_pool(order_pool(remaining), target_seconds - sum(row["duration_sec"] for row in selected))
    return selected


def main():
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    motions_dir = args.output_dir / "motions"
    motions_dir.mkdir(parents=True)

    candidates = []
    for path in sorted(args.input_root.rglob("*.pkl")):
        candidate = read_candidate(path, args.input_root)
        if candidate:
            candidates.append(candidate)
    selected = choose(candidates, args.target_hours * 3600, args.max_hours * 3600)

    for row in selected:
        source = (args.input_root / row["relative_file"]).resolve()
        target = motions_dir / row["relative_file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(source, target)

    fields = ["relative_file", "category", "motion_family", "duration_sec", "frames", "fps"]
    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    category_summary = defaultdict(lambda: [0, 0.0])
    for row in selected:
        category_summary[row["category"]][0] += 1
        category_summary[row["category"]][1] += row["duration_sec"]
    summary = {
        "input_root": str(args.input_root),
        "selected_files": len(selected),
        "selected_hours": round(sum(row["duration_sec"] for row in selected) / 3600, 6),
        "available_files": len(candidates),
        "available_hours": round(sum(row["duration_sec"] for row in candidates) / 3600, 6),
        "category_summary": {
            category: {"files": count, "hours": round(seconds / 3600, 6)}
            for category, (count, seconds) in sorted(category_summary.items())
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
