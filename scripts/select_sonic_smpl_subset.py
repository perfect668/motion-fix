"""Select a traceable Sonic SMPL source subset before format conversion.

Purpose:
    Pick a balanced 10-12 hour subset from a large gear-sonic SMPL directory.
    The script only creates symlinks and manifest files; it does not copy the
    large source .pkl files.

Typical usage:
    conda run -n gear_sonic_train python scripts/select_sonic_smpl_subset.py \
        --src_folder ~/robot_software/drl/wbc/GR00T-WholeBodyControl/data/smpl_filtered \
        --out_folder data/sonic_smpl_data/selected_10_12h \
        --target_hours 11 --min_hours 10 --max_hours 12 --overwrite

Outputs:
    motions/ symlinks, manifest.csv, selected_sonic_smpl.txt, summary.json,
    and selection_principles.md.

Defaults:
    --src_folder can also be supplied by SONIC_SMPL_SRC.
    --out_folder can also be supplied by SONIC_SMPL_OUT.
"""

import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_SRC = Path(
    os.environ.get(
        "SONIC_SMPL_SRC",
        "/home/user/robot_software/drl/wbc/GR00T-WholeBodyControl/data/smpl_filtered",
    )
)
DEFAULT_OUT = Path(os.environ.get("SONIC_SMPL_OUT", "data/sonic_smpl_data/selected_10_12h"))

TARGET_RATIOS = {
    "locomotion": 0.50,
    "idle": 0.15,
    "upper_body": 0.20,
    "transition_mild": 0.10,
    "dynamic": 0.05,
}

EXCLUDE_KEYWORDS = (
    "jump",
    "high_jump",
    "reach_jump",
    "dance",
    "dancing",
    "kneel",
    "kneeling",
    "sit",
    "sit_on_heels",
    "crawl",
    "lie",
    "on_ground",
    "horse_riding",
    "lasso",
    "screaming",
    "body_check",
)

UPPER_BODY_KEYWORDS = (
    "clap",
    "salute",
    "thinking",
    "confusion",
    "welcoming",
    "reaching_up",
    "reaching_far",
    "pocket_searching",
    "checking_time",
    "looking_around",
    "itching",
    "chefs_kiss",
    "omg",
    "don_t_know",
    "no_see",
    "no_hear",
    "fixing_something",
)

TRANSITION_MILD_KEYWORDS = (
    "body_stretch",
    "reaching_down",
    "brush_of_dust",
    "body_search",
    "exercise_1",
    "exercise_2",
    "freezing_cold",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Select a 10-12 hour Sonic SMPL subset using symlinks.")
    parser.add_argument("--src_folder", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out_folder", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target_hours", type=float, default=11.0)
    parser.add_argument("--min_hours", type=float, default=10.0)
    parser.add_argument("--max_hours", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--min_duration_sec", type=float, default=1.5)
    parser.add_argument("--max_duration_sec", type=float, default=30.0)
    parser.add_argument("--max_root_speed", type=float, default=4.0)
    parser.add_argument("--max_vertical_span", type=float, default=0.6)
    parser.add_argument("--max_family_minutes", type=float, default=10.0)
    parser.add_argument("--max_actor_minutes", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def motion_name(path):
    return path.stem.split("__")[0]


def actor_id(path):
    match = re.search(r"__(A[0-9]+)", path.stem)
    return match.group(1) if match else "unknown"


def is_mirror(path):
    return path.stem.endswith("_M")


def categorize(name):
    lower = name.lower()
    if any(keyword in lower for keyword in EXCLUDE_KEYWORDS):
        return None, "excluded_action_keyword"
    if lower.startswith("walk") or "walk_" in lower:
        return "locomotion", "walk_family"
    if lower.startswith("idle_turn"):
        return "locomotion", "idle_turn_family"
    if lower.startswith("idle") or lower.startswith("looking_around"):
        return "idle", "idle_or_look_family"
    if lower.startswith("jog"):
        return "dynamic", "jog_family_limited"
    if any(keyword in lower for keyword in UPPER_BODY_KEYWORDS):
        return "upper_body", "standing_upper_body_keyword"
    if any(keyword in lower for keyword in TRANSITION_MILD_KEYWORDS):
        return "transition_mild", "transition_mild_keyword"
    return None, "not_in_selected_action_families"


def load_metadata(path):
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("This selector needs joblib. Run it with the gear_sonic_train conda env.") from exc

    data = joblib.load(path)
    if not isinstance(data, dict):
        return None, "not_dict"
    for key in ("pose_aa", "transl", "fps"):
        if key not in data:
            return None, f"missing_{key}"

    pose = np.asarray(data["pose_aa"])
    trans = np.asarray(data["transl"])
    fps = float(data["fps"])

    if pose.ndim != 2 or pose.shape[1] != 72:
        return None, f"bad_pose_shape_{pose.shape}"
    if trans.ndim != 2 or trans.shape[1] != 3:
        return None, f"bad_trans_shape_{trans.shape}"
    if pose.shape[0] != trans.shape[0]:
        return None, "pose_trans_frame_mismatch"
    if fps <= 0 or fps > 240:
        return None, f"bad_fps_{fps}"
    if not np.isfinite(pose).all() or not np.isfinite(trans).all():
        return None, "non_finite_values"

    # Sonic transl is Y-up. Convert to GMR Z-up only for geometric checks.
    trans_gmr = np.stack([trans[:, 0], -trans[:, 2], trans[:, 1]], axis=1)
    if len(trans_gmr) > 1:
        root_speed = np.linalg.norm(np.diff(trans_gmr, axis=0), axis=1) * fps
        max_root_speed = float(root_speed.max())
    else:
        max_root_speed = 0.0

    return {
        "frames": int(pose.shape[0]),
        "fps": fps,
        "duration_sec": float(pose.shape[0] / fps),
        "max_root_speed": max_root_speed,
        "vertical_span": float(trans_gmr[:, 2].max() - trans_gmr[:, 2].min()),
    }, None


def write_principles(path, args, summary):
    text = f"""# Sonic SMPL Selection Principles

Source:

```text
{args.src_folder}
```

Output:

```text
{args.out_folder}
```

Target duration: {args.target_hours:.1f} hours, accepted range {args.min_hours:.1f}-{args.max_hours:.1f} hours.

## Category Ratios

```text
locomotion       50%
idle             15%
upper_body       20%
transition_mild  10%
dynamic           5%
```

## Include First

- walk, walk_ff, walk_forward, walk_backward, walk_left, walk_right
- idle, idle_loop, idle_turn
- looking_around, checking_time, salute, clap, thinking, confusion, welcoming
- reaching_up, reaching_far, pocket_searching, body_stretch

## Limited Include

- jog and mild exercise motions, capped by the dynamic ratio
- reaching_down, brush_of_dust, body_search, itching-like upper-body motions

## Exclude By Default

- jump, high_jump, reach_jump
- dance, dancing
- kneel, kneeling, sit, sit_on_heels
- crawl, lie, on_ground
- horse_riding, lasso, screaming, body_check

## Quality Gates

- `pose_aa` must be `(T, 72)`.
- `transl` must be `(T, 3)`.
- `pose_aa` and `transl` must have the same frame count.
- `fps` must be valid.
- No NaN or Inf values.
- Duration must be between {args.min_duration_sec:.1f}s and {args.max_duration_sec:.1f}s.
- After Sonic Y-up to GMR Z-up check, max root speed must be <= {args.max_root_speed:.1f} m/s.
- Vertical span must be <= {args.max_vertical_span:.2f} m.
- Per motion family cap: {args.max_family_minutes:.1f} min.
- Per actor cap: {args.max_actor_minutes:.1f} min.

## Result Summary

```json
{json.dumps(summary, indent=2, sort_keys=True)}
```
"""
    path.write_text(text)


def main():
    args = parse_args()
    if args.out_folder.exists() and any(args.out_folder.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output folder is not empty: {args.out_folder}. Use --overwrite.")

    motions_dir = args.out_folder / "motions"
    args.out_folder.mkdir(parents=True, exist_ok=True)
    motions_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    candidates = defaultdict(list)
    excluded_by_name = defaultdict(int)

    for path in sorted(args.src_folder.glob("*.pkl")):
        name = motion_name(path)
        category, reason = categorize(name)
        if category is None:
            excluded_by_name[reason] += 1
            continue
        candidates[category].append((path, name, reason))

    for items in candidates.values():
        rng.shuffle(items)

    target_total_sec = args.target_hours * 3600.0
    category_targets = {key: target_total_sec * ratio for key, ratio in TARGET_RATIOS.items()}
    max_family_sec = args.max_family_minutes * 60.0
    max_actor_sec = args.max_actor_minutes * 60.0

    selected = []
    category_sec = defaultdict(float)
    family_sec = defaultdict(float)
    actor_sec = defaultdict(float)
    rejection_counts = defaultdict(int)

    def try_select(category, strict_category_target=True):
        changed = False
        for path, name, reason in candidates.get(category, []):
            if strict_category_target and category_sec[category] >= category_targets[category]:
                break
            if sum(category_sec.values()) >= target_total_sec:
                break
            actor = actor_id(path)
            metadata, reject_reason = load_metadata(path)
            if reject_reason:
                rejection_counts[reject_reason] += 1
                continue
            duration = metadata["duration_sec"]
            if duration < args.min_duration_sec:
                rejection_counts["too_short"] += 1
                continue
            if duration > args.max_duration_sec:
                rejection_counts["too_long"] += 1
                continue
            if metadata["max_root_speed"] > args.max_root_speed:
                rejection_counts["root_speed_high"] += 1
                continue
            if metadata["vertical_span"] > args.max_vertical_span:
                rejection_counts["vertical_span_high"] += 1
                continue
            if family_sec[name] + duration > max_family_sec:
                rejection_counts["family_cap"] += 1
                continue
            if actor_sec[actor] + duration > max_actor_sec:
                rejection_counts["actor_cap"] += 1
                continue

            rel_link = Path("motions") / path.name
            dst = args.out_folder / rel_link
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(path, dst)

            row = {
                "relative_link": str(rel_link),
                "source_path": str(path),
                "source_file": path.name,
                "motion_family": name,
                "actor": actor,
                "is_mirror": str(is_mirror(path)),
                "category": category,
                "fps": f"{metadata['fps']:.6g}",
                "num_frames": str(metadata["frames"]),
                "duration_sec": f"{duration:.6f}",
                "keep_reason": reason,
                "max_root_speed": f"{metadata['max_root_speed']:.6f}",
                "vertical_span": f"{metadata['vertical_span']:.6f}",
            }
            selected.append(row)
            category_sec[category] += duration
            family_sec[name] += duration
            actor_sec[actor] += duration
            changed = True
        return changed

    for category in TARGET_RATIOS:
        try_select(category, strict_category_target=True)

    if sum(category_sec.values()) < args.min_hours * 3600.0:
        for category in TARGET_RATIOS:
            try_select(category, strict_category_target=False)
            if sum(category_sec.values()) >= args.min_hours * 3600.0:
                break

    total_sec = sum(category_sec.values())
    summary = {
        "selected_files": len(selected),
        "total_hours": round(total_sec / 3600.0, 3),
        "category_hours": {key: round(value / 3600.0, 3) for key, value in sorted(category_sec.items())},
        "category_files": {
            key: sum(1 for row in selected if row["category"] == key) for key in sorted(TARGET_RATIOS)
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "excluded_by_name_counts": dict(sorted(excluded_by_name.items())),
        "seed": args.seed,
    }

    fieldnames = [
        "relative_link",
        "source_path",
        "source_file",
        "motion_family",
        "actor",
        "is_mirror",
        "category",
        "fps",
        "num_frames",
        "duration_sec",
        "keep_reason",
        "max_root_speed",
        "vertical_span",
    ]
    with (args.out_folder / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    with (args.out_folder / "selected_sonic_smpl.txt").open("w") as f:
        for row in selected:
            f.write(row["source_path"] + "\n")

    (args.out_folder / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_principles(args.out_folder / "selection_principles.md", args, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if total_sec < args.min_hours * 3600.0 or total_sec > args.max_hours * 3600.0:
        raise RuntimeError(
            f"Selected {total_sec / 3600.0:.2f}h, outside requested range "
            f"{args.min_hours:.1f}-{args.max_hours:.1f}h."
        )


if __name__ == "__main__":
    main()
