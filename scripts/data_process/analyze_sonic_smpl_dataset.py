"""Analyze Sonic SMPL action coverage and segment durations.

The script reads Sonic SMPL .pkl/.joblib files, extracts motion metadata, and
writes reproducible CSV/JSON/Markdown summaries. Action categories are inferred
from filename keywords; the raw motion family is preserved in the manifest for
later manual review or stricter dataset selection.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np


DEFAULT_SRC = Path(
    os.environ.get(
        "SONIC_SMPL_SRC",
        "/home/user/robot_software/drl/wbc/GR00T-WholeBodyControl/data/smpl_filtered",
    )
)
DEFAULT_OUT = Path(os.environ.get("SONIC_SMPL_ANALYSIS_OUT", "data/sonic_smpl_analysis/full_source"))

POSE_KEYS = ("pose_aa", "poses", "original_pose_aa")
FPS_KEYS = ("fps", "mocap_framerate", "mocap_frame_rate", "original_fps")

DURATION_BINS = (
    (0.0, 1.0, "<1s"),
    (1.0, 2.0, "1-2s"),
    (2.0, 3.0, "2-3s"),
    (3.0, 5.0, "3-5s"),
    (5.0, 10.0, "5-10s"),
    (10.0, 20.0, "10-20s"),
    (20.0, 30.0, "20-30s"),
    (30.0, 60.0, "30-60s"),
    (60.0, math.inf, ">=60s"),
)

CATEGORY_DESCRIPTIONS = {
    "object_manipulation_carry": "carry, hold, pick/place, tool and prop interactions",
    "locomotion_walk": "walking clips, side steps, walking loops, starts and stops",
    "locomotion_jog_run": "jogging and running clips, loops, starts and stops",
    "jump": "jumps, high jumps, reach jumps and jump turns",
    "dance": "dance and dancing routines",
    "ground_low_posture": "sit, kneel, crawl, lie and on-ground motions",
    "turn_transition": "turns, stance changes and step-rotate transitions",
    "idle_stance": "idle, stance, relaxed and looking-around standing clips",
    "upper_body_gesture": "standing gestures, reaching, clapping, saluting, itching and similar upper-body actions",
    "obstacle_contact_avoidance": "obstacle avoidance, bumps, collisions and body checks",
    "injury_impaired_gait": "injured or impaired gait variants",
    "exercise_sport": "exercise, sport-like and training motions",
    "daily_social_expression": "daily-life gestures, emotions and social expressions",
    "kick_throw_stoop": "kick, throw, stoop and similar short full-body actions",
    "other": "uncategorized filename patterns",
}

PIPELINE_TARGET_RATIOS = {
    "locomotion": 0.50,
    "idle": 0.15,
    "upper_body": 0.20,
    "transition_mild": 0.10,
    "dynamic": 0.05,
}

PIPELINE_EXCLUDE_KEYWORDS = (
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

PIPELINE_UPPER_BODY_KEYWORDS = (
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

PIPELINE_TRANSITION_MILD_KEYWORDS = (
    "body_stretch",
    "reaching_down",
    "brush_of_dust",
    "body_search",
    "exercise_1",
    "exercise_2",
    "freezing_cold",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Sonic SMPL source dataset coverage and durations.")
    parser.add_argument("--src_folder", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out_folder", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--progress_interval", type=float, default=5.0)
    parser.add_argument("--top_n", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def motion_family_from_path(path):
    return Path(path).stem.split("__")[0]


def actor_id_from_path(path):
    match = re.search(r"__(A[0-9]+)", Path(path).stem)
    return match.group(1) if match else "unknown"


def is_mirror_path(path):
    return Path(path).stem.endswith("_M")


def normalized_family(name):
    tokens = name.lower().split("_")
    while tokens and (tokens[-1].isdigit() or re.fullmatch(r"v[0-9]+", tokens[-1])):
        tokens.pop()
    while tokens and tokens[-1] in {"r", "l", "left", "right"}:
        tokens.pop()
    return "_".join(tokens) if tokens else name.lower()


def has_any(text, keywords):
    return any(keyword in text for keyword in keywords)


def categorize_motion(name):
    text = name.lower()

    if has_any(text, ("bump", "obstacle", "body_check", "avoid_")):
        return "obstacle_contact_avoidance"
    if has_any(text, ("jump", "hop")):
        return "jump"
    if has_any(text, ("dance", "dancing", "mohak")):
        return "dance"
    if has_any(text, ("kneel", "sit", "crawl", "lie", "on_ground", "balled_up")):
        return "ground_low_posture"
    if has_any(text, ("injured", "inj_")):
        return "injury_impaired_gait"
    if has_any(
        text,
        (
            "one_hand",
            "two_hands",
            "pick_up",
            "put_down",
            "hold",
            "carry",
            "crate",
            "box",
            "bucket",
            "big_",
            "small_",
            "medium_",
            "heavy",
            "light",
            "tool",
            "axe",
            "saw",
            "broom",
            "mop",
            "watering",
            "painting",
            "operating",
            "item",
            "trash",
            "apple",
            "binoculars",
        ),
    ):
        return "object_manipulation_carry"
    if has_any(text, ("walk", "sideway_walk", "loop_forward_walk", "loop_backward_walk")):
        return "locomotion_walk"
    if has_any(text, ("jog", "run", "sprint")):
        return "locomotion_jog_run"
    if has_any(text, ("turn", "step_rotate", "stance_change", "change_right", "change_left")):
        return "turn_transition"
    if has_any(text, ("idle", "stance", "relax", "looking_around")):
        return "idle_stance"
    if has_any(text, ("exercise", "burpee", "ab_bicycle", "push_up", "sport", "training")):
        return "exercise_sport"
    if has_any(
        text,
        (
            "clap",
            "salute",
            "reach",
            "reaching",
            "checking_time",
            "thinking",
            "confusion",
            "welcoming",
            "pocket_searching",
            "itching",
            "chefs_kiss",
            "omg",
            "don_t_know",
            "no_see",
            "no_hear",
            "fixing_something",
            "brush",
            "dust",
            "body_search",
            "body_stretch",
            "rubbing",
            "wiping",
            "show_bicep",
            "praying",
            "listening",
            "clearing_ear",
            "yawn",
            "sneeze",
            "bow",
            "beckon",
            "greeting",
            "bye",
            "wave",
        ),
    ):
        return "upper_body_gesture"
    if has_any(text, ("kick", "throw", "stoop")):
        return "kick_throw_stoop"
    if has_any(
        text,
        (
            "triumph",
            "victory",
            "crowd",
            "screaming",
            "lamenting",
            "puke",
            "eureka",
            "angry",
            "alone",
            "bravo",
            "calm_down",
            "as_you_wish",
            "maybe",
            "tasty",
            "just_realised",
            "hurry",
            "eating",
            "drinking",
            "smoke",
            "stinky",
            "sweat",
            "freezing_cold",
            "horse_riding",
            "lasso",
        ),
    ):
        return "daily_social_expression"
    return "other"


def categorize_for_pipeline(name):
    """Mirror scripts/data_process/select_sonic_smpl_subset.py filename categories."""
    lower = name.lower()
    if any(keyword in lower for keyword in PIPELINE_EXCLUDE_KEYWORDS):
        return None, "excluded_action_keyword"
    if lower.startswith("walk") or "walk_" in lower:
        return "locomotion", "walk_family"
    if lower.startswith("idle_turn"):
        return "locomotion", "idle_turn_family"
    if lower.startswith("idle") or lower.startswith("looking_around"):
        return "idle", "idle_or_look_family"
    if lower.startswith("jog"):
        return "dynamic", "jog_family_limited"
    if any(keyword in lower for keyword in PIPELINE_UPPER_BODY_KEYWORDS):
        return "upper_body", "standing_upper_body_keyword"
    if any(keyword in lower for keyword in PIPELINE_TRANSITION_MILD_KEYWORDS):
        return "transition_mild", "transition_mild_keyword"
    return None, "not_in_selected_action_families"


def choose_key(data, keys):
    for key in keys:
        if key in data:
            return key
    return None


def scalar_float(value):
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def analyze_one(path_string):
    path = Path(path_string)
    family = motion_family_from_path(path)
    result = {
        "source_file": path.name,
        "source_path": str(path),
        "motion_family": family,
        "normalized_family": normalized_family(family),
        "actor": actor_id_from_path(path),
        "is_mirror": str(is_mirror_path(path)),
        "category": categorize_motion(family),
        "size_bytes": str(path.stat().st_size),
        "status": "ok",
        "error": "",
        "pose_key": "",
        "fps_key": "",
        "num_frames": "",
        "fps": "",
        "duration_sec": "",
        "original_num_frames": "",
        "original_fps": "",
        "original_duration_sec": "",
    }

    try:
        import joblib

        data = joblib.load(path)
        if not isinstance(data, dict):
            raise ValueError(f"not_dict:{type(data).__name__}")

        pose_key = choose_key(data, POSE_KEYS)
        fps_key = choose_key(data, FPS_KEYS)
        if not pose_key:
            raise KeyError("missing_pose_key")
        if not fps_key:
            raise KeyError("missing_fps_key")

        pose = data[pose_key]
        if not hasattr(pose, "shape") or len(pose.shape) < 1:
            raise ValueError("pose_has_no_frame_axis")
        frames = int(pose.shape[0])
        fps = scalar_float(data[fps_key])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"bad_fps:{fps}")

        result.update(
            {
                "pose_key": pose_key,
                "fps_key": fps_key,
                "num_frames": str(frames),
                "fps": f"{fps:.8g}",
                "duration_sec": f"{frames / fps:.8f}",
            }
        )

        if "original_pose_aa" in data and "original_fps" in data:
            original_frames = int(data["original_pose_aa"].shape[0])
            original_fps = scalar_float(data["original_fps"])
            if math.isfinite(original_fps) and original_fps > 0:
                result["original_num_frames"] = str(original_frames)
                result["original_fps"] = f"{original_fps:.8g}"
                result["original_duration_sec"] = f"{original_frames / original_fps:.8f}"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}:{exc}"

    return result


def numeric(row, key):
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def duration_stats(values):
    clean = np.asarray([value for value in values if value is not None and math.isfinite(value)], dtype=np.float64)
    if clean.size == 0:
        return {
            "count": 0,
            "total_hours": 0.0,
            "min_sec": None,
            "p05_sec": None,
            "p10_sec": None,
            "p25_sec": None,
            "median_sec": None,
            "p75_sec": None,
            "p90_sec": None,
            "p95_sec": None,
            "p99_sec": None,
            "max_sec": None,
            "mean_sec": None,
            "std_sec": None,
        }

    quantiles = np.percentile(clean, [5, 10, 25, 50, 75, 90, 95, 99])
    return {
        "count": int(clean.size),
        "total_hours": round(float(clean.sum() / 3600.0), 6),
        "min_sec": round(float(clean.min()), 6),
        "p05_sec": round(float(quantiles[0]), 6),
        "p10_sec": round(float(quantiles[1]), 6),
        "p25_sec": round(float(quantiles[2]), 6),
        "median_sec": round(float(quantiles[3]), 6),
        "p75_sec": round(float(quantiles[4]), 6),
        "p90_sec": round(float(quantiles[5]), 6),
        "p95_sec": round(float(quantiles[6]), 6),
        "p99_sec": round(float(quantiles[7]), 6),
        "max_sec": round(float(clean.max()), 6),
        "mean_sec": round(float(clean.mean()), 6),
        "std_sec": round(float(clean.std()), 6),
    }


def bin_label(duration):
    for lower, upper, label in DURATION_BINS:
        if lower <= duration < upper:
            return label
    return "unknown"


def summary_rows(rows, group_key):
    grouped = defaultdict(list)
    for row in rows:
        duration = numeric(row, "duration_sec")
        grouped[row[group_key]].append(duration)

    output = []
    for key, values in grouped.items():
        stats = duration_stats(values)
        out = {group_key: key}
        out.update(stats)
        output.append(out)
    output.sort(key=lambda item: (-item["count"], item[group_key]))
    return output


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, summary, category_rows, family_rows, bin_rows, pipeline_rows, args):
    def table(headers, rows):
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
        return "\n".join(lines)

    top_categories = []
    for row in category_rows:
        top_categories.append(
            {
                "category": row["category"],
                "files": row["count"],
                "hours": f"{row['total_hours']:.3f}",
                "median_s": row["median_sec"],
                "p90_s": row["p90_sec"],
            }
        )

    top_families = []
    for row in family_rows[: args.top_n]:
        top_families.append(
            {
                "family": row["normalized_family"],
                "files": row["count"],
                "hours": f"{row['total_hours']:.3f}",
                "median_s": row["median_sec"],
                "p90_s": row["p90_sec"],
            }
        )

    pipeline_table_rows = []
    for row in pipeline_rows:
        ratio = PIPELINE_TARGET_RATIOS.get(row["pipeline_category_or_reason"])
        pipeline_table_rows.append(
            {
                "category_or_reason": row["pipeline_category_or_reason"],
                "files": row["count"],
                "hours": f"{row['total_hours']:.3f}",
                "target_ratio": "" if ratio is None else f"{ratio * 100:.0f}%",
                "median_s": row["median_sec"],
                "p90_s": row["p90_sec"],
            }
        )

    text = [
        "# Sonic SMPL Source Analysis",
        "",
        f"Source: `{args.src_folder}`",
        "",
        "## Overall",
        "",
        "```json",
        json.dumps(summary["overall"], indent=2, sort_keys=True),
            "```",
            "",
            "## Pipeline Selection View",
            "",
            "This view mirrors the filename categories used by `scripts/data_process/select_sonic_smpl_subset.py` and the SMPL retargeting pipeline document.",
            "",
            table(["category_or_reason", "files", "hours", "target_ratio", "median_s", "p90_s"], pipeline_table_rows),
            "",
            "## Action Categories",
            "",
            table(["category", "files", "hours", "median_s", "p90_s"], top_categories),
        "",
        "Category definitions are filename-keyword based:",
        "",
    ]
    for key, description in CATEGORY_DESCRIPTIONS.items():
        text.append(f"- `{key}`: {description}")
    text.extend(
        [
            "",
            "## Duration Histogram",
            "",
            table(["duration_bin", "files", "percent", "hours"], bin_rows),
            "",
            f"## Top {args.top_n} Normalized Motion Families",
            "",
            table(["family", "files", "hours", "median_s", "p90_s"], top_families),
            "",
            "## Files",
            "",
            "- `manifest.csv`: per-file metadata and inferred category",
            "- `category_summary.csv`: duration stats by inferred high-level action category",
            "- `family_summary.csv`: duration stats by normalized motion family",
            "- `duration_bins.csv`: global duration histogram",
            "- `pipeline_selection_category_summary.csv`: stats using the current source-selection script categories",
            "- `summary.json`: machine-readable summary",
        ]
    )
    path.write_text("\n".join(text) + "\n")


def main():
    args = parse_args()
    if not args.src_folder.exists():
        raise FileNotFoundError(args.src_folder)
    if args.out_folder.exists() and any(args.out_folder.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output folder is not empty: {args.out_folder}. Use --overwrite.")
    args.out_folder.mkdir(parents=True, exist_ok=True)

    paths = sorted(str(path) for path in args.src_folder.glob("*.pkl"))
    if args.max_files is not None:
        paths = paths[: args.max_files]

    start = time.time()
    rows = []
    next_progress = time.time() + args.progress_interval
    total = len(paths)

    if args.workers <= 1:
        iterator = map(analyze_one, paths)
        for index, row in enumerate(iterator, start=1):
            rows.append(row)
            now = time.time()
            if now >= next_progress:
                print(f"processed={index}/{total} elapsed_sec={now - start:.1f}", file=sys.stderr, flush=True)
                next_progress = now + args.progress_interval
    else:
        with Pool(processes=args.workers) as pool:
            for index, row in enumerate(pool.imap_unordered(analyze_one, paths, chunksize=64), start=1):
                rows.append(row)
                now = time.time()
                if now >= next_progress:
                    print(f"processed={index}/{total} elapsed_sec={now - start:.1f}", file=sys.stderr, flush=True)
                    next_progress = now + args.progress_interval

    rows.sort(key=lambda row: row["source_file"])

    manifest_fields = [
        "source_file",
        "source_path",
        "motion_family",
        "normalized_family",
        "actor",
        "is_mirror",
        "category",
        "size_bytes",
        "status",
        "error",
        "pose_key",
        "fps_key",
        "num_frames",
        "fps",
        "duration_sec",
        "original_num_frames",
        "original_fps",
        "original_duration_sec",
    ]
    write_csv(args.out_folder / "manifest.csv", rows, manifest_fields)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    durations = [numeric(row, "duration_sec") for row in ok_rows]
    overall_stats = duration_stats(durations)
    status_counts = Counter(row["status"] for row in rows)
    category_counts = Counter(row["category"] for row in rows)
    fps_counts = Counter(row["fps"] for row in ok_rows)

    category_rows = summary_rows(ok_rows, "category")
    family_rows = summary_rows(ok_rows, "normalized_family")

    pipeline_rows_input = []
    for row in ok_rows:
        pipeline_category, pipeline_reason = categorize_for_pipeline(row["motion_family"])
        copied = dict(row)
        copied["pipeline_category_or_reason"] = pipeline_category if pipeline_category else pipeline_reason
        pipeline_rows_input.append(copied)
    pipeline_rows = summary_rows(pipeline_rows_input, "pipeline_category_or_reason")

    bin_counts = Counter()
    bin_hours = Counter()
    for duration in durations:
        if duration is None:
            continue
        label = bin_label(duration)
        bin_counts[label] += 1
        bin_hours[label] += duration / 3600.0

    bin_rows = []
    ok_count = len(ok_rows)
    for _, _, label in DURATION_BINS:
        count = bin_counts[label]
        percent = (count / ok_count * 100.0) if ok_count else 0.0
        bin_rows.append(
            {
                "duration_bin": label,
                "files": count,
                "percent": f"{percent:.3f}",
                "hours": f"{bin_hours[label]:.6f}",
            }
        )

    summary = {
        "source_folder": str(args.src_folder),
        "output_folder": str(args.out_folder),
        "workers": args.workers,
        "elapsed_sec": round(time.time() - start, 3),
        "overall": {
            "files": len(rows),
            "ok_files": len(ok_rows),
            "error_files": len(rows) - len(ok_rows),
            "unique_motion_families": len({row["motion_family"] for row in rows}),
            "unique_normalized_families": len({row["normalized_family"] for row in rows}),
            "unique_actors": len({row["actor"] for row in rows}),
            "mirror_files": sum(1 for row in rows if row["is_mirror"] == "True"),
            "duration": overall_stats,
        },
        "status_counts": dict(sorted(status_counts.items())),
        "category_file_counts": dict(sorted(category_counts.items())),
        "fps_counts": dict(sorted(fps_counts.items(), key=lambda item: (float(item[0]) if item[0] else -1.0, item[0]))),
        "category_descriptions": CATEGORY_DESCRIPTIONS,
        "duration_bins": bin_rows,
        "top_categories": category_rows,
        "pipeline_target_ratios": PIPELINE_TARGET_RATIOS,
        "pipeline_selection_categories": pipeline_rows,
        "top_normalized_families": family_rows[: args.top_n],
    }

    summary_fields = [
        "count",
        "total_hours",
        "min_sec",
        "p05_sec",
        "p10_sec",
        "p25_sec",
        "median_sec",
        "p75_sec",
        "p90_sec",
        "p95_sec",
        "p99_sec",
        "max_sec",
        "mean_sec",
        "std_sec",
    ]
    write_csv(args.out_folder / "category_summary.csv", category_rows, ["category"] + summary_fields)
    write_csv(args.out_folder / "family_summary.csv", family_rows, ["normalized_family"] + summary_fields)
    write_csv(
        args.out_folder / "pipeline_selection_category_summary.csv",
        pipeline_rows,
        ["pipeline_category_or_reason"] + summary_fields,
    )
    write_csv(args.out_folder / "duration_bins.csv", bin_rows, ["duration_bin", "files", "percent", "hours"])
    (args.out_folder / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(args.out_folder / "README.md", summary, category_rows, family_rows, bin_rows, pipeline_rows, args)

    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    print(f"Wrote analysis to {args.out_folder}")


if __name__ == "__main__":
    main()
