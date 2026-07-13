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
import shutil
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

try:
    from .motion_semantics import CATEGORY_DESCRIPTIONS, describe_motion
    from .sonic_smpl import load_sonic_motion
except ImportError:
    from motion_semantics import CATEGORY_DESCRIPTIONS, describe_motion
    from sonic_smpl import load_sonic_motion


DEFAULT_SRC = Path(
    os.environ.get(
        "SONIC_SMPL_SRC",
        "/home/user/robot_software/drl/wbc/GR00T-WholeBodyControl/data/smpl_filtered",
    )
)
DEFAULT_OUT = Path(os.environ.get("SONIC_SMPL_ANALYSIS_OUT", "data/sonic_smpl_analysis/full_source"))

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


def analyze_one(path_string):
    path = Path(path_string)
    semantics = describe_motion(path)
    result = {
        "source_file": path.name,
        "source_path": str(path),
        "motion_family": semantics.motion_family,
        "normalized_family": semantics.normalized_family,
        "actor": semantics.actor,
        "is_mirror": str(semantics.is_mirror),
        "category": semantics.category,
        "external_support_dependency": str(semantics.external_support_dependency),
        "size_bytes": str(path.stat().st_size),
        "status": "ok",
        "error": "",
        "pose_key": "",
        "trans_key": "",
        "fps_key": "",
        "normalization_adjustments": "",
        "num_frames": "",
        "fps": "",
        "duration_sec": "",
        "original_num_frames": "",
        "original_fps": "",
        "original_duration_sec": "",
    }

    try:
        motion = load_sonic_motion(path)
        result.update(
            {
                "pose_key": motion.pose_key,
                "trans_key": motion.trans_key,
                "fps_key": motion.fps_key,
                "normalization_adjustments": ",".join(motion.adjustments),
                "num_frames": str(motion.poses.shape[0]),
                "fps": f"{motion.fps:.8g}",
                "duration_sec": f"{motion.poses.shape[0] / motion.fps:.8f}",
            }
        )
        if motion.original_num_frames is not None and motion.original_fps is not None:
            result["original_num_frames"] = str(motion.original_num_frames)
            result["original_fps"] = f"{motion.original_fps:.8g}"
            result["original_duration_sec"] = f"{motion.original_num_frames / motion.original_fps:.8f}"
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


def write_markdown(path, summary, category_rows, family_rows, bin_rows, args):
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
            "- `summary.json`: machine-readable summary",
        ]
    )
    path.write_text("\n".join(text) + "\n")


def main():
    args = parse_args()
    if not args.src_folder.exists():
        raise FileNotFoundError(args.src_folder)
    if args.out_folder.exists() and any(args.out_folder.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output folder is not empty: {args.out_folder}. Use --overwrite.")
        shutil.rmtree(args.out_folder)
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
        "external_support_dependency",
        "size_bytes",
        "status",
        "error",
        "pose_key",
        "trans_key",
        "fps_key",
        "normalization_adjustments",
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
    write_csv(args.out_folder / "duration_bins.csv", bin_rows, ["duration_bin", "files", "percent", "hours"])
    (args.out_folder / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(args.out_folder / "README.md", summary, category_rows, family_rows, bin_rows, args)

    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    print(f"Wrote analysis to {args.out_folder}")


if __name__ == "__main__":
    main()
