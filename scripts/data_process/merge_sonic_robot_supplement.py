"""Merge an incremental Sonic/robot supplement into an existing robot dataset.

The SMPL-X to robot retargeting step is expensive, so this helper assumes the
base robot dataset already exists and only copies newly retargeted supplement
motions into it. It also writes a combined Sonic manifest/symlink directory for
downstream balanced selection.
"""

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_manifest", type=Path, required=True)
    parser.add_argument("--supplement_manifest", type=Path, required=True)
    parser.add_argument("--combined_sonic_out", type=Path, required=True)
    parser.add_argument("--base_robot_root", type=Path, required=True)
    parser.add_argument("--supplement_robot_root", type=Path, required=True)
    parser.add_argument("--overwrite_sonic_out", action="store_true")
    parser.add_argument("--overwrite_robot_files", action="store_true")
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_source_path(value):
    if not value:
        return ""
    return str(Path(value).expanduser().resolve(strict=False))


def robot_filename(row):
    return Path(row["source_file"]).with_suffix(".pkl").name


def prepare_combined_sonic_dir(path, overwrite):
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output folder is not empty: {path}. Use --overwrite_sonic_out.")
        shutil.rmtree(path)
    (path / "motions").mkdir(parents=True, exist_ok=True)


def include_row(row, seen_paths, seen_files):
    source_path = normalize_source_path(row.get("source_path", ""))
    source_file = Path(row.get("source_file", "")).name
    if not source_file:
        return False, "missing_source_file"
    if source_path and source_path in seen_paths:
        return False, "duplicate_source_path"
    if source_file in seen_files:
        return False, "duplicate_source_file"
    if source_path:
        seen_paths.add(source_path)
    seen_files.add(source_file)
    return True, ""


def normalize_manifest_row(row):
    out = dict(row)
    source_file = Path(out["source_file"]).name
    out["source_file"] = source_file
    out["relative_link"] = str(Path("motions") / source_file)
    return out


def symlink_source(row, combined_sonic_out):
    source_path = Path(row["source_path"]).expanduser()
    dst = combined_sonic_out / "motions" / Path(row["source_file"]).name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(source_path)


def copy_supplement_robot(row, args):
    filename = robot_filename(row)
    src = args.supplement_robot_root / filename
    dst = args.base_robot_root / filename
    if not src.exists():
        return "missing"
    if dst.exists() or dst.is_symlink():
        if not args.overwrite_robot_files:
            return "existing"
        if dst.is_dir():
            raise IsADirectoryError(dst)
        dst.unlink()
    shutil.copy2(src, dst)
    return "copied"


def duration_sec(row):
    try:
        return float(row.get("duration_sec") or 0.0)
    except ValueError:
        return 0.0


def main():
    args = parse_args()
    if not args.base_robot_root.exists():
        raise FileNotFoundError(args.base_robot_root)
    if not args.supplement_robot_root.exists():
        raise FileNotFoundError(args.supplement_robot_root)

    base_rows = read_csv(args.base_manifest)
    supplement_rows = read_csv(args.supplement_manifest)
    prepare_combined_sonic_dir(args.combined_sonic_out, args.overwrite_sonic_out)

    fieldnames = list(base_rows[0].keys()) if base_rows else list(supplement_rows[0].keys())
    if "relative_link" not in fieldnames:
        fieldnames.insert(0, "relative_link")

    seen_paths = set()
    seen_files = set()
    combined_rows = []
    skipped = Counter()

    for row in base_rows:
        status, reason = include_row(row, seen_paths, seen_files)
        if not status:
            skipped[f"base_{reason}"] += 1
            continue
        if not (args.base_robot_root / robot_filename(row)).exists():
            skipped["base_missing_robot"] += 1
            continue
        combined_rows.append(normalize_manifest_row(row))

    robot_copy_counts = Counter()
    for row in supplement_rows:
        status, reason = include_row(row, seen_paths, seen_files)
        if not status:
            skipped[f"supplement_{reason}"] += 1
            continue
        copy_status = copy_supplement_robot(row, args)
        robot_copy_counts[copy_status] += 1
        if copy_status == "missing":
            skipped["supplement_missing_robot"] += 1
            continue
        if copy_status == "existing" and not (args.base_robot_root / robot_filename(row)).exists():
            skipped["supplement_existing_robot_unusable"] += 1
            continue
        combined_rows.append(normalize_manifest_row(row))

    for row in combined_rows:
        symlink_source(row, args.combined_sonic_out)

    write_csv(args.combined_sonic_out / "manifest.csv", combined_rows, fieldnames)
    with (args.combined_sonic_out / "selected_sonic_smpl.txt").open("w", encoding="utf-8") as f:
        for row in combined_rows:
            f.write(row["source_path"] + "\n")

    category_hours = Counter()
    category_files = Counter()
    for row in combined_rows:
        category = row.get("category", "unknown")
        category_hours[category] += duration_sec(row) / 3600.0
        category_files[category] += 1

    summary = {
        "base_manifest": str(args.base_manifest),
        "supplement_manifest": str(args.supplement_manifest),
        "combined_sonic_out": str(args.combined_sonic_out),
        "base_robot_root": str(args.base_robot_root),
        "supplement_robot_root": str(args.supplement_robot_root),
        "base_rows": len(base_rows),
        "supplement_rows": len(supplement_rows),
        "combined_rows": len(combined_rows),
        "combined_hours": round(sum(duration_sec(row) for row in combined_rows) / 3600.0, 6),
        "category_files": dict(sorted(category_files.items())),
        "category_hours": {key: round(value, 6) for key, value in sorted(category_hours.items())},
        "robot_copy_counts": dict(sorted(robot_copy_counts.items())),
        "skipped": dict(sorted(skipped.items())),
    }
    (args.combined_sonic_out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
