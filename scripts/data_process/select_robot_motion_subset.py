"""Select a balanced robot-motion subset from a filter relaxed report."""

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


EXTERNAL_SUPPORT_PHRASES = (
    "against_wall",
    "brace_against",
    "bracing_against",
    "lean_against",
    "lean_on",
    "leaning_on",
    "leaning_wall",
    "jump_off_wall",
    "nailing_wall",
    "off_wall",
    "prop_against",
    "propped_against",
    "rest_on",
    "resting_on",
    "supported_by",
    "wall_lean",
)
EXTERNAL_SUPPORT_ACTION_TOKENS = {
    "brace",
    "bracing",
    "hang",
    "hanging",
    "lean",
    "leaning",
    "prop",
    "propped",
    "propping",
    "rest",
    "resting",
    "support",
    "supported",
    "supporting",
}
EXTERNAL_SUPPORT_OBJECT_TOKENS = {
    "bar",
    "chair",
    "counter",
    "door",
    "fence",
    "ladder",
    "pole",
    "rail",
    "railing",
    "table",
    "wall",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_manifest", type=Path, required=True)
    parser.add_argument("--relaxed_report", type=Path, required=True)
    parser.add_argument("--robot_motion_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target_hours", type=float, default=8.0)
    parser.add_argument("--max_hours", type=float, default=8.1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def duration_sec(row):
    frames = float(row.get("num_frames") or 0.0)
    fps = float(row.get("fps") or 0.0)
    return frames / fps if fps > 0 else 0.0


def normalize_family(name):
    stem = Path(name).stem
    if "__" in stem:
        stem = stem.split("__", 1)[0]
    for suffix in ("_M", "_R"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    parts = stem.split("_")
    if parts and parts[-1].isdigit():
        stem = "_".join(parts[:-1])
    return stem.lower()


def normalize_motion_text(value):
    text = str(value).lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def has_external_support_dependency(value):
    text = normalize_motion_text(value)
    if any(phrase in text for phrase in EXTERNAL_SUPPORT_PHRASES):
        return True

    tokens = {token for token in text.split("_") if token}
    return bool(tokens & EXTERNAL_SUPPORT_ACTION_TOKENS) and bool(
        tokens & EXTERNAL_SUPPORT_OBJECT_TOKENS
    )


def read_source_metadata(source_manifest):
    rows = read_csv(source_manifest)
    by_file = {}
    category_seconds = defaultdict(float)
    for row in rows:
        filename = Path(row["source_file"]).name
        by_file[filename] = row
        category_seconds[row["category"]] += float(row["duration_sec"])
    total = sum(category_seconds.values())
    ratios = {
        category: seconds / total
        for category, seconds in category_seconds.items()
        if total > 0
    }
    return by_file, ratios


def enrich_candidates(report_rows, source_meta):
    candidates = []
    for row in report_rows:
        if row["relaxed_status"] not in {"pass", "tag"}:
            continue
        filename = Path(row["relative_file"]).name
        source = source_meta.get(filename, {})
        if has_external_support_dependency(
            f"{filename} {source.get('motion_family', '')} {source.get('normalized_family', '')}"
        ):
            continue
        seconds = duration_sec(row)
        if seconds <= 0:
            continue
        category = source.get("category", "unknown")
        candidates.append(
            {
                **row,
                "filename": filename,
                "category": category,
                "duration_sec": seconds,
                "duration_hours": seconds / 3600.0,
                "motion_family": source.get("motion_family", normalize_family(filename)),
                "normalized_family": source.get("normalized_family", normalize_family(filename)),
                "actor": source.get("actor", ""),
                "source_path": source.get("source_path", ""),
                "source_category": category,
            }
        )
    return candidates


def candidate_sort_key(row, category_counts, family_counts, actor_counts):
    status_rank = 0 if row["relaxed_status"] == "pass" else 1
    return (
        status_rank,
        family_counts[row["normalized_family"]],
        actor_counts[row["actor"]],
        category_counts[row["category"]],
        abs(row["duration_sec"] - 8.0),
        row["filename"],
    )


def select_subset(candidates, source_ratios, target_seconds, max_seconds):
    by_category = defaultdict(list)
    for row in candidates:
        by_category[row["category"]].append(row)

    category_targets = {
        category: target_seconds * source_ratios.get(category, 0.0)
        for category in by_category
    }
    # If a new category is absent from source ratios, keep a small nonzero target.
    for category in by_category:
        if category_targets[category] == 0.0:
            category_targets[category] = min(target_seconds * 0.02, 10 * 60)

    selected = []
    selected_files = set()
    category_seconds = defaultdict(float)
    category_counts = Counter()
    family_counts = Counter()
    actor_counts = Counter()

    for category in sorted(by_category, key=lambda c: category_targets[c], reverse=True):
        pool = by_category[category][:]
        while pool and category_seconds[category] < category_targets[category]:
            pool.sort(key=lambda row: candidate_sort_key(row, category_counts, family_counts, actor_counts))
            chosen = None
            for row in pool:
                if row["filename"] in selected_files:
                    continue
                if sum(item["duration_sec"] for item in selected) + row["duration_sec"] <= max_seconds:
                    chosen = row
                    break
            if chosen is None:
                break
            selected.append(chosen)
            selected_files.add(chosen["filename"])
            category_seconds[category] += chosen["duration_sec"]
            category_counts[category] += 1
            family_counts[chosen["normalized_family"]] += 1
            actor_counts[chosen["actor"]] += 1
            pool.remove(chosen)

    remaining = [row for row in candidates if row["filename"] not in selected_files]
    while sum(item["duration_sec"] for item in selected) < target_seconds and remaining:
        remaining.sort(key=lambda row: candidate_sort_key(row, category_counts, family_counts, actor_counts))
        chosen = None
        total = sum(item["duration_sec"] for item in selected)
        for row in remaining:
            if total + row["duration_sec"] <= max_seconds:
                chosen = row
                break
        if chosen is None:
            break
        selected.append(chosen)
        selected_files.add(chosen["filename"])
        category_seconds[chosen["category"]] += chosen["duration_sec"]
        category_counts[chosen["category"]] += 1
        family_counts[chosen["normalized_family"]] += 1
        actor_counts[chosen["actor"]] += 1
        remaining.remove(chosen)

    return selected


def symlink_selected(selected, robot_motion_root, output_dir):
    motions_dir = output_dir / "motions"
    motions_dir.mkdir(parents=True, exist_ok=True)
    for row in selected:
        src = (robot_motion_root / row["relative_file"]).resolve()
        dst = motions_dir / row["filename"]
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)


def write_outputs(selected, candidates, source_ratios, args):
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    symlink_selected(selected, args.robot_motion_root, args.output_dir)

    manifest_fields = [
        "relative_file",
        "filename",
        "relaxed_status",
        "relaxed_reasons",
        "strict_status",
        "reject_reasons",
        "tag_reasons",
        "category",
        "duration_sec",
        "duration_hours",
        "num_frames",
        "fps",
        "motion_family",
        "normalized_family",
        "actor",
        "source_path",
    ]
    write_csv(args.output_dir / "manifest.csv", selected, manifest_fields)
    write_csv(
        args.output_dir / "category_summary.csv",
        summarize_categories(selected, source_ratios),
        [
            "category",
            "source_ratio",
            "selected_files",
            "selected_hours",
            "pass_files",
            "tag_files",
            "unique_families",
            "unique_actors",
        ],
    )
    with (args.output_dir / "selected_robot_motions.txt").open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(str((args.robot_motion_root / row["relative_file"]).resolve()) + "\n")

    summary = {
        "target_hours": args.target_hours,
        "max_hours": args.max_hours,
        "selected_files": len(selected),
        "selected_hours": round(sum(row["duration_sec"] for row in selected) / 3600.0, 6),
        "available_keep_files": len(candidates),
        "available_keep_hours": round(sum(row["duration_sec"] for row in candidates) / 3600.0, 6),
        "status_counts": dict(Counter(row["relaxed_status"] for row in selected)),
        "category_counts": dict(Counter(row["category"] for row in selected)),
        "outputs": {
            "motions": str(args.output_dir / "motions"),
            "manifest": str(args.output_dir / "manifest.csv"),
            "selected_robot_motions": str(args.output_dir / "selected_robot_motions.txt"),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def summarize_categories(rows, source_ratios):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    out = []
    for category, items in sorted(grouped.items()):
        out.append(
            {
                "category": category,
                "source_ratio": f"{source_ratios.get(category, 0.0):.6f}",
                "selected_files": len(items),
                "selected_hours": f"{sum(row['duration_sec'] for row in items) / 3600.0:.6f}",
                "pass_files": sum(row["relaxed_status"] == "pass" for row in items),
                "tag_files": sum(row["relaxed_status"] == "tag" for row in items),
                "unique_families": len({row["normalized_family"] for row in items}),
                "unique_actors": len({row["actor"] for row in items if row["actor"]}),
            }
        )
    return out


def main():
    args = parse_args()
    source_meta, source_ratios = read_source_metadata(args.source_manifest)
    report_rows = read_csv(args.relaxed_report)
    candidates = enrich_candidates(report_rows, source_meta)
    selected = select_subset(
        candidates,
        source_ratios,
        target_seconds=args.target_hours * 3600.0,
        max_seconds=args.max_hours * 3600.0,
    )
    summary = write_outputs(selected, candidates, source_ratios, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
