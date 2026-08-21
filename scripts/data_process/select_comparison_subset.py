#!/usr/bin/env python3
"""Create a reproducible, stratified SMPL-X subset for retargeter comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_QUOTAS = {
    "run_jog": 92,
    "aerial_acrobatics": 53,
    "combat_kick": 48,
    "dance": 7,
}


def _stable_key(relative_file: str) -> str:
    return hashlib.sha256(relative_file.encode("utf-8")).hexdigest()


def _motion_metadata(path: Path) -> tuple[int, float]:
    """Read only the fields needed for sampling, without loading object arrays."""
    with np.load(path, allow_pickle=False) as data:
        frames = int(data["root_orient"].shape[0])
        fps = float(np.asarray(data["mocap_frame_rate"]).reshape(-1)[0])
    if frames < 1 or fps <= 0.0:
        raise ValueError(f"invalid frames/fps: {path}")
    return frames, fps


def _allocate_proportional(bucket_sizes: dict[str, int], quota: int) -> dict[str, int]:
    total = sum(bucket_sizes.values())
    if quota > total:
        raise ValueError(f"quota {quota} exceeds {total} candidates")
    raw = {key: quota * size / total for key, size in bucket_sizes.items()}
    allocated = {key: int(value) for key, value in raw.items()}
    remaining = quota - sum(allocated.values())
    order = sorted(
        bucket_sizes,
        key=lambda key: (raw[key] - allocated[key], bucket_sizes[key], key),
        reverse=True,
    )
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def _duration_stratified(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    """Pick deterministic duration quantiles so short and long clips both appear."""
    ordered = sorted(rows, key=lambda row: (float(row["duration_sec"]), str(row["hash"])))
    if count == 0:
        return []
    return [ordered[(2 * index + 1) * len(ordered) // (2 * count)] for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="1444-set directory with manifest.csv and motions/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    manifest = source / "manifest.csv"
    motions = source / "motions"
    if not manifest.is_file() or not motions.is_dir():
        parser.error("source must contain manifest.csv and motions/")
    if args.output.exists():
        if not args.overwrite:
            parser.error(f"output already exists: {args.output}")
        shutil.rmtree(args.output)

    rows_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    with manifest.open(newline="", encoding="utf-8") as stream:
        for item in csv.DictReader(stream):
            relative = item["relative_file"]
            group = item["group"]
            path = motions / relative
            frames, fps = _motion_metadata(path)
            rows_by_group[group].append({
                "relative_file": relative,
                "group": group,
                "source_dataset": Path(relative).parts[0],
                "num_frames": frames,
                "fps": fps,
                "duration_sec": frames / fps,
                "hash": _stable_key(relative),
            })

    unknown = set(rows_by_group) - set(DEFAULT_QUOTAS)
    if unknown:
        parser.error(f"unconfigured groups: {sorted(unknown)}")

    selected: list[dict[str, object]] = []
    for group, quota in DEFAULT_QUOTAS.items():
        candidates = rows_by_group[group]
        by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in candidates:
            by_source[str(row["source_dataset"])].append(row)
        allocation = _allocate_proportional(
            {name: len(items) for name, items in by_source.items()}, quota,
        )
        for source_name, count in allocation.items():
            selected.extend(_duration_stratified(by_source[source_name], count))

    selected.sort(key=lambda row: (str(row["group"]), str(row["relative_file"])))
    if len(selected) != sum(DEFAULT_QUOTAS.values()):
        raise RuntimeError(f"selected {len(selected)}, expected {sum(DEFAULT_QUOTAS.values())}")

    output = args.output.resolve()
    target_motions = output / "motions"
    target_motions.mkdir(parents=True)
    for order, row in enumerate(selected, start=1):
        relative = Path(str(row["relative_file"]))
        target = target_motions / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to((motions / relative).resolve())
        row["selection_order"] = order
        row.pop("hash")

    fields = ["selection_order", "relative_file", "group", "source_dataset", "num_frames", "fps", "duration_sec"]
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    group_counts = {group: sum(row["group"] == group for row in selected) for group in DEFAULT_QUOTAS}
    summary = {
        "source": str(source),
        "selection_policy": "stratified by original group, source dataset, and clip duration",
        "selected_count": len(selected),
        "group_quotas": DEFAULT_QUOTAS,
        "group_counts": group_counts,
        "duration_sec": {
            "min": min(float(row["duration_sec"]) for row in selected),
            "max": max(float(row["duration_sec"]) for row in selected),
            "mean": sum(float(row["duration_sec"]) for row in selected) / len(selected),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
