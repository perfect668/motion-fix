#!/usr/bin/env python3
"""Build a symlinked SMPL-X subset for flat-ground, self-motion training."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


GROUP_PATTERNS = {
    "aerial_acrobatics": (
        "cartwheel",
        "flip",
        "somersault",
        "backflip",
        "frontflip",
        "roll",
        "gymnast",
        "acrobat",
        "jump",
        "hop",
        "burpee",
    ),
    "run_jog": ("run", "running", "jog", "sprint"),
    "dance": ("dance", "dancing", "hiphop", "vouge", "retro"),
    "combat_kick": (
        "kick",
        "martial",
        "boxing",
        "kungfu",
        "wushu",
        "kendo",
        "taichi",
        "karate",
        "combat",
        "capoera",
    ),
    "exercise_sport": ("exercise", "sport", "training"),
}

# These terms indicate a prop, terrain, external support, unsuitable low posture,
# or deliberately impaired motion. Matching is done before the include groups.
EXCLUDE_TERMS = (
    "grab",
    "object",
    "chair",
    "table",
    "box",
    "crate",
    "bucket",
    "tool",
    "prop",
    "carry",
    "hold",
    "pick",
    "place",
    "lift",
    "push",
    "pull",
    "throw",
    "catch",
    "door",
    "wall",
    "stairs",
    "stair",
    "bench",
    "obstacle",
    "vault",
    "bar",
    "rail",
    "ladder",
    "crawl",
    "kneel",
    "sit",
    "lying",
    "on_ground",
    "injured",
    "inj_",
    "hurt",
    "limp",
    "horse",
    "bike",
    "bicycle",
)


def normalise(path: Path) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(path))


def classify(path: Path) -> tuple[str | None, str | None]:
    text = normalise(path)
    matched_excludes = [term for term in EXCLUDE_TERMS if term in text]
    if matched_excludes:
        return None, "excluded:" + ",".join(matched_excludes)
    for group, terms in GROUP_PATTERNS.items():
        if any(term in text for term in terms):
            return group, None
    return None, "not_self_motion"


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        parser.error(f"Input root does not exist: {input_root}")
    prepare_output(args.output_dir, args.overwrite)

    motions_dir = args.output_dir / "motions"
    rows: list[dict[str, str]] = []
    skipped: Counter[str] = Counter()
    selected: Counter[str] = Counter()
    for source in sorted(input_root.rglob("*_stageii.npz")):
        relative = source.relative_to(input_root)
        group, reason = classify(relative)
        if group is None:
            skipped[reason or "unknown"] += 1
            continue
        target = motions_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
        selected[group] += 1
        rows.append({"relative_file": str(relative), "group": group})

    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_file", "group"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "input_root": str(input_root),
        "selection_policy": "flat-ground self-motion only",
        "selected_count": len(rows),
        "group_counts": dict(sorted(selected.items())),
        "skipped_counts": dict(sorted(skipped.items())),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
