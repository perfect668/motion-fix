"""Select motions suitable for flat-ground, environment-free whole-body tracking."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from data_process.motion_semantics import describe_motion, has_external_support_dependency


EXCLUDED_CATEGORIES = {
    "ground_low_posture",
    "object_manipulation_carry",
    "obstacle_contact_avoidance",
}

EXCLUDED_TOKENS = {
    "against",
    "bar",
    "box",
    "brace",
    "carry",
    "chair",
    "climb",
    "crawl",
    "crate",
    "door",
    "grab",
    "hang",
    "hold",
    "kneel",
    "ladder",
    "lean",
    "lie",
    "lift",
    "mop",
    "object",
    "pick",
    "place",
    "prop",
    "rail",
    "rest",
    "sit",
    "stairs",
    "stair",
    "support",
    "table",
    "tool",
    "upstairs",
    "downstairs",
    "wall",
}


def exclusion_reasons(path: Path) -> list[str]:
    semantics = describe_motion(path)
    tokens = {token for token in path.stem.lower().replace("-", "_").split("_") if token}
    reasons = []
    if semantics.category in EXCLUDED_CATEGORIES:
        reasons.append(f"category:{semantics.category}")
    if semantics.external_support_dependency or has_external_support_dependency(path):
        reasons.append("external_support_dependency")
    matched_tokens = sorted(tokens & EXCLUDED_TOKENS)
    if matched_tokens:
        reasons.append("environment_or_object_tokens:" + ",".join(matched_tokens))
    return reasons


def create_selected_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to replace nonmatching selection entry: {destination}")
    destination.symlink_to(source.resolve())


def move_to_quarantine(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Quarantine destination already exists: {destination}")
    shutil.move(str(source), str(destination))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--suffix", choices=[".npz", ".pkl"], required=True)
    parser.add_argument("--stageii_only", action="store_true")
    parser.add_argument("--report_csv", type=Path, required=True)
    parser.add_argument("--selected_root", type=Path)
    parser.add_argument("--quarantine_root", type=Path)
    parser.add_argument("--move_rejected", action="store_true")
    args = parser.parse_args()

    if args.move_rejected and args.quarantine_root is None:
        parser.error("--move_rejected requires --quarantine_root")
    if not args.input_root.is_dir():
        parser.error(f"Input root does not exist: {args.input_root}")

    files = sorted(
        path
        for path in args.input_root.rglob(f"*{args.suffix}")
        if path.is_file() and (not args.stageii_only or path.name.endswith("_stageii.npz"))
    )
    rows = []
    for path in files:
        relative_path = path.relative_to(args.input_root)
        reasons = exclusion_reasons(path)
        status = "reject" if reasons else "keep"
        rows.append(
            {
                "relative_file": str(relative_path),
                "status": status,
                "reasons": ";".join(reasons),
                "category": describe_motion(path).category,
            }
        )

        if status == "keep" and args.selected_root is not None:
            create_selected_symlink(path, args.selected_root / relative_path)
        if status == "reject" and args.move_rejected:
            move_to_quarantine(path, args.quarantine_root / relative_path)

    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.report_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_file", "status", "reasons", "category"])
        writer.writeheader()
        writer.writerows(rows)

    kept = sum(row["status"] == "keep" for row in rows)
    rejected = len(rows) - kept
    print(f"Scanned: {len(rows)}")
    print(f"Keep: {kept}")
    print(f"Reject: {rejected}")
    print(f"Report: {args.report_csv}")


if __name__ == "__main__":
    main()
