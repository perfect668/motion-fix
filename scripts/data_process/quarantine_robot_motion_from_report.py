"""Move robot-motion files listed as rejected in a GMR relaxed report."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_csv", type=Path, required=True)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--quarantine_root", type=Path, required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    rejected = []
    with args.report_csv.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("relaxed_status") == "reject":
                rejected.append(Path(row["relative_file"]))

    moved = 0
    missing = []
    for relative_path in rejected:
        source = args.input_root / relative_path
        destination = args.quarantine_root / relative_path
        if not source.is_file():
            missing.append(str(relative_path))
            continue
        if destination.exists():
            raise FileExistsError(f"Quarantine destination already exists: {destination}")
        if not args.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        moved += 1

    print(f"Rejected rows: {len(rejected)}")
    print(f"Moved: {moved}")
    print(f"Missing: {len(missing)}")
    for relative_path in missing[:20]:
        print(relative_path)
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
