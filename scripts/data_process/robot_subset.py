"""Shared relaxed-report loading and subset materialization primitives."""

from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RelaxedReport:
    rows: list[dict]
    eligible_by_filename: dict[str, dict]
    contract: dict[str, str]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, lines):
    with Path(path).open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


def filter_contract(rows):
    versions = {row.get("filter_schema_version", "legacy") or "legacy" for row in rows}
    policies = {row.get("filter_policy", "legacy") or "legacy" for row in rows}
    if len(versions) > 1 or len(policies) > 1:
        raise ValueError(f"Mixed filter contracts: versions={sorted(versions)}, policies={sorted(policies)}")
    return {
        "schema_version": next(iter(versions), "legacy"),
        "policy": next(iter(policies), "legacy"),
    }


def load_relaxed_report(path):
    rows = read_csv(path)
    eligible = {
        Path(row["relative_file"]).name: row
        for row in rows
        if row.get("relaxed_status") in {"pass", "tag"}
    }
    return RelaxedReport(
        rows=rows,
        eligible_by_filename=eligible,
        contract=filter_contract(rows),
    )


def prepare_output_dir(path, overwrite):
    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output path exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def symlink_file(src, dst):
    src = Path(src).resolve()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def materialize_robot_links(selected, robot_motion_root, output_dir):
    motions_dir = Path(output_dir) / "motions"
    for row in selected:
        symlink_file(
            Path(robot_motion_root) / row["relative_file"],
            motions_dir / row["filename"],
        )
    return motions_dir
