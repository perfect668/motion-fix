"""Run the complete flat-output NE01 Laplacian retargeting pipeline.

The script combines the native SMPL-X motions in ``on_use/smplx`` with SMPL
motions converted from ``on_use/smpl``, pre-filters motions unsuitable for a
flat floor, and presents accepted files to the existing Laplacian dataset
entry point through a flat symlink directory.

All intermediate state is resumable. Existing PKL files are deliberately not
overwritten by the downstream dataset script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from tqdm import tqdm

from filter_nonflat_motions import inspect_motion
from smpl_to_smplx import convert_smpl_to_smplx


def _signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "source": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _read_jsonl(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if "flat_name" in row:
                records[row["flat_name"]] = row
    return records


def _replace_symlink(link: Path, source: Path) -> None:
    source = source.resolve()
    if link.is_symlink() and link.resolve() == source:
        return
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(source)


def _flat_names(files: list[tuple[str, Path, Path]]) -> list[tuple[str, Path]]:
    """Assign deterministic names, preserving the first basename in each group."""
    by_name: dict[str, list[tuple[str, Path, Path]]] = {}
    for item in files:
        by_name.setdefault(item[1].name, []).append(item)

    resolved: list[tuple[str, Path]] = []
    for basename, group in sorted(by_name.items()):
        # Prefer native SMPL-X for the unsuffixed name so existing output from
        # earlier SMPL-X samples remains a valid resume marker.
        group.sort(key=lambda item: (item[0] != "smplx", item[2].as_posix()))
        for index, (kind, path, identity) in enumerate(group):
            if index == 0:
                flat_name = basename
            else:
                digest = hashlib.sha1(
                    f"{kind}:{identity.as_posix()}".encode("utf-8")
                ).hexdigest()[:10]
                flat_name = f"{path.stem}__{digest}{path.suffix}"
            resolved.append((flat_name, path))
    return resolved


def _convert_smpl_tree(src_root: Path, target_root: Path, error_log: Path) -> None:
    sources = sorted(src_root.rglob("*.npz"))
    failures = []
    for source in tqdm(sources, desc="SMPL -> SMPL-X"):
        relative = source.relative_to(src_root)
        target = target_root / relative
        source_stat = source.stat()
        if (
            target.exists()
            and target.stat().st_mtime_ns >= source_stat.st_mtime_ns
            and target.stat().st_size > 0
        ):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            convert_smpl_to_smplx(str(source), str(target), "neutral")
            os.utime(target, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
        except Exception as exc:
            target.unlink(missing_ok=True)
            failures.append(
                f"{source}\n{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
    if failures:
        error_log.write_text("\n\n".join(failures), encoding="utf-8")
        print(f"SMPL conversion failures: {len(failures)}; see {error_log}")
    else:
        error_log.unlink(missing_ok=True)


def _write_filter_csv(path: Path, records: dict[str, dict]) -> None:
    rows = sorted(records.values(), key=lambda row: row["flat_name"])
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_root",
        default="/home/user/桌面/humanoid_motion_sources/on_use",
    )
    parser.add_argument(
        "--output",
        default="/home/user/桌面/humanoid_retarget_results/in_use_all/small_sample",
    )
    parser.add_argument(
        "--work_dir",
        default="work/all_on_use_laplacian_flat",
    )
    parser.add_argument("--robot", default="ne01")
    parser.add_argument("--tgt_fps", type=float, default=50.0)
    parser.add_argument("--sample_fps", type=float, default=30.0)
    parser.add_argument("--num_cpus", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reject_in_place", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    source_root = Path(args.source_root).resolve()
    smpl_root = source_root / "smpl"
    native_smplx_root = source_root / "smplx"
    work = (repo_root / args.work_dir).resolve()
    converted_root = work / "smplx_from_smpl"
    candidate_flat = work / "candidates_flat"
    accepted_flat = work / "accepted_flat"
    output = Path(args.output).resolve()
    state_jsonl = work / "filter_state.jsonl"
    report_csv = work / "filter_report.csv"
    body_models = repo_root / "assets" / "body_models"

    if not smpl_root.is_dir() or not native_smplx_root.is_dir():
        raise FileNotFoundError(
            f"Expected both {smpl_root} and {native_smplx_root}"
        )
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    candidate_flat.mkdir(parents=True, exist_ok=True)
    accepted_flat.mkdir(parents=True, exist_ok=True)

    _convert_smpl_tree(smpl_root, converted_root, work / "smpl_conversion_failures.txt")

    sources: list[tuple[str, Path, Path]] = []
    for path in sorted(native_smplx_root.rglob("*.npz")):
        if not path.name.endswith("_stagei.npz"):
            sources.append(("smplx", path, path.relative_to(native_smplx_root)))
    for path in sorted(converted_root.rglob("*.npz")):
        if not path.name.endswith("_stagei.npz"):
            sources.append(("smpl", path, path.relative_to(converted_root)))
    flat_sources = _flat_names(sources)
    active_names = {name for name, _ in flat_sources}

    for stale in candidate_flat.glob("*.npz"):
        if stale.name not in active_names:
            stale.unlink()
    for flat_name, source in flat_sources:
        _replace_symlink(candidate_flat / flat_name, source)

    records = _read_jsonl(state_jsonl)
    with state_jsonl.open("a", encoding="utf-8") as state_handle:
        for flat_name, source in tqdm(flat_sources, desc="Flat-floor pre-filter"):
            signature = _signature(source)
            old = records.get(flat_name)
            reusable = old is not None and all(
                old.get(key) == value for key, value in signature.items()
            )
            if reusable:
                row = old
            else:
                try:
                    status, metrics = inspect_motion(
                        source,
                        body_models,
                        sample_fps=args.sample_fps,
                        reject_in_place=args.reject_in_place,
                    )
                except Exception as exc:
                    status = "error"
                    metrics = {"reason": f"{type(exc).__name__}: {exc}"}
                row = {
                    "flat_name": flat_name,
                    **signature,
                    "status": status,
                    **metrics,
                }
                state_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                state_handle.flush()
                records[flat_name] = row

            accepted_link = accepted_flat / flat_name
            if row.get("status") == "accept":
                _replace_symlink(accepted_link, source)
            elif accepted_link.exists() or accepted_link.is_symlink():
                accepted_link.unlink()

    for stale in accepted_flat.glob("*.npz"):
        if stale.name not in active_names:
            stale.unlink()
    _write_filter_csv(report_csv, records)

    accepted = sum(
        records.get(name, {}).get("status") == "accept" for name in active_names
    )
    existing = len(list(output.glob("*.pkl")))
    print(
        f"Candidates: {len(flat_sources)}; accepted: {accepted}; "
        f"existing top-level PKL: {existing}"
    )
    command = [
        sys.executable,
        str(repo_root / "scripts" / "smplx_to_robot_laplacian_dataset.py"),
        "--src_folder",
        str(accepted_flat),
        "--tgt_folder",
        str(output),
        "--robot",
        args.robot,
        "--tgt_fps",
        str(args.tgt_fps),
        "--num_cpus",
        str(args.num_cpus),
        "--device",
        args.device,
    ]
    subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
