"""Select high-dynamic SMPL-X and robot-motion subsets.

This helper is intentionally small and dataset-oriented. It has two modes:

1. ``candidates`` scores SMPL-X files from a Sonic manifest and writes a
   symlinked candidate SMPL-X folder for retargeting.
2. ``final`` reads the post-retarget relaxed report and writes the final
   symlinked robot-motion subset.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .motion_semantics import (
        HIGH_DYNAMIC_CATEGORIES,
        dynamic_group_from_text,
        has_external_support_dependency,
        motion_family_from_path,
        normalize_motion_text,
        normalized_family,
    )
except ImportError:
    from motion_semantics import (
        HIGH_DYNAMIC_CATEGORIES,
        dynamic_group_from_text,
        has_external_support_dependency,
        motion_family_from_path,
        normalize_motion_text,
        normalized_family,
    )
try:
    from .robot_subset import (
        load_relaxed_report,
        prepare_output_dir,
        read_csv,
        symlink_file,
        write_csv,
        write_text,
    )
except ImportError:
    from robot_subset import (
        load_relaxed_report,
        prepare_output_dir,
        read_csv,
        symlink_file,
        write_csv,
        write_text,
    )

CATEGORY_WEIGHTS = {
    "jump": 7.0,
    "dance": 6.0,
    "locomotion_jog_run": 5.0,
    "exercise_sport": 4.0,
    "kick_throw_stoop": 4.0,
    "obstacle_contact_avoidance": 3.0,
}

KEYWORD_WEIGHTS = {
    "high_jump": 5.0,
    "broad_jump": 5.0,
    "frog_jump": 5.0,
    "jumping": 4.5,
    "jump": 4.0,
    "hop": 3.5,
    "dance": 4.0,
    "hiphop": 4.0,
    "vouge": 4.0,
    "retro": 3.0,
    "latino": 3.0,
    "western": 2.5,
    "macarena": 3.0,
    "jog": 3.5,
    "run": 3.5,
    "sprint": 4.0,
    "kick": 3.0,
}

EXCLUDE_TOKENS = {
    "crawl",
    "crawling",
    "lie",
    "lying",
    "upstairs",
    "downstairs",
    "stairs",
    "sit",
    "sitting",
    "on_all_fours",
}

GROUP_QUOTAS = {
    "jump": 6,
    "dance": 6,
    "run_jog": 5,
    "sport_kick_obstacle": 3,
}

def has_excluded_token(value):
    text = normalize_motion_text(value)
    return any(token in text for token in EXCLUDE_TOKENS)


def keyword_score(text):
    normalized = normalize_motion_text(text)
    return sum(weight for token, weight in KEYWORD_WEIGHTS.items() if token in normalized)


def load_motion_metrics(path):
    data = np.load(path, allow_pickle=True)
    trans = np.asarray(data["trans"], dtype=np.float64)
    fps = float(np.asarray(data["mocap_frame_rate"]).item())
    frame_count = int(trans.shape[0])
    duration = frame_count / fps if fps > 0 else 0.0

    if frame_count > 1 and fps > 0:
        root_speed = np.linalg.norm(np.diff(trans, axis=0) * fps, axis=1)
        max_root_speed = float(np.max(root_speed))
        mean_root_speed = float(np.mean(root_speed))
        root_path_length = float(np.sum(np.linalg.norm(np.diff(trans[:, :2], axis=0), axis=1)))
        vertical_speed = float(np.max(np.abs(np.diff(trans[:, 2]) * fps)))
    else:
        max_root_speed = 0.0
        mean_root_speed = 0.0
        root_path_length = 0.0
        vertical_speed = 0.0

    root_height_span = float(np.percentile(trans[:, 2], 95) - np.percentile(trans[:, 2], 5))

    pose_body = np.asarray(data["pose_body"], dtype=np.float64)
    root_orient = np.asarray(data["root_orient"], dtype=np.float64)
    if frame_count > 1 and fps > 0:
        pose_speed = float(np.percentile(np.linalg.norm(np.diff(pose_body, axis=0), axis=1) * fps, 95))
        root_rot_speed = float(np.percentile(np.linalg.norm(np.diff(root_orient, axis=0), axis=1) * fps, 95))
    else:
        pose_speed = 0.0
        root_rot_speed = 0.0

    return {
        "fps": fps,
        "num_frames": frame_count,
        "duration_sec": duration,
        "max_root_speed": max_root_speed,
        "mean_root_speed": mean_root_speed,
        "root_path_length": root_path_length,
        "vertical_speed": vertical_speed,
        "root_height_span": root_height_span,
        "pose_speed_p95": pose_speed,
        "root_rot_speed_p95": root_rot_speed,
    }


def dynamic_score(row, metrics):
    text = " ".join(
        [
            row.get("source_file", ""),
            row.get("motion_family", ""),
            row.get("normalized_family", ""),
            row.get("category", ""),
        ]
    )
    score = CATEGORY_WEIGHTS.get(row.get("category", ""), 0.0)
    score += keyword_score(text)
    score += min(metrics["max_root_speed"], 5.0) * 1.2
    score += min(metrics["mean_root_speed"], 3.0) * 0.8
    score += min(metrics["vertical_speed"], 4.0) * 1.5
    score += min(metrics["root_height_span"], 1.0) * 4.0
    score += min(metrics["pose_speed_p95"], 40.0) * 0.08
    score += min(metrics["root_rot_speed_p95"], 20.0) * 0.05
    return score


def select_candidates(args):
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    smplx_out = output_dir / "smplx"
    smplx_out.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = Counter()
    for row in read_csv(args.source_manifest):
        source_file = Path(row["source_file"]).name
        smplx_name = Path(source_file).with_suffix(".npz").name
        smplx_path = args.smplx_root / smplx_name
        if not smplx_path.exists():
            skipped["missing_smplx"] += 1
            continue

        text = " ".join(
            [
                source_file,
                row.get("motion_family", ""),
                row.get("normalized_family", ""),
                row.get("category", ""),
            ]
        )
        if row.get("category") not in HIGH_DYNAMIC_CATEGORIES and keyword_score(text) <= 0:
            skipped["not_high_dynamic"] += 1
            continue
        if has_external_support_dependency(text):
            skipped["external_support"] += 1
            continue
        if has_excluded_token(text):
            skipped["excluded_token"] += 1
            continue

        metrics = load_motion_metrics(smplx_path)
        if metrics["duration_sec"] < args.min_duration_sec:
            skipped["too_short"] += 1
            continue
        if metrics["duration_sec"] > args.max_duration_sec:
            skipped["too_long"] += 1
            continue

        score = dynamic_score(row, metrics)
        rows.append(
            {
                "smplx_file": smplx_name,
                "smplx_path": str(smplx_path),
                "source_file": source_file,
                "source_path": row.get("source_path", ""),
                "motion_family": row.get("motion_family") or motion_family_from_path(source_file),
                "normalized_family": row.get("normalized_family")
                or normalized_family(motion_family_from_path(source_file)),
                "actor": row.get("actor", ""),
                "is_mirror": row.get("is_mirror", ""),
                "category": row.get("category", ""),
                "dynamic_group": dynamic_group_from_text(text, row.get("category", "")),
                "dynamic_score": round(score, 6),
                **{key: round(value, 6) if isinstance(value, float) else value for key, value in metrics.items()},
            }
        )

    rows.sort(key=lambda row: (-float(row["dynamic_score"]), row["normalized_family"], row["smplx_file"]))

    selected = []
    seen_families = Counter()
    seen_actors = Counter()
    seen_files = set()

    def add_from_pool(pool, limit):
        nonlocal selected
        pool = [row for row in pool if row["smplx_file"] not in seen_files]
        pool.sort(
            key=lambda row: (
                seen_families[row["normalized_family"]],
                seen_actors[row["actor"]],
                -float(row["dynamic_score"]),
                row["smplx_file"],
            )
        )
        for row in pool:
            if len(selected) >= limit:
                break
            selected.append(row)
            seen_files.add(row["smplx_file"])
            seen_families[row["normalized_family"]] += 1
            seen_actors[row["actor"]] += 1

    per_group_targets = {
        "jump": max(1, int(args.candidate_count * 0.30)),
        "dance": max(1, int(args.candidate_count * 0.30)),
        "run_jog": max(1, int(args.candidate_count * 0.25)),
    }
    per_group_targets["sport_kick_obstacle"] = max(
        0, args.candidate_count - sum(per_group_targets.values())
    )

    for group, target in per_group_targets.items():
        pool = [row for row in rows if row["dynamic_group"] == group]
        add_from_pool(pool, min(args.candidate_count, len(selected) + target))

    add_from_pool(rows, args.candidate_count)
    selected = selected[: args.candidate_count]

    for row in selected:
        symlink_file(row["smplx_path"], smplx_out / row["smplx_file"])

    fieldnames = list(selected[0].keys()) if selected else []
    write_csv(output_dir / "manifest.csv", selected, fieldnames)
    write_text(output_dir / "selected_smplx.txt", [row["smplx_file"] for row in selected])

    summary = {
        "source_manifest": str(args.source_manifest),
        "smplx_root": str(args.smplx_root),
        "candidate_count": len(selected),
        "available_high_dynamic": len(rows),
        "skipped": dict(skipped),
        "category_counts": dict(Counter(row["category"] for row in selected)),
        "dynamic_group_counts": dict(Counter(row["dynamic_group"] for row in selected)),
        "duration_sec": round(sum(float(row["duration_sec"]) for row in selected), 3),
        "outputs": {
            "smplx": str(smplx_out),
            "manifest": str(output_dir / "manifest.csv"),
            "selected_smplx": str(output_dir / "selected_smplx.txt"),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def group_for(row):
    if row.get("dynamic_group"):
        return row["dynamic_group"]
    text = " ".join([row.get("filename", ""), row.get("category", "")])
    return dynamic_group_from_text(text, row.get("category", ""))


def semantic_quality_rank(row):
    action_text = normalize_motion_text(
        " ".join(
            [
                row.get("filename", ""),
                row.get("motion_family", ""),
                row.get("normalized_family", ""),
            ]
        )
    )
    if "fall" in action_text:
        return 3
    if row.get("group") == "sport_kick_obstacle":
        if any(token in action_text for token in ("kick", "throw", "exercise", "burpee", "combat")):
            return 0
        if any(token in action_text for token in ("obstacle", "avoid", "bump")):
            return 1
        if "stoop" in action_text:
            return 2
    return 0


def select_final(args):
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    motions_out = output_dir / "motions"
    smplx_out = output_dir / "smplx"
    motions_out.mkdir(parents=True, exist_ok=True)
    smplx_out.mkdir(parents=True, exist_ok=True)

    candidate_rows = read_csv(args.candidate_manifest)
    by_pkl_name = {Path(row["smplx_file"]).with_suffix(".pkl").name: row for row in candidate_rows}

    relaxed_report = load_relaxed_report(args.relaxed_report)
    report_rows = []
    for pkl_name, row in relaxed_report.eligible_by_filename.items():
        candidate = by_pkl_name.get(pkl_name)
        if not candidate:
            continue
        report_rows.append(
            {
                **candidate,
                "filename": pkl_name,
                "relative_file": row["relative_file"],
                "relaxed_status": row["relaxed_status"],
                "relaxed_reasons": row.get("relaxed_reasons", ""),
                "strict_status": row.get("strict_status", ""),
                "reject_reasons": row.get("reject_reasons", ""),
                "tag_reasons": row.get("tag_reasons", ""),
                "robot_num_frames": row.get("num_frames", ""),
                "robot_fps": row.get("fps", ""),
            }
        )

    for row in report_rows:
        row["group"] = group_for(row)

    selected = []
    selected_files = set()
    family_counts = Counter()
    actor_counts = Counter()

    def sort_key(row):
        status_rank = 0 if row["relaxed_status"] == "pass" else 1
        strict_status_rank = {"pass": 0, "tag": 1, "reject": 2}.get(row.get("strict_status", ""), 3)
        return (
            status_rank,
            strict_status_rank,
            semantic_quality_rank(row),
            family_counts[row["normalized_family"]],
            actor_counts[row["actor"]],
            -float(row["dynamic_score"]),
            row["filename"],
        )

    def add_group(group, target):
        pool = [row for row in report_rows if row["group"] == group and row["filename"] not in selected_files]
        while len([row for row in selected if row["group"] == group]) < target and pool:
            pool.sort(key=sort_key)
            row = pool.pop(0)
            selected.append(row)
            selected_files.add(row["filename"])
            family_counts[row["normalized_family"]] += 1
            actor_counts[row["actor"]] += 1

    quotas = dict(GROUP_QUOTAS)
    scale = args.target_count / sum(quotas.values())
    quotas = {key: int(round(value * scale)) for key, value in quotas.items()}
    while sum(quotas.values()) < args.target_count:
        quotas["jump"] += 1
    while sum(quotas.values()) > args.target_count:
        quotas["sport_kick_obstacle"] = max(0, quotas["sport_kick_obstacle"] - 1)

    for group, target in quotas.items():
        add_group(group, target)

    remaining = [row for row in report_rows if row["filename"] not in selected_files]
    while len(selected) < args.target_count and remaining:
        remaining.sort(key=sort_key)
        row = remaining.pop(0)
        selected.append(row)
        selected_files.add(row["filename"])
        family_counts[row["normalized_family"]] += 1
        actor_counts[row["actor"]] += 1

    selected = selected[: args.target_count]

    for row in selected:
        robot_src = args.robot_motion_root / row["relative_file"]
        smplx_src = Path(row["smplx_path"])
        symlink_file(robot_src, motions_out / row["filename"])
        symlink_file(smplx_src, smplx_out / Path(row["smplx_file"]).name)

    fieldnames = list(selected[0].keys()) if selected else []
    write_csv(output_dir / "manifest.csv", selected, fieldnames)
    write_text(output_dir / "selected_robot_motions.txt", [row["filename"] for row in selected])

    summary = {
        "candidate_manifest": str(args.candidate_manifest),
        "relaxed_report": str(args.relaxed_report),
        "robot_motion_root": str(args.robot_motion_root),
        "filter_contract": relaxed_report.contract,
        "target_count": args.target_count,
        "selected_count": len(selected),
        "eligible_relaxed_keep": len(report_rows),
        "relaxed_status_counts": dict(Counter(row["relaxed_status"] for row in selected)),
        "strict_status_counts": dict(Counter(row["strict_status"] for row in selected)),
        "category_counts": dict(Counter(row["category"] for row in selected)),
        "group_counts": dict(Counter(row["group"] for row in selected)),
        "duration_sec": round(sum(float(row["duration_sec"]) for row in selected), 3),
        "outputs": {
            "motions": str(motions_out),
            "smplx": str(smplx_out),
            "manifest": str(output_dir / "manifest.csv"),
            "selected_robot_motions": str(output_dir / "selected_robot_motions.txt"),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidates = subparsers.add_parser("candidates")
    candidates.add_argument("--source_manifest", type=Path, required=True)
    candidates.add_argument("--smplx_root", type=Path, required=True)
    candidates.add_argument("--output_dir", type=Path, required=True)
    candidates.add_argument("--candidate_count", type=int, default=80)
    candidates.add_argument("--min_duration_sec", type=float, default=1.0)
    candidates.add_argument("--max_duration_sec", type=float, default=30.0)
    candidates.add_argument("--overwrite", action="store_true")
    candidates.set_defaults(func=select_candidates)

    final = subparsers.add_parser("final")
    final.add_argument("--candidate_manifest", type=Path, required=True)
    final.add_argument("--relaxed_report", type=Path, required=True)
    final.add_argument("--robot_motion_root", type=Path, required=True)
    final.add_argument("--output_dir", type=Path, required=True)
    final.add_argument("--target_count", type=int, default=20)
    final.add_argument("--overwrite", action="store_true")
    final.set_defaults(func=select_final)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
