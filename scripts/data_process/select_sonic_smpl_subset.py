"""Select a traceable Sonic SMPL source subset before format conversion.

Purpose:
    Pick a subset from a large gear-sonic SMPL directory. The script only
    creates symlinks and manifest files; it does not copy the large source .pkl
    files.

Typical usage:
    conda run -n gear_sonic_train python scripts/data_process/select_sonic_smpl_subset.py \
        --mode conservative \
        --src_folder ~/robot_software/drl/wbc/GR00T-WholeBodyControl/data/smpl_filtered \
        --out_folder data/sonic_smpl_data/selected_10_12h \
        --target_hours 11 --min_hours 10 --max_hours 12 --overwrite

    conda run -n gear_sonic_train python scripts/data_process/analyze_sonic_smpl_dataset.py \
        --src_folder ~/robot_software/drl/wbc/GR00T-WholeBodyControl/data/smpl_filtered \
        --out_folder data/sonic_smpl_analysis/full_source --overwrite

    conda run -n gear_sonic_train python scripts/data_process/select_sonic_smpl_subset.py \
        --mode diverse_all_actions \
        --metadata_csv data/sonic_smpl_analysis/full_source/manifest.csv \
        --out_folder data/sonic_smpl_data/diverse_all_actions_12h \
        --target_hours 12 --min_hours 11.8 --max_hours 12.2 --overwrite

Outputs:
    motions/ symlinks, manifest.csv, selected_sonic_smpl.txt, summary.json,
    and selection_principles.md.

Defaults:
    --src_folder can also be supplied by SONIC_SMPL_SRC.
    --out_folder can also be supplied by SONIC_SMPL_OUT.
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .motion_semantics import (
        CATEGORY_DESCRIPTIONS,
        actor_id_from_path,
        categorize_motion,
        has_external_support_dependency,
        is_mirror_path,
        motion_family_from_path,
        normalized_family,
    )
    from .sonic_smpl import apply_coord_transform, load_sonic_motion
except ImportError:
    from motion_semantics import (
        CATEGORY_DESCRIPTIONS,
        actor_id_from_path,
        categorize_motion,
        has_external_support_dependency,
        is_mirror_path,
        motion_family_from_path,
        normalized_family,
    )
    from sonic_smpl import apply_coord_transform, load_sonic_motion


DEFAULT_SRC = Path(
    os.environ.get(
        "SONIC_SMPL_SRC",
        "/home/user/robot_software/drl/wbc/GR00T-WholeBodyControl/data/smpl_filtered",
    )
)
DEFAULT_METADATA = Path(
    os.environ.get("SONIC_SMPL_ANALYSIS_MANIFEST", "data/sonic_smpl_analysis/full_source/manifest.csv")
)
DEFAULT_OUT = Path(os.environ.get("SONIC_SMPL_OUT", "data/sonic_smpl_data/selected_10_12h"))

CONSERVATIVE_TARGET_RATIOS = {
    "locomotion": 0.50,
    "idle": 0.15,
    "upper_body": 0.20,
    "transition_mild": 0.10,
    "dynamic": 0.05,
}

DIVERSE_TARGET_RATIOS = {
    "locomotion_walk": 0.14,
    "locomotion_jog_run": 0.09,
    "jump": 0.08,
    "dance": 0.07,
    "idle_stance": 0.08,
    "upper_body_gesture": 0.10,
    "object_manipulation_carry": 0.10,
    "ground_low_posture": 0.07,
    "injury_impaired_gait": 0.06,
    "turn_transition": 0.05,
    "obstacle_contact_avoidance": 0.05,
    "daily_social_expression": 0.04,
    "exercise_sport": 0.03,
    "kick_throw_stoop": 0.02,
    "other": 0.02,
}

EXCLUDE_KEYWORDS = (
    "jump",
    "high_jump",
    "reach_jump",
    "dance",
    "dancing",
    "kneel",
    "kneeling",
    "sit",
    "sit_on_heels",
    "crawl",
    "lie",
    "on_ground",
    "horse_riding",
    "lasso",
    "screaming",
    "body_check",
)

UPPER_BODY_KEYWORDS = (
    "clap",
    "salute",
    "thinking",
    "confusion",
    "welcoming",
    "reaching_up",
    "reaching_far",
    "pocket_searching",
    "checking_time",
    "looking_around",
    "itching",
    "chefs_kiss",
    "omg",
    "don_t_know",
    "no_see",
    "no_hear",
    "fixing_something",
)

TRANSITION_MILD_KEYWORDS = (
    "body_stretch",
    "reaching_down",
    "brush_of_dust",
    "body_search",
    "exercise_1",
    "exercise_2",
    "freezing_cold",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Select a 10-12 hour Sonic SMPL subset using symlinks.")
    parser.add_argument(
        "--mode",
        choices=("conservative", "diverse_all_actions"),
        default="conservative",
        help=(
            "conservative keeps the original stable locomotion-oriented policy; "
            "diverse_all_actions uses a precomputed analysis manifest and keeps "
            "jump/dance/run style high-dynamic actions."
        ),
    )
    parser.add_argument("--src_folder", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--metadata_csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out_folder", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--exclude_manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Manifest CSV(s) whose source_path/source_file entries should be excluded. "
            "Can be passed multiple times for incremental supplement selection."
        ),
    )
    parser.add_argument("--target_hours", type=float, default=None)
    parser.add_argument("--min_hours", type=float, default=None)
    parser.add_argument("--max_hours", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min_duration_sec", type=float, default=None)
    parser.add_argument("--max_duration_sec", type=float, default=None)
    parser.add_argument("--max_root_speed", type=float, default=4.0)
    parser.add_argument("--max_vertical_span", type=float, default=0.6)
    parser.add_argument("--max_family_minutes", type=float, default=None)
    parser.add_argument("--max_actor_minutes", type=float, default=None)
    parser.add_argument("--top_family_examples", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    apply_mode_defaults(args)
    return args


def apply_mode_defaults(args):
    if args.mode == "diverse_all_actions":
        defaults = {
            "target_hours": 12.0,
            "min_hours": 11.8,
            "max_hours": 12.2,
            "seed": 20260707,
            "min_duration_sec": 2.5,
            "max_duration_sec": 20.0,
            "max_family_minutes": 3.0,
            "max_actor_minutes": 8.0,
        }
    else:
        defaults = {
            "target_hours": 11.0,
            "min_hours": 10.0,
            "max_hours": 12.0,
            "seed": 20260629,
            "min_duration_sec": 1.5,
            "max_duration_sec": 30.0,
            "max_family_minutes": 10.0,
            "max_actor_minutes": 60.0,
        }

    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def motion_name(path):
    return motion_family_from_path(path)


def normalize_source_path(value):
    if not value:
        return None
    return str(Path(value).expanduser().resolve(strict=False))


def load_excluded_sources(manifest_paths):
    excluded_paths = set()
    excluded_files = set()
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Exclude manifest not found: {manifest_path}")
        with manifest_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                source_path = normalize_source_path(row.get("source_path", ""))
                if source_path:
                    excluded_paths.add(source_path)
                source_file = row.get("source_file", "")
                if source_file:
                    excluded_files.add(Path(source_file).name)
    return {"paths": excluded_paths, "files": excluded_files}


def is_excluded_source(path, excluded_sources):
    return (
        normalize_source_path(path) in excluded_sources["paths"]
        or Path(path).name in excluded_sources["files"]
    )


def has_any(text, keywords):
    return any(keyword in text for keyword in keywords)


def categorize(name):
    lower = name.lower()
    if has_external_support_dependency(lower):
        return None, "external_support_dependency"
    if any(keyword in lower for keyword in EXCLUDE_KEYWORDS):
        return None, "excluded_action_keyword"
    if lower.startswith("walk") or "walk_" in lower:
        return "locomotion", "walk_family"
    if lower.startswith("idle_turn"):
        return "locomotion", "idle_turn_family"
    if lower.startswith("idle") or lower.startswith("looking_around"):
        return "idle", "idle_or_look_family"
    if lower.startswith("jog"):
        return "dynamic", "jog_family_limited"
    if has_any(lower, UPPER_BODY_KEYWORDS):
        return "upper_body", "standing_upper_body_keyword"
    if has_any(lower, TRANSITION_MILD_KEYWORDS):
        return "transition_mild", "transition_mild_keyword"
    return None, "not_in_selected_action_families"


def load_metadata(path):
    try:
        motion = load_sonic_motion(path)
    except Exception as exc:
        return None, f"{type(exc).__name__}:{exc}"

    _, trans_gmr = apply_coord_transform(
        motion.poses,
        motion.trans,
        "sonic_yup_to_gmr_zup",
    )
    if len(trans_gmr) > 1:
        import numpy as np

        root_speed = np.linalg.norm(np.diff(trans_gmr, axis=0), axis=1) * motion.fps
        max_root_speed = float(root_speed.max())
    else:
        max_root_speed = 0.0

    return {
        "frames": int(motion.poses.shape[0]),
        "fps": motion.fps,
        "duration_sec": float(motion.poses.shape[0] / motion.fps),
        "max_root_speed": max_root_speed,
        "vertical_span": float(trans_gmr[:, 2].max() - trans_gmr[:, 2].min()),
        "pose_key": motion.pose_key,
        "trans_key": motion.trans_key,
        "fps_key": motion.fps_key,
        "normalization_adjustments": ",".join(motion.adjustments),
    }, None


def write_principles(path, args, summary):
    exclude_lines = "\n".join(f"- {manifest}" for manifest in args.exclude_manifest) or "- none"
    text = f"""# Sonic SMPL Selection Principles

Source:

```text
{args.src_folder}
```

Output:

```text
{args.out_folder}
```

Exclude manifests:

```text
{exclude_lines}
```

Target duration: {args.target_hours:.1f} hours, accepted range {args.min_hours:.1f}-{args.max_hours:.1f} hours.

## Category Ratios

```text
locomotion       50%
idle             15%
upper_body       20%
transition_mild  10%
dynamic           5%
```

## Include First

- walk, walk_ff, walk_forward, walk_backward, walk_left, walk_right
- idle, idle_loop, idle_turn
- looking_around, checking_time, salute, clap, thinking, confusion, welcoming
- reaching_up, reaching_far, pocket_searching, body_stretch

## Limited Include

- jog and mild exercise motions, capped by the dynamic ratio
- reaching_down, brush_of_dust, body_search, itching-like upper-body motions

## Exclude By Default

- non-foot external-support-dependent motions such as wall leaning, lean_on, lean_against, resting_on and supported_by
- jump, high_jump, reach_jump
- dance, dancing
- kneel, kneeling, sit, sit_on_heels
- crawl, lie, on_ground
- horse_riding, lasso, screaming, body_check

## Quality Gates

- `pose_aa` must be `(T, 72)`.
- `transl` must be `(T, 3)`.
- `pose_aa` and `transl` must have the same frame count.
- `fps` must be valid.
- No NaN or Inf values.
- Duration must be between {args.min_duration_sec:.1f}s and {args.max_duration_sec:.1f}s.
- After Sonic Y-up to GMR Z-up check, max root speed must be <= {args.max_root_speed:.1f} m/s.
- Vertical span must be <= {args.max_vertical_span:.2f} m.
- Per motion family cap: {args.max_family_minutes:.1f} min.
- Per actor cap: {args.max_actor_minutes:.1f} min.

## Result Summary

```json
{json.dumps(summary, indent=2, sort_keys=True)}
```
"""
    path.write_text(text)


def run_conservative(args):
    if args.out_folder.exists() and any(args.out_folder.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output folder is not empty: {args.out_folder}. Use --overwrite.")

    motions_dir = args.out_folder / "motions"
    args.out_folder.mkdir(parents=True, exist_ok=True)
    motions_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    candidates = defaultdict(list)
    excluded_by_name = defaultdict(int)
    excluded_sources = load_excluded_sources(args.exclude_manifest)

    for path in sorted(args.src_folder.glob("*.pkl")):
        if is_excluded_source(path, excluded_sources):
            excluded_by_name["exclude_manifest_duplicate"] += 1
            continue
        name = motion_name(path)
        category, reason = categorize(name)
        if category is None:
            excluded_by_name[reason] += 1
            continue
        candidates[category].append((path, name, reason))

    for items in candidates.values():
        rng.shuffle(items)

    target_total_sec = args.target_hours * 3600.0
    category_targets = {key: target_total_sec * ratio for key, ratio in CONSERVATIVE_TARGET_RATIOS.items()}
    max_family_sec = args.max_family_minutes * 60.0
    max_actor_sec = args.max_actor_minutes * 60.0

    selected = []
    category_sec = defaultdict(float)
    family_sec = defaultdict(float)
    actor_sec = defaultdict(float)
    rejection_counts = defaultdict(int)

    def try_select(category, strict_category_target=True):
        changed = False
        for path, name, reason in candidates.get(category, []):
            if strict_category_target and category_sec[category] >= category_targets[category]:
                break
            if sum(category_sec.values()) >= target_total_sec:
                break
            actor = actor_id_from_path(path)
            metadata, reject_reason = load_metadata(path)
            if reject_reason:
                rejection_counts[reject_reason] += 1
                continue
            duration = metadata["duration_sec"]
            if duration < args.min_duration_sec:
                rejection_counts["too_short"] += 1
                continue
            if duration > args.max_duration_sec:
                rejection_counts["too_long"] += 1
                continue
            if metadata["max_root_speed"] > args.max_root_speed:
                rejection_counts["root_speed_high"] += 1
                continue
            if metadata["vertical_span"] > args.max_vertical_span:
                rejection_counts["vertical_span_high"] += 1
                continue
            if family_sec[name] + duration > max_family_sec:
                rejection_counts["family_cap"] += 1
                continue
            if actor_sec[actor] + duration > max_actor_sec:
                rejection_counts["actor_cap"] += 1
                continue

            rel_link = Path("motions") / path.name
            dst = args.out_folder / rel_link
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(path, dst)

            row = {
                "relative_link": str(rel_link),
                "source_path": str(path),
                "source_file": path.name,
                "motion_family": name,
                "actor": actor,
                "is_mirror": str(is_mirror_path(path)),
                "category": category,
                "fps": f"{metadata['fps']:.6g}",
                "num_frames": str(metadata["frames"]),
                "duration_sec": f"{duration:.6f}",
                "keep_reason": reason,
                "max_root_speed": f"{metadata['max_root_speed']:.6f}",
                "vertical_span": f"{metadata['vertical_span']:.6f}",
            }
            selected.append(row)
            category_sec[category] += duration
            family_sec[name] += duration
            actor_sec[actor] += duration
            changed = True
        return changed

    for category in CONSERVATIVE_TARGET_RATIOS:
        try_select(category, strict_category_target=True)

    if sum(category_sec.values()) < args.min_hours * 3600.0:
        for category in CONSERVATIVE_TARGET_RATIOS:
            try_select(category, strict_category_target=False)
            if sum(category_sec.values()) >= args.min_hours * 3600.0:
                break

    total_sec = sum(category_sec.values())
    summary = {
        "selected_files": len(selected),
        "total_hours": round(total_sec / 3600.0, 3),
        "category_hours": {key: round(value / 3600.0, 3) for key, value in sorted(category_sec.items())},
        "category_files": {
            key: sum(1 for row in selected if row["category"] == key) for key in sorted(CONSERVATIVE_TARGET_RATIOS)
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "excluded_by_name_counts": dict(sorted(excluded_by_name.items())),
        "seed": args.seed,
    }

    fieldnames = [
        "relative_link",
        "source_path",
        "source_file",
        "motion_family",
        "actor",
        "is_mirror",
        "category",
        "fps",
        "num_frames",
        "duration_sec",
        "keep_reason",
        "max_root_speed",
        "vertical_span",
    ]
    with (args.out_folder / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    with (args.out_folder / "selected_sonic_smpl.txt").open("w") as f:
        for row in selected:
            f.write(row["source_path"] + "\n")

    (args.out_folder / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_principles(args.out_folder / "selection_principles.md", args, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if total_sec < args.min_hours * 3600.0 or total_sec > args.max_hours * 3600.0:
        raise RuntimeError(
            f"Selected {total_sec / 3600.0:.2f}h, outside requested range "
            f"{args.min_hours:.1f}-{args.max_hours:.1f}h."
        )


def float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_diverse_candidates_from_manifest(args):
    if not args.metadata_csv.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: {args.metadata_csv}. "
            "Run scripts/data_process/analyze_sonic_smpl_dataset.py first or pass --metadata_csv."
        )

    rows = []
    rejection_counts = Counter()
    excluded_sources = load_excluded_sources(args.exclude_manifest)
    with args.metadata_csv.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                rejection_counts[f"metadata_{row.get('status', 'unknown')}"] += 1
                continue

            source_path = Path(row["source_path"])
            if is_excluded_source(source_path, excluded_sources):
                rejection_counts["exclude_manifest_duplicate"] += 1
                continue
            if not source_path.exists():
                rejection_counts["missing_source_file"] += 1
                continue

            duration = float_or_none(row.get("duration_sec"))
            if duration is None:
                rejection_counts["missing_duration"] += 1
                continue
            if duration < args.min_duration_sec:
                rejection_counts["too_short"] += 1
                continue
            if duration > args.max_duration_sec:
                rejection_counts["too_long"] += 1
                continue

            family = row.get("motion_family") or motion_name(source_path)
            if has_external_support_dependency(f"{source_path.name} {family}"):
                rejection_counts["external_support_dependency"] += 1
                continue

            category = row.get("category") or categorize_motion(family)
            if category not in DIVERSE_TARGET_RATIOS:
                category = categorize_motion(family)

            rows.append(
                {
                    "source_path": str(source_path),
                    "source_file": source_path.name,
                    "motion_family": family,
                    "normalized_family": row.get("normalized_family") or normalized_family(family),
                    "actor": row.get("actor") or actor_id_from_path(source_path),
                    "is_mirror": row.get("is_mirror") or str(is_mirror_path(source_path)),
                    "category": category,
                    "fps": row.get("fps", ""),
                    "num_frames": row.get("num_frames", ""),
                    "duration_sec": duration,
                }
            )

    return rows, rejection_counts


def duration_stats(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {}

    def quantile(q):
        pos = (len(values) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(values) - 1)
        frac = pos - lo
        return values[lo] * (1 - frac) + values[hi] * frac

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "total_hours": round(sum(values) / 3600.0, 6),
        "min_sec": round(values[0], 6),
        "p10_sec": round(quantile(0.10), 6),
        "p25_sec": round(quantile(0.25), 6),
        "median_sec": round(quantile(0.50), 6),
        "p75_sec": round(quantile(0.75), 6),
        "p90_sec": round(quantile(0.90), 6),
        "max_sec": round(values[-1], 6),
        "mean_sec": round(mean, 6),
        "std_sec": round(math.sqrt(variance), 6),
    }


def make_round_robin_order(rows, rng):
    family_rows = defaultdict(list)
    for row in rows:
        family_rows[row["normalized_family"]].append(row)
    families = list(family_rows)
    rng.shuffle(families)
    for family in families:
        rng.shuffle(family_rows[family])

    ordered = []
    max_depth = max((len(items) for items in family_rows.values()), default=0)
    for depth in range(max_depth):
        depth_families = families[:]
        rng.shuffle(depth_families)
        for family in depth_families:
            if depth < len(family_rows[family]):
                ordered.append(family_rows[family][depth])
    return ordered


def prepare_output_dir(path, overwrite):
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output folder is not empty: {path}. Use --overwrite.")
        shutil.rmtree(path)
    (path / "motions").mkdir(parents=True, exist_ok=True)


def select_diverse_rows(candidates, args):
    rng = random.Random(args.seed)
    target_sec = args.target_hours * 3600.0
    min_sec = args.min_hours * 3600.0
    max_sec = args.max_hours * 3600.0
    max_family_sec = args.max_family_minutes * 60.0
    max_actor_sec = args.max_actor_minutes * 60.0

    rows_by_category = defaultdict(list)
    for row in candidates:
        rows_by_category[row["category"]].append(row)

    ordered_by_category = {
        category: make_round_robin_order(rows_by_category.get(category, []), rng)
        for category in DIVERSE_TARGET_RATIOS
    }
    fill_order = make_round_robin_order(candidates, rng)

    selected = []
    selected_paths = set()
    category_sec = Counter()
    family_sec = Counter()
    actor_sec = Counter()
    rejection_counts = Counter()

    def try_add(row, category_target=None, allow_category_fill=False):
        duration = row["duration_sec"]
        category = row["category"]
        family = row["normalized_family"]
        actor = row["actor"]
        path = row["source_path"]

        if path in selected_paths:
            rejection_counts["already_selected"] += 1
            return False
        if sum(category_sec.values()) + duration > max_sec:
            rejection_counts["would_exceed_max_hours"] += 1
            return False
        if family_sec[family] + duration > max_family_sec:
            rejection_counts["family_cap"] += 1
            return False
        if actor_sec[actor] + duration > max_actor_sec:
            rejection_counts["actor_cap"] += 1
            return False
        if category_target is not None and not allow_category_fill and category_sec[category] >= category_target:
            rejection_counts["category_target_met"] += 1
            return False

        selected.append(row)
        selected_paths.add(path)
        category_sec[category] += duration
        family_sec[family] += duration
        actor_sec[actor] += duration
        return True

    category_targets = {
        category: target_sec * ratio for category, ratio in DIVERSE_TARGET_RATIOS.items()
    }

    for category in DIVERSE_TARGET_RATIOS:
        category_target = category_targets[category]
        for row in ordered_by_category.get(category, []):
            if category_sec[category] >= category_target:
                break
            try_add(row, category_target=category_target)

    if sum(category_sec.values()) < min_sec:
        for row in fill_order:
            if sum(category_sec.values()) >= target_sec:
                break
            try_add(row, allow_category_fill=True)

    return selected, {
        "category_sec": dict(category_sec),
        "family_sec": dict(family_sec),
        "actor_sec": dict(actor_sec),
        "category_targets": category_targets,
        "rejection_counts": rejection_counts,
    }


def write_diverse_manifest(out_folder, selected):
    fieldnames = [
        "relative_link",
        "source_path",
        "source_file",
        "motion_family",
        "normalized_family",
        "actor",
        "is_mirror",
        "category",
        "fps",
        "num_frames",
        "duration_sec",
    ]
    with (out_folder / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            output = dict(row)
            output["relative_link"] = str(Path("motions") / row["source_file"])
            output["duration_sec"] = f"{row['duration_sec']:.8f}"
            writer.writerow(output)


def write_diverse_principles(out_folder, args, summary):
    ratio_lines = "\n".join(f"{key:28s} {value * 100:5.1f}%" for key, value in DIVERSE_TARGET_RATIOS.items())
    category_lines = "\n".join(f"- `{key}`: {value}" for key, value in CATEGORY_DESCRIPTIONS.items())
    exclude_lines = "\n".join(f"- {manifest}" for manifest in args.exclude_manifest) or "- none"
    text = f"""# Diverse Sonic SMPL Selection Principles

Source:

```text
{args.src_folder}
```

Metadata:

```text
{args.metadata_csv}
```

Output:

```text
{args.out_folder}
```

Exclude manifests:

```text
{exclude_lines}
```

Target duration: {args.target_hours:.2f} hours, accepted range {args.min_hours:.2f}-{args.max_hours:.2f} hours.

Duration gate: {args.min_duration_sec:.2f}s <= clip <= {args.max_duration_sec:.2f}s.

## Category Ratios

```text
{ratio_lines}
```

## Category Definitions

{category_lines}

## Diversity Controls

- Round-robin sampling across normalized motion families inside each category.
- Per normalized family cap: {args.max_family_minutes:.2f} minutes.
- Per actor cap: {args.max_actor_minutes:.2f} minutes.
- High-dynamic actions are included explicitly: jump, dance, jog/run, obstacle/contact, exercise/sport.
- Files are symlinked under `motions/`; original Sonic SMPL files are not copied.

## Exclude By Default

- Motions with non-foot body support from external geometry are excluded before selection.
- This includes explicit wall/object support families such as `wall_leaning`, `lean_on`, `lean_against`, `resting_on`, `supported_by`, and support-object combinations involving wall/table/chair/door/counter/bar/rail/fence/pole/ladder.
- This rule is independent from high-dynamic filtering: jump, run, dance and obstacle clips remain eligible when they are self-contained and do not rely on an external support object.

## Result Summary

```json
{json.dumps(summary, indent=2, sort_keys=True)}
```
"""
    (out_folder / "selection_principles.md").write_text(text)


def run_diverse_all_actions(args):
    if not args.src_folder.exists():
        raise FileNotFoundError(args.src_folder)
    if not math.isclose(sum(DIVERSE_TARGET_RATIOS.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"DIVERSE_TARGET_RATIOS must sum to 1.0, got {sum(DIVERSE_TARGET_RATIOS.values())}")
    if args.min_hours > args.target_hours or args.target_hours > args.max_hours:
        raise ValueError("--min_hours <= --target_hours <= --max_hours is required")

    candidates, metadata_rejections = load_diverse_candidates_from_manifest(args)
    prepare_output_dir(args.out_folder, args.overwrite)

    selected, selection_state = select_diverse_rows(candidates, args)
    total_sec = sum(row["duration_sec"] for row in selected)
    if total_sec < args.min_hours * 3600.0 or total_sec > args.max_hours * 3600.0:
        raise RuntimeError(
            f"Selected {total_sec / 3600.0:.3f}h, outside requested range "
            f"{args.min_hours:.2f}-{args.max_hours:.2f}h."
        )

    for row in selected:
        dst = args.out_folder / "motions" / row["source_file"]
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(row["source_path"], dst)

    write_diverse_manifest(args.out_folder, selected)
    with (args.out_folder / "selected_sonic_smpl.txt").open("w") as f:
        for row in selected:
            f.write(row["source_path"] + "\n")

    category_rows = []
    for category in DIVERSE_TARGET_RATIOS:
        rows = [row for row in selected if row["category"] == category]
        durations = [row["duration_sec"] for row in rows]
        category_rows.append(
            {
                "category": category,
                "target_ratio": DIVERSE_TARGET_RATIOS[category],
                "target_hours": round(selection_state["category_targets"][category] / 3600.0, 6),
                "selected_files": len(rows),
                "selected_hours": round(sum(durations) / 3600.0, 6),
                "unique_families": len({row["normalized_family"] for row in rows}),
                "unique_actors": len({row["actor"] for row in rows}),
                "duration": duration_stats(durations),
            }
        )

    family_counter = Counter()
    family_hours = Counter()
    for row in selected:
        family_counter[row["normalized_family"]] += 1
        family_hours[row["normalized_family"]] += row["duration_sec"] / 3600.0

    category_summary_fields = [
        "category",
        "target_ratio",
        "target_hours",
        "selected_files",
        "selected_hours",
        "unique_families",
        "unique_actors",
        "median_sec",
        "p90_sec",
        "max_sec",
    ]
    with (args.out_folder / "category_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=category_summary_fields)
        writer.writeheader()
        for row in category_rows:
            stats = row["duration"]
            writer.writerow(
                {
                    "category": row["category"],
                    "target_ratio": f"{row['target_ratio']:.6f}",
                    "target_hours": f"{row['target_hours']:.6f}",
                    "selected_files": row["selected_files"],
                    "selected_hours": f"{row['selected_hours']:.6f}",
                    "unique_families": row["unique_families"],
                    "unique_actors": row["unique_actors"],
                    "median_sec": stats.get("median_sec", ""),
                    "p90_sec": stats.get("p90_sec", ""),
                    "max_sec": stats.get("max_sec", ""),
                }
            )

    top_families = [
        {
            "normalized_family": family,
            "files": family_counter[family],
            "hours": round(family_hours[family], 6),
        }
        for family, _ in family_counter.most_common(args.top_family_examples)
    ]

    summary = {
        "mode": args.mode,
        "source_folder": str(args.src_folder),
        "metadata_csv": str(args.metadata_csv),
        "output_folder": str(args.out_folder),
        "seed": args.seed,
        "selected_files": len(selected),
        "total_hours": round(total_sec / 3600.0, 6),
        "duration_gate_sec": [args.min_duration_sec, args.max_duration_sec],
        "duration": duration_stats([row["duration_sec"] for row in selected]),
        "unique_motion_families": len({row["motion_family"] for row in selected}),
        "unique_normalized_families": len({row["normalized_family"] for row in selected}),
        "unique_actors": len({row["actor"] for row in selected}),
        "mirror_files": sum(1 for row in selected if row["is_mirror"] == "True"),
        "category_summary": category_rows,
        "target_ratios": DIVERSE_TARGET_RATIOS,
        "category_descriptions": CATEGORY_DESCRIPTIONS,
        "top_normalized_families": top_families,
        "candidate_files_after_duration_gate": len(candidates),
        "metadata_rejection_counts": dict(sorted(metadata_rejections.items())),
        "selection_rejection_counts": dict(sorted(selection_state["rejection_counts"].items())),
        "max_family_minutes": args.max_family_minutes,
        "max_actor_minutes": args.max_actor_minutes,
    }

    (args.out_folder / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_diverse_principles(args.out_folder, args, summary)

    print(json.dumps({"selected_files": len(selected), "total_hours": round(total_sec / 3600.0, 6)}, indent=2))
    print(f"Wrote selection to {args.out_folder}")


def main():
    args = parse_args()
    if args.mode == "diverse_all_actions":
        run_diverse_all_actions(args)
    else:
        run_conservative(args)


if __name__ == "__main__":
    main()
