"""Fast SMPL-X motion pre-filter for non-flat or unsuitable interactions.

The filter is intentionally conservative: it reports suspicious files instead
of pretending to understand arbitrary objects or terrain. It can optionally
copy only accepted motions while preserving their relative paths.
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np

from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast


def _series(frame_list, name):
    return np.asarray([frame[name][0] for frame in frame_list if name in frame], dtype=float)


def inspect_motion(
    path,
    body_model_folder,
    sample_fps=30.0,
    reject_in_place=False,
    in_place_root_span=0.12,
    in_place_foot_span=0.35,
):
    data, body_model, output, _ = load_smplx_file(path, body_model_folder)
    frames, _ = get_smplx_data_offline_fast(data, body_model, output, tgt_fps=sample_fps)
    frames = frames[: min(len(frames), 1200)]
    if len(frames) < 4:
        return "too_short", {"reason": "too_short"}

    foot_z = []
    foot_xy = []
    for side in ("left", "right"):
        names = [f"{side}_heel", f"{side}_big_toe", f"{side}_small_toe", f"{side}_foot"]
        points = [
            _series(frames, name)
            for name in names
            if all(name in frame for frame in frames)
        ]
        if not points:
            continue
        points = np.stack(points, axis=1)
        foot_z.append(np.min(points[..., 2], axis=1))
        foot_xy.append(np.mean(points[..., :2], axis=1))
    if not foot_z:
        return "unknown", {"reason": "missing_foot_joints"}

    foot_z = np.stack(foot_z, axis=1)
    floor = float(np.percentile(foot_z, 10))
    low_envelope = np.min(foot_z, axis=1)
    contact = low_envelope <= floor + 0.045
    if foot_xy:
        xy = np.mean(np.stack(foot_xy, axis=1), axis=1)
        speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) * sample_fps
        contact[1:] &= speed < 1.0

    contact_heights = foot_z[contact]
    terrain_span = 0.0 if len(contact_heights) == 0 else float(
        np.percentile(contact_heights, 95) - np.percentile(contact_heights, 5)
    )

    pelvis = _series(frames, "pelvis")
    spine = _series(frames, "spine3")
    torso_axis = spine - pelvis
    torso_norm = np.linalg.norm(torso_axis, axis=1, keepdims=True).clip(min=1e-6)
    # Joint positions are used here deliberately. SMPL-X joint local axes are
    # model-dependent and are not a reliable semantic "up" direction.
    torso_up_z = (torso_axis / torso_norm)[:, 2]
    foot_floor = np.min(foot_z, axis=1)
    pelvis_gap = pelvis[:, 2] - foot_floor
    inverted_fraction = float(np.mean(torso_up_z < -0.2)) if len(torso_up_z) else 0.0
    prone_fraction = float(np.mean((torso_up_z < 0.45) & (pelvis_gap < 0.45))) if len(torso_up_z) else 0.0

    root_xy_span = float(np.linalg.norm(np.ptp(pelvis[:, :2], axis=0)))
    foot_xy_spans = []
    for side_xy in foot_xy:
        foot_xy_spans.append(float(np.linalg.norm(np.ptp(side_xy, axis=0))))
    median_foot_xy_span = float(np.median(foot_xy_spans)) if foot_xy_spans else 0.0

    flags = []
    if terrain_span > 0.10:
        flags.append("nonflat_or_stairs")
    if inverted_fraction > 0.10:
        flags.append("inverted_or_handstand")
    if prone_fraction > 0.20:
        flags.append("prone_or_floor_interaction")
    in_place_locomotion = (
        root_xy_span < in_place_root_span
        and median_foot_xy_span > in_place_foot_span
        and float(np.mean(contact)) > 0.20
    )
    if reject_in_place and in_place_locomotion:
        flags.append("in_place_locomotion")
    status = "reject" if flags else "accept"
    return status, {
        "reason": ";".join(flags) if flags else "ok",
        "frames": len(frames),
        "terrain_span_m": terrain_span,
        "inverted_fraction": inverted_fraction,
        "prone_fraction": prone_fraction,
        "contact_fraction": float(np.mean(contact)),
        "root_xy_span_m": root_xy_span,
        "median_foot_xy_span_m": median_foot_xy_span,
        "in_place_locomotion": in_place_locomotion,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--accepted_folder", default=None)
    parser.add_argument("--body_model_folder", default="assets/body_models")
    parser.add_argument("--sample_fps", type=float, default=30.0)
    parser.add_argument(
        "--reject_in_place",
        action="store_true",
        help="Reject locomotion with near-stationary pelvis and large foot excursions.",
    )
    parser.add_argument("--in_place_root_span", type=float, default=0.12)
    parser.add_argument("--in_place_foot_span", type=float, default=0.35)
    args = parser.parse_args()

    rows = []
    src_root = Path(args.src_folder)
    for path in sorted(src_root.rglob("*.npz")):
        try:
            status, metrics = inspect_motion(
                path,
                args.body_model_folder,
                args.sample_fps,
                args.reject_in_place,
                args.in_place_root_span,
                args.in_place_foot_span,
            )
        except Exception as exc:
            status, metrics = "error", {"reason": f"{type(exc).__name__}: {exc}"}
        row = {"path": str(path.relative_to(src_root)), "status": status, **metrics}
        rows.append(row)
        print(row)
        if status == "accept" and args.accepted_folder:
            target = Path(args.accepted_folder) / path.relative_to(src_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with open(args.report, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved report: {args.report}; accepted={sum(r['status'] == 'accept' for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
