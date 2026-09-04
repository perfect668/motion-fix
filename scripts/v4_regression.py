"""Run a short WholeBody V4 regression and emit geometry/QP metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import subprocess
import sys

import mujoco as mj
import numpy as np


def _metrics(path: Path) -> dict:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    model = mj.MjModel.from_xml_path(str(payload["robot_xml"]))
    data = mj.MjData(model)
    soles = []
    for pose in np.asarray(payload["qpos"], dtype=float):
        data.qpos[:] = pose
        mj.mj_forward(model, data)
        for geom_id in range(model.ngeom):
            name = model.geom(geom_id).name or ""
            if "foot_" in name and "collision" in name:
                soles.append(float(data.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 0]))
    diagnostics = payload.get("terrain_diagnostics", [])
    failures = sum(bool(item.get("qp_failure") or item.get("qp_failures")) for item in diagnostics)
    q = np.asarray(payload["qpos"], dtype=float)
    root_xy = q[:, :2]
    root_z = q[:, 2]
    root_quat = q[:, 3:7]
    yaw = np.unwrap(np.arctan2(
        2.0 * (root_quat[:, 0] * root_quat[:, 3] + root_quat[:, 1] * root_quat[:, 2]),
        1.0 - 2.0 * (root_quat[:, 2] ** 2 + root_quat[:, 3] ** 2),
    ))
    joint_names = {
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    }
    knee_ranges = {}
    for name in ("KNEE_PITCH_L_JOINT", "KNEE_PITCH_R_JOINT"):
        if name in joint_names:
            address = int(model.jnt_qposadr[model.joint(name).id])
            knee_ranges[name] = [float(q[:, address].min()), float(q[:, address].max())]
    penetration = np.asarray([max(0.0, float(item.get("maximum_penetration", 0.0))) for item in diagnostics])
    return {
        "motion": str(path),
        "frames": int(len(payload["qpos"])),
        "fps": float(payload["fps"]),
        "source_format": payload.get("source_format"),
        "scene_scale": float(payload.get("scene_transform", {}).get("scale", 1.0)),
        "floor_z": float(payload.get("terrain_primitives", {}).get("floor_z", 0.0)),
        "qp_failures": int(failures),
        "sole_min_m": float(min(soles, default=np.inf)),
        "sole_max_m": float(max(soles, default=-np.inf)),
        "penetration_p99_m": float(np.quantile(penetration, 0.99)) if len(penetration) else 0.0,
        "pelvis_xy_min": root_xy.min(axis=0).tolist(),
        "pelvis_xy_max": root_xy.max(axis=0).tolist(),
        "pelvis_z_range": [float(root_z.min()), float(root_z.max())],
        "pelvis_yaw_range_rad": [float(yaw.min()), float(yaw.max())],
        "knee_angle_ranges_rad": knee_ranges,
        "active_collision_mean": float(np.mean([item.get("active_collision_points", 0) for item in diagnostics])) if diagnostics else 0.0,
        "active_collision_max": int(max([item.get("active_collision_points", 0) for item in diagnostics], default=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--terrain", type=Path)
    parser.add_argument("--save_path", required=True, type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    command = [
        sys.executable, str(Path(__file__).with_name("retarget_motion.py")),
        "--motion", str(args.motion), "--robot", "ne01", "--version", "v4",
        "--tgt_fps", "50", "--max_frames", "50", "--save_path", str(args.save_path),
    ]
    if args.terrain is not None:
        command.extend(["--terrain", str(args.terrain)])
    subprocess.run(command, check=True)
    report = _metrics(args.save_path.with_suffix(".pkl"))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
