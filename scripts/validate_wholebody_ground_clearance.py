"""Validate exported WholeBody Omni GMR PKL files against the final visual mesh."""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle

import mink
import mujoco as mj
import numpy as np

from general_motion_retargeting.wholebody_omni_gmr import GroundNonPenetrationLimit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_motion_path", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--xml", default=None)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    config_path = pathlib.Path(args.config) if args.config else root / "general_motion_retargeting/ik_configs/smplx_to_ne01_wholebody_omni_gmr.json"
    xml_path = pathlib.Path(args.xml) if args.xml else root / "assets/ne01/ne01_wholebody_omni_gmr.xml"
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    with pathlib.Path(args.robot_motion_path).open("rb") as file:
        motion = pickle.load(file)

    model = mj.MjModel.from_xml_path(str(xml_path))
    configuration = mink.Configuration(model)
    nonpenetration = config["ground_nonpenetration"]
    limit = GroundNonPenetrationLimit(
        model,
        config["ground_interaction_graph"]["regions"],
        nonpenetration["always_active_sites"],
        nonpenetration.get("collision_geoms", {}),
        nonpenetration.get("dynamic_mesh_guards", {}),
        float(config["whole_body_ground"].get("floor_z", 0.0)),
        float(config["whole_body_ground"].get("clearance", 0.002)),
    )

    records = []
    for frame, (root_pos, root_rot, dof_pos) in enumerate(zip(motion["root_pos"], motion["root_rot"], motion["dof_pos"])):
        # Exported rotations are xyzw; MuJoCo free-joint qpos uses wxyz.
        qpos = np.r_[root_pos, np.asarray(root_rot)[[3, 0, 1, 2]], dof_pos]
        configuration.update(qpos)
        for name, item in limit.measure_current_slacks(configuration).items():
            records.append((item["slack"], frame, name, item["height"], item["required_margin"]))

    print(f"Validated {len(motion['root_pos'])} exported frames from {args.robot_motion_path}")
    print("Worst final-qpos clearances:")
    for slack, frame, name, height, margin in sorted(records)[: max(1, args.top)]:
        print(
            f"  frame={frame:5d} slack={slack * 1000:9.4f} mm "
            f"height={height * 1000:9.4f} mm margin={margin * 1000:6.2f} mm {name}"
        )
    minimum = min(records)
    if minimum[0] < -1e-5:
        raise SystemExit(f"FAILED: final exported qpos penetrates at frame {minimum[1]} ({minimum[2]})")
    print("PASS: collision surfaces, hard points, and complete guarded visual meshes have nonnegative slack.")


if __name__ == "__main__":
    main()
