"""Single-motion entry point for the graph-regularized soft-contact retargeter."""

from __future__ import annotations

import argparse
import pathlib
import pickle

import numpy as np
import torch

from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.laplacian_soft_retarget import (
    LaplacianSoftContactRetargeting,
)
from general_motion_retargeting.utils.smpl import (
    get_smplx_data_offline_fast,
    load_smplx_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_file", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--robot", default="ne01")
    parser.add_argument("--graph_config", default=None)
    parser.add_argument("--tgt_fps", type=float, default=50.0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--solver", default="proxqp")
    parser.add_argument("--velocity_limit", type=float, default=3 * np.pi)
    args = parser.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    body_model_folder = here.parent / "assets" / "body_models"
    graph_config = args.graph_config
    if graph_config is None and args.robot == "ne01":
        graph_config = str(
            here.parent
            / "general_motion_retargeting"
            / "ik_configs"
            / "smplx_to_ne01_laplacian_soft.json"
        )

    smplx_data, body_model, smplx_output, actual_height = load_smplx_file(
        args.smplx_file, body_model_folder
    )
    frame_data, aligned_fps = get_smplx_data_offline_fast(
        smplx_data,
        body_model,
        smplx_output,
        tgt_fps=args.tgt_fps,
    )

    retargeter = LaplacianSoftContactRetargeting(
        src_human="smplx",
        tgt_robot=args.robot,
        actual_human_height=actual_height,
        solver=args.solver,
        use_velocity_limit=True,
        velocity_limit=args.velocity_limit,
        motion_fps=args.tgt_fps,
        graph_config_path=graph_config,
        verbose=True,
    )
    retargeter.set_motion_floor(frame_data)

    indices = list(range(1, len(frame_data))) if len(frame_data) > 1 else [0]
    if args.max_frames is not None:
        indices = indices[: args.max_frames]

    qpos_list = []
    for index in indices:
        qpos_list.append(retargeter.retarget(frame_data[index]).copy())

    qpos = np.asarray(qpos_list, dtype=float)
    if len(qpos) > 0:
        qpos[:, :2] -= qpos[0, :2]

    root_pos = qpos[:, :3]
    root_rot_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]
    dof_pos = qpos[:, 7:]

    local_body_pos = None
    body_names = None
    try:
        kinematics = KinematicsModel(retargeter.xml_file, device="cpu")
        with torch.no_grad():
            zero_root = torch.zeros((len(qpos), 3), dtype=torch.float32)
            identity_root = torch.zeros((len(qpos), 4), dtype=torch.float32)
            identity_root[:, -1] = 1.0
            local_body_pos_t, _ = kinematics.forward_kinematics(
                zero_root,
                identity_root,
                torch.as_tensor(dof_pos, dtype=torch.float32),
            )
        local_body_pos = local_body_pos_t.detach().cpu().numpy()
        body_names = kinematics.body_names
    except Exception as exc:
        print(f"Warning: local FK export skipped: {type(exc).__name__}: {exc}")

    output = pathlib.Path(args.save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(
            {
                "fps": float(aligned_fps),
                "root_pos": root_pos,
                "root_rot": root_rot_xyzw,
                "dof_pos": dof_pos,
                "local_body_pos": local_body_pos,
                "link_body_list": body_names,
            },
            file,
        )
    print(f"Saved graph/soft-contact motion to {output}")


if __name__ == "__main__":
    main()
