"""Retarget one SMPL-X motion to one robot for debugging or quick export.

Purpose:
    This is the single-file companion to smplx_to_robot_dataset.py. Use it to
    inspect one SMPL-X file, debug coordinate/IK issues, optionally open the
    RobotMotionViewer, and optionally save one robot-motion .pkl.

Typical usage:
    conda run --no-capture-output -n gmr python scripts/smplx_to_robot.py \
        --smplx_file data/smplx_data/selected_10_12h/example.npz \
        --robot unitree_g1_24dof \
        --save_path /tmp/example_g1_24dof.pkl \
        --max_frames 300

Headless quick export:
    conda run --no-capture-output -n gmr python scripts/smplx_to_robot.py \
        --smplx_file data/smplx_data/selected_10_12h/example.npz \
        --robot ne01 \
        --save_path /tmp/example_ne01.pkl \
        --headless

Notes:
    Dataset production should use smplx_to_robot_dataset.py because it saves
    local_body_pos and link_body_list for downstream filtering. This script is
    mainly for visual inspection and targeted debugging.
"""

import argparse
import pathlib
import os
import time

import numpy as np
import torch

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from rich import print


def adjust_qpos_to_ground(qpos_list, xml_file, height_adjust=True, root_origin_offset=True):
    qpos_list = qpos_list.copy()
    root_pos = qpos_list[:, :3].copy()
    original_root_pos = root_pos.copy()
    root_rot_xyzw = qpos_list[:, 3:7][:, [1, 2, 3, 0]].copy()
    dof_pos = qpos_list[:, 7:].copy()

    if height_adjust:
        kinematics_model = KinematicsModel(xml_file, device="cpu")
        with torch.no_grad():
            body_pos, _ = kinematics_model.forward_kinematics(
                torch.as_tensor(root_pos, dtype=torch.float32),
                torch.as_tensor(root_rot_xyzw, dtype=torch.float32),
                torch.as_tensor(dof_pos, dtype=torch.float32),
            )
            lowest_height = torch.min(body_pos[..., 2]).item()
        root_pos[:, 2] -= lowest_height

    if root_origin_offset:
        root_pos[:, :2] -= root_pos[0, :2]

    qpos_list[:, :3] = root_pos
    return qpos_list, root_pos - original_root_pos


def offset_human_motion(human_motion_list, root_deltas):
    adjusted_motion = []
    for human_data, delta in zip(human_motion_list, root_deltas):
        frame_data = {}
        for body_name, (pos, quat) in human_data.items():
            frame_data[body_name] = [np.asarray(pos).copy() + delta, np.asarray(quat).copy()]
        adjusted_motion.append(frame_data)
    return adjusted_motion

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--robot",
        choices=["ne01", "unitree_g1", "unitree_g1_with_hands", "unitree_g1_24dof", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1",
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung", "fourier_gr3"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )
    parser.add_argument(
        "--headless",
        default=False,
        action="store_true",
        help="Do not open the MuJoCo viewer.",
    )
    parser.add_argument(
        "--max_frames",
        default=None,
        type=int,
        help="Retarget only the first N aligned frames for quick validation.",
    )

    parser.add_argument(
        "--no_height_adjust",
        default=False,
        action="store_true",
        help="Do not lift/lower the robot motion so the lowest body point is on the ground.",
    )

    parser.add_argument(
        "--no_root_origin_offset",
        default=False,
        action="store_true",
        help="Do not offset the initial root XY position to the world origin.",
    )

    args = parser.parse_args()


    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
   
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )
    
    frame_indices = range(1, len(smplx_data_frames)) if len(smplx_data_frames) > 1 else range(len(smplx_data_frames))
    if args.max_frames is not None:
        frame_indices = list(frame_indices)[: args.max_frames]
    qpos_list = []
    human_motion_list = []
    for frame_idx in frame_indices:
        qpos = retarget.retarget(smplx_data_frames[frame_idx])
        qpos_list.append(qpos.copy())
        human_motion_list.append(
            {body_name: [data[0].copy(), data[1].copy()] for body_name, data in retarget.scaled_human_data.items()}
        )

    qpos_list = np.asarray(qpos_list)
    qpos_list, root_deltas = adjust_qpos_to_ground(
        qpos_list,
        retarget.xml_file,
        height_adjust=not args.no_height_adjust,
        root_origin_offset=not args.no_root_origin_offset,
    )
    human_motion_list = offset_human_motion(human_motion_list, root_deltas)

    robot_motion_viewer = None
    if not args.headless:
        try:
            robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                                    motion_fps=aligned_fps,
                                                    transparent_robot=0,
                                                    record_video=args.record_video,
                                                    video_path=f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4",)
        except Exception as e:
            print(f"Warning: cannot create viewer ({e}), running headless.")

    if robot_motion_viewer is not None:
        frame_idx = 0
        fps_counter = 0
        fps_start_time = time.time()
        fps_display_interval = 2.0
        while True:
            qpos = qpos_list[frame_idx]
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=human_motion_list[frame_idx],
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=args.rate_limit,
                follow_camera=False,
            )

            frame_idx += 1
            fps_counter += 1
            if time.time() - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (time.time() - fps_start_time)
                print(f"Actual rendering FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = time.time()
            if args.loop:
                frame_idx %= len(qpos_list)
            elif frame_idx >= len(qpos_list):
                break
            
    if args.save_path is not None:
        import pickle
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")
            
      
    
    if robot_motion_viewer is not None:
        robot_motion_viewer.close()
