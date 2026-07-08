"""Batch retarget SMPL-X motions to a GMR robot-motion dataset.

Purpose:
    Convert a folder of SMPL-X .npz files into robot-motion .pkl files for a
    target robot, such as unitree_g1 or unitree_g1_24dof.

Typical usage:
    conda run --no-capture-output -n gmr python -u scripts/smplx_to_robot_dataset.py \
        --src_folder data/smplx_data/selected_10_12h \
        --tgt_folder data/retarget_data/g1_24dof/sonic_smpl_selected_10_12h \
        --robot unitree_g1_24dof \
        --num_cpus 8 \
        --device cpu

Outputs:
    One .pkl per source motion. Each robot-motion pkl contains fps, root_pos,
    root_rot, dof_pos, local_body_pos, and link_body_list.

Config paths:
    --body_model_folder defaults to assets/body_models.
    --hard_motions_folder defaults to assets/hard_motions.
"""

import argparse
import json
import pathlib
import os
import multiprocessing as mp

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from natsort import natsorted
from rich import print
import torch
import pickle

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import IK_CONFIG_ROOT
import gc
import time
import tracemalloc

try:
    import psutil
except ImportError:
    psutil = None


def check_memory(threshold_gb=30):  # adjust based on your available memory
    if psutil is None:
        return False
    mem = psutil.virtual_memory()
    used_memory_gb = (mem.total - mem.available) / (1024 ** 3)
    available_memory_gb = mem.available / (1024 ** 3)
    if available_memory_gb < threshold_gb:
        print(f"[WARNING] Memory usage:{used_memory_gb:.2f} GB, available:{available_memory_gb:.2f} GB, exceeding the threshold of {threshold_gb} GB.")
        return True
    return False


HERE = pathlib.Path(__file__).parent


def process_file(
    smplx_file_path,
    tgt_file_path,
    tgt_robot,
    smplx_folder,
    tgt_folder,
    total_files,
    device,
    verbose=False,
    use_velocity_limit=False,
    velocity_limit=3 * np.pi,
):
    def log_memory(message):
        if verbose and psutil is not None:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024 ** 3)  # Convert to GB
            print(f"[MEMORY] {message}: {memory_usage:.2f} GB")
    
    # Start memory tracking if verbose
    if verbose:
        tracemalloc.start()
        
    # Initial checks (with optional logging)
    log_memory("Initial memory usage")
    
    num_pause = 0
    while check_memory():
        print(f"[PAUSE] Paused processing {smplx_file_path} to prevent memory overflow. num_pause: {num_pause}")
        time.sleep(60*2)
        num_pause += 1
        if num_pause > 10:
            print(f"[ERROR] Memory usage is still high after 10 pauses. Exiting.")
            return

    try:
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(smplx_file_path, smplx_folder)
        mocap_frame_rate = smplx_data["mocap_frame_rate"]
        log_memory("After loading SMPL-X data")
    except Exception as e:
        print(f"Error loading {smplx_file_path}: {e}")
        return
    
  
    tgt_fps = 30
    try:
        smplx_frame_data_list, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    except Exception as e:
        print(f"Error processing {smplx_file_path}: {e}")
        return
    
    try:
        # retarget
        retargeter = GMR(
            src_human="smplx",
            tgt_robot=tgt_robot,
            actual_human_height=actual_human_height,
            verbose=verbose,
            use_velocity_limit=use_velocity_limit,
            velocity_limit=velocity_limit,
        )
        qpos_list = []
        for smplx_frame_data in smplx_frame_data_list:
            qpos = retargeter.retarget(smplx_frame_data)
            qpos_list.append(qpos.copy())

        qpos_list = np.array(qpos_list)
    except Exception as e:
        print(f"Error retargeting {smplx_file_path}: {type(e).__name__}: {e}")
        return

    log_memory("After retargeting")
    
    try:
        kinematics_model = KinematicsModel(retargeter.xml_file, device=device)
    except Exception as e:
        print(f"Error loading kinematics model for {smplx_file_path}: {type(e).__name__}: {e}")
        return

    try:
        root_pos = qpos_list[:, :3]
    except Exception as e:
        print(f"Error processing {smplx_file_path}: {e}")
        return
    root_rot = qpos_list[:, 3:7]
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
    dof_pos = qpos_list[:, 7:]
    num_frames = root_pos.shape[0]

    fk_root_pos = torch.zeros((num_frames, 3), device=device)
    fk_root_rot = torch.zeros((num_frames, 4), device=device)
    fk_root_rot[:, -1] = 1.0

    try:
        local_body_pos, _ = kinematics_model.forward_kinematics(
            fk_root_pos, fk_root_rot, torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
        )
    except Exception as e:
        print(f"Error running local FK for {smplx_file_path}: {type(e).__name__}: {e}")
        return

    log_memory("After forward kinematics")

    body_names = kinematics_model.body_names
    
    HEIGHT_ADJUST = True
    if HEIGHT_ADJUST:
        try:
            # height adjust to ensure the lowerset part is on the ground
            body_pos, _ = kinematics_model.forward_kinematics(torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
                                                            torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
                                                            torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)) # TxNx3
            ground_offset = 0.0
            lowerst_height = torch.min(body_pos[..., 2]).item()
            root_pos[:, 2] = root_pos[:, 2] - lowerst_height + ground_offset # make sure motion on the ground
        except Exception as e:
            print(f"Error running height-adjust FK for {smplx_file_path}: {type(e).__name__}: {e}")
            return
        
    ROOT_ORIGIN_OFFSET = True
    if ROOT_ORIGIN_OFFSET:
        # offset using the first frame
        root_pos[:, :2] -= root_pos[0, :2]
        
        
    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "link_body_list": body_names,
    }


    os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
    with open(tgt_file_path, "wb") as f:
        pickle.dump(motion_data, f)
        
    # Progress print based on tgt_folder
    done = 0
    for root, _, files in os.walk(tgt_folder):
        done += len([f for f in files if f.endswith('.pkl')])
    print(f"Processed {done}/{total_files}: {tgt_file_path}")
    
    if verbose:
        # Get memory snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        print("\nTop 10 memory-consuming lines:")
        for stat in top_stats[:10]:
            print(stat)
        
        tracemalloc.stop()
        
    # clean cache
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()
    


def run_process_pool(
    args_list,
    total_files,
    device,
    verbose,
    num_cpus,
    use_velocity_limit=False,
    velocity_limit=3 * np.pi,
):
    process_args = [
        args + (total_files, device, verbose, use_velocity_limit, velocity_limit)
        for args in args_list
    ]
    if num_cpus <= 1:
        for args in process_args:
            process_file(*args)
        return

    if str(device).startswith("cuda"):
        print("Using multiprocessing start method: spawn for CUDA FK.")
        context = mp.get_context("spawn")
    else:
        context = mp.get_context()

    with context.Pool(num_cpus) as pool:
        pool.starmap(process_file, process_args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--src_folder", type=str,
                        required=True,
                        )
    parser.add_argument("--tgt_folder", type=str,
                        required=True,
                        )
    
    parser.add_argument("--override", default=False, action="store_true")
    parser.add_argument("--num_cpus", default=4, type=int)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for forward kinematics: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--body_model_folder",
        type=str,
        default=None,
        help="Path to SMPL-X body models. Defaults to assets/body_models.",
    )
    parser.add_argument(
        "--hard_motions_folder",
        type=str,
        default=None,
        help="Path to hard motion exclusion txt files. Defaults to assets/hard_motions.",
    )
    parser.add_argument(
        "--disable_hard_motion_filter",
        action="store_true",
        help="Do not exclude motions listed in assets/hard_motions/*.txt.",
    )
    parser.add_argument(
        "--disable_name_exclude_filter",
        action="store_true",
        help="Do not exclude motions by built-in filename keywords such as crawl or _lie.",
    )
    parser.add_argument(
        "--use_velocity_limit",
        action="store_true",
        help="Enable GMR IK joint velocity limits during retargeting.",
    )
    parser.add_argument(
        "--velocity_limit",
        type=float,
        default=3 * np.pi,
        help="Joint velocity limit in rad/s for GMR IK when --use_velocity_limit is set.",
    )
    args = parser.parse_args()
    
    # print the total number of cpus and gpus
    print(f"Total CPUs: {mp.cpu_count()}")
    print(f"Using {args.num_cpus} CPUs.")
    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        device = "cuda:0"
    else:
        device = args.device
    print(f"Using FK device: {device}")
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder

    smplx_folder = pathlib.Path(args.body_model_folder) if args.body_model_folder else HERE / ".." / "assets" / "body_models"
    hard_motions_folder = pathlib.Path(args.hard_motions_folder) if args.hard_motions_folder else HERE / ".." / "assets" / "hard_motions"

    verbose = False

    hard_motions = []
    if not args.disable_hard_motion_filter:
        hard_motions_paths = [hard_motions_folder / "0.txt",
                              hard_motions_folder / "1.txt"]
        for hard_motions_path in hard_motions_paths:
            if not hard_motions_path.exists():
                continue
            with open(hard_motions_path, "r") as f:
                for line in f:
                    if "Motion:" in line:
                        motion_path = line.split(":")[1].strip()
                    else:
                        continue
                    motion_path = motion_path.split(",")[0].strip().split(".")[0]
                    hard_motions.append(motion_path)
                
                
    all_args_list = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in natsorted(filenames):
            if filename.endswith("_stagei.npz"):
                continue
            if filename.endswith((".pkl", ".npz")):
                smplx_file_path = os.path.join(dirpath, filename)
                tgt_file_path = smplx_file_path.replace(src_folder, tgt_folder).replace(".npz", ".pkl")
                all_args_list.append((smplx_file_path, tgt_file_path, args.robot, smplx_folder, tgt_folder))
    print("full args_list:", len(all_args_list))
    
    # remove hard and infeasible motions
    exclude_file_content = []
    if not args.disable_name_exclude_filter:
        exclude_file_content = ["BMLrub", "EKUT", "crawl", "_lie", "upstairs", "downstairs"]
    
    expected_args_list = []
    pending_args_list = []
    for arguments in all_args_list:
        motion_name = arguments[0].split("/")[-1].split('.')[0]
        if motion_name in hard_motions:
            continue
        if any(content in motion_name for content in exclude_file_content):
            continue
        expected_args_list.append(arguments)
        if not os.path.exists(arguments[1]) or args.override:
            pending_args_list.append(arguments)
    args_list = pending_args_list
    
    
    print("expected args_list:", len(expected_args_list))
    print("pending args_list:", len(args_list))
    
    total_files = len(expected_args_list)
    print(f"Total number of files to process: {total_files}")
    run_process_pool(
        args_list,
        total_files,
        device,
        verbose,
        args.num_cpus,
        use_velocity_limit=args.use_velocity_limit,
        velocity_limit=args.velocity_limit,
    )

    print("Done. Saved to ", tgt_folder)


if __name__ == "__main__":
    main()
