"""Recursive batch entry point for the independent WholeBody Omni GMR path."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import pickle
import traceback

import numpy as np
import torch
from tqdm import tqdm
from smplx.joint_names import JOINT_NAMES
from scipy.interpolate import interp1d

from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast, load_smplx_file
from general_motion_retargeting.wholebody_omni_gmr import WholeBodyOmniGMR


def load_motion_frames(source: pathlib.Path, root: pathlib.Path, fps: float):
    smplx_data = np.load(source, allow_pickle=True)
    if "global_joint_positions" in smplx_data.files:
        joints = np.asarray(smplx_data["global_joint_positions"], dtype=np.float32)
        if joints.ndim != 3 or joints.shape[-1] != 3:
            raise ValueError(f"global_joint_positions must have shape (T, J, 3), got {joints.shape}")
        source_fps = float(np.asarray(smplx_data["fps"]).item()) if "fps" in smplx_data.files else fps
        if source_fps <= 0 or fps <= 0:
            raise ValueError(f"FPS values must be positive, got src={source_fps}, target={fps}")
        if len(joints) > 1 and not np.isclose(source_fps, fps):
            new_num_frames = max(1, int(np.floor((len(joints) - 1) * fps / source_fps)) + 1)
            original_time = np.arange(len(joints), dtype=np.float32)
            target_time = np.arange(new_num_frames, dtype=np.float32) * (source_fps / fps)
            resampled = []
            for joint_index in range(joints.shape[1]):
                axes = [
                    interp1d(original_time, joints[:, joint_index, axis], kind="linear")(target_time)
                    for axis in range(3)
                ]
                resampled.append(np.stack(axes, axis=-1))
            joints = np.stack(resampled, axis=1).astype(np.float32)
            aligned_fps = fps
        else:
            aligned_fps = source_fps if len(joints) > 1 else fps
        names = JOINT_NAMES[: joints.shape[1]]
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        frames = [
            {name: (frame[index].copy(), identity.copy()) for index, name in enumerate(names)}
            for frame in joints
        ]
        height = float(np.asarray(smplx_data["height"]).item()) if "height" in smplx_data.files else 1.66
        return frames, aligned_fps, height, False

    data, body_model, output, height = load_smplx_file(source, root / "assets" / "body_models")
    frames, aligned_fps = get_smplx_data_offline_fast(data, body_model, output, tgt_fps=fps)
    surface_names = ("left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel")
    joint_positions = output.joints.detach().cpu().numpy()
    source_fps = float(np.asarray(data["mocap_frame_rate"]).item())
    target_times = np.arange(len(frames), dtype=float) / float(aligned_fps)
    source_times = np.arange(len(joint_positions), dtype=float) / source_fps
    for name in surface_names:
        index = JOINT_NAMES.index(name)
        if index >= joint_positions.shape[1]:
            continue
        values = np.stack([np.interp(target_times, source_times, joint_positions[:, index, axis]) for axis in range(3)], axis=-1)
        for frame, point in zip(frames, values):
            foot = "left_foot" if name.startswith("left_") else "right_foot"
            frame[name] = (point.astype(np.float32), np.asarray(frame[foot][1], dtype=np.float32).copy())
    return frames, aligned_fps, height, True


def process_file(job):
    source, target, robot, config, fps, device, velocity_limit = job
    source, target = pathlib.Path(source), pathlib.Path(target)
    try:
        root = pathlib.Path(__file__).resolve().parent.parent
        frames, aligned_fps, height, orientation_valid = load_motion_frames(source, root, fps)
        retargeter = WholeBodyOmniGMR(src_human="smplx", tgt_robot=robot, actual_human_height=height, solver="proxqp", use_velocity_limit=True, velocity_limit=velocity_limit, motion_fps=fps, graph_config_path=config, verbose=False)
        retargeter.set_orientation_valid(orientation_valid)
        retargeter.set_motion_floor(frames)
        schedule = retargeter.build_contact_schedule(frames)
        indices = range(1, len(frames)) if len(frames) > 1 else range(len(frames))
        qpos = np.asarray([retargeter.retarget(frames[i], contact_frame=schedule[i]).copy() for i in indices])
        if len(qpos) == 0:
            return str(source), "empty output"
        qpos[:, :2] -= qpos[0, :2]
        root_pos, root_rot, dof_pos = qpos[:, :3], qpos[:, 3:7][:, [1, 2, 3, 0]], qpos[:, 7:]
        local_body_pos, body_names = None, None
        try:
            kinematics = KinematicsModel(retargeter.xml_file, device=device)
            resolved = torch.device(device)
            with torch.no_grad():
                zeros = torch.zeros((len(qpos), 3), dtype=torch.float32, device=resolved)
                identity = torch.zeros((len(qpos), 4), dtype=torch.float32, device=resolved); identity[:, -1] = 1.0
                local, _ = kinematics.forward_kinematics(zeros, identity, torch.as_tensor(dof_pos, dtype=torch.float32, device=resolved))
            local_body_pos, body_names = local.cpu().numpy(), kinematics.body_names
        except Exception:
            pass
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as file:
            pickle.dump({"fps": float(aligned_fps), "root_pos": root_pos, "root_rot": root_rot, "dof_pos": dof_pos, "local_body_pos": local_body_pos, "link_body_list": body_names}, file)
        return str(source), None
    except Exception as exc:
        return str(source), f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", required=True)
    parser.add_argument("--tgt_folder", required=True)
    parser.add_argument("--robot", default="ne01")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tgt_fps", type=float, default=50.0)
    parser.add_argument("--num_cpus", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--velocity_limit", type=float, default=3 * np.pi)
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parent.parent
    source_root, target_root = pathlib.Path(args.src_folder).resolve(), pathlib.Path(args.tgt_folder).resolve()
    config = args.config or str(root / "general_motion_retargeting" / "ik_configs" / "smplx_to_ne01_wholebody_omni_gmr.json")
    jobs = []
    for source in sorted(source_root.rglob("*.npz")):
        if source.name.endswith("_stagei.npz"):
            continue
        target = target_root / source.relative_to(source_root).with_suffix(".pkl")
        if target.exists() and not args.override:
            continue
        jobs.append((str(source), str(target), args.robot, config, args.tgt_fps, args.device, args.velocity_limit))
    print(f"Pending motions: {len(jobs)}")
    failures = []
    if args.num_cpus <= 1:
        iterator = map(process_file, jobs)
    else:
        pool = mp.get_context("spawn").Pool(args.num_cpus)
        iterator = pool.imap_unordered(process_file, jobs)
    try:
        for source, error in tqdm(iterator, total=len(jobs)):
            if error:
                failures.append((source, error))
    finally:
        if args.num_cpus > 1:
            pool.close(); pool.join()
    if failures:
        target_root.mkdir(parents=True, exist_ok=True)
        report = target_root / "wholebody_omni_gmr_failures.txt"
        report.write_text("\n\n".join(f"{source}\n{error}" for source, error in failures), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} motions failed; see {report}")


if __name__ == "__main__":
    main()
