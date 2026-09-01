"""Single-motion entry point for the independent WholeBody Omni GMR V2 path."""

from __future__ import annotations

import argparse
import pathlib
import pickle

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from smplx.joint_names import JOINT_NAMES

from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast, load_smplx_file
from general_motion_retargeting.wholebody_omni_gmr_v2 import WholeBodyOmniGMRV2


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

        def frame_orientations(frame: np.ndarray) -> dict[str, np.ndarray]:
            """Construct only the reliable pelvis/spine orientations from positions."""
            result = {name: identity.copy() for name in names}
            index = {name: i for i, name in enumerate(names)}

            def make_basis(origin, left, up):
                left = np.asarray(left, dtype=float)
                up = np.asarray(up, dtype=float)
                if np.linalg.norm(left) < 1e-7 or np.linalg.norm(up) < 1e-7:
                    return None
                # Robot convention: +X forward, +Y left, +Z up.
                y = left / np.linalg.norm(left)
                z = up - y * np.dot(y, up)
                if np.linalg.norm(z) < 1e-7:
                    return None
                z /= np.linalg.norm(z)
                x = np.cross(y, z)
                x /= max(np.linalg.norm(x), 1e-7)
                z = np.cross(x, y)
                return np.column_stack((x, y, z))

            def assign(name, basis):
                if basis is not None and name in index:
                    # scipy uses x,y,z,w; mink/this loader uses w,x,y,z.
                    q = Rotation.from_matrix(basis).as_quat()
                    result[name] = np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float32)

            if all(key in index for key in ("left_hip", "right_hip", "spine3")):
                assign("pelvis", make_basis(
                    frame[index["pelvis"]] if "pelvis" in index else frame[index["spine3"]],
                    frame[index["left_hip"]] - frame[index["right_hip"]],
                    frame[index["spine3"]] - 0.5 * (frame[index["left_hip"]] + frame[index["right_hip"]]),
                ))
                assign("spine3", make_basis(
                    frame[index["spine3"]],
                    frame[index["left_shoulder"]] - frame[index["right_shoulder"]]
                    if "left_shoulder" in index and "right_shoulder" in index
                    else frame[index["left_hip"]] - frame[index["right_hip"]],
                    frame[index["spine3"]] - 0.5 * (frame[index["left_hip"]] + frame[index["right_hip"]]),
                ))
            return result

        frames = [
            {name: (frame[index].copy(), quat) for index, name in enumerate(names)
             for quat in [frame_orientations(frame).get(name, identity.copy())]}
            for frame in joints
        ]
        frames_orientation_valid = False
        height = float(np.asarray(smplx_data["height"]).item()) if "height" in smplx_data.files else 1.66
        return frames, aligned_fps, height, frames_orientation_valid

    data, model, output, height = load_smplx_file(source, root / "assets" / "body_models")
    frames, aligned_fps = get_smplx_data_offline_fast(data, model, output, tgt_fps=fps)
    surface_names = ("left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel")
    joint_positions = output.joints.detach().cpu().numpy()
    source_fps = float(np.asarray(data["mocap_frame_rate"]).item())
    target_times = np.arange(len(frames), dtype=float) / float(aligned_fps)
    source_times = np.arange(len(joint_positions), dtype=float) / source_fps
    for name in surface_names:
        index = JOINT_NAMES.index(name)
        if index >= joint_positions.shape[1]:
            continue
        values = np.stack([
            np.interp(target_times, source_times, joint_positions[:, index, axis])
            for axis in range(3)
        ], axis=-1)
        for frame, point in zip(frames, values):
            foot = "left_foot" if name.startswith("left_") else "right_foot"
            frame[name] = (point.astype(np.float32), np.asarray(frame[foot][1], dtype=np.float32).copy())
    return frames, aligned_fps, height, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_file", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--robot", default="ne01")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tgt_fps", type=float, default=50.0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--velocity_limit", type=float, default=3 * np.pi)
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    config = args.config or str(root / "general_motion_retargeting" / "ik_configs" / "smplx_to_ne01_wholebody_omni_gmr_v2.json")
    frames, fps, height, orientation_valid = load_motion_frames(pathlib.Path(args.smplx_file), root, args.tgt_fps)
    retargeter = WholeBodyOmniGMRV2(
        src_human="smplx", tgt_robot=args.robot, actual_human_height=height,
        solver="proxqp", use_velocity_limit=True, velocity_limit=args.velocity_limit,
        motion_fps=args.tgt_fps, graph_config_path=config, verbose=True,
    )
    retargeter.set_orientation_valid(orientation_valid)
    retargeter.set_motion_floor(frames)
    schedule = retargeter.build_contact_schedule(frames)
    indices = list(range(1, len(frames))) if len(frames) > 1 else [0]
    if args.max_frames is not None:
        indices = indices[:args.max_frames]
    qpos = np.asarray([retargeter.retarget(frames[i], contact_frame=schedule[i]).copy() for i in indices])
    if len(qpos) == 0:
        raise RuntimeError("No retargeted frames were produced")
    qpos[:, :2] -= qpos[0, :2]
    root_pos, root_rot, dof_pos = qpos[:, :3], qpos[:, 3:7][:, [1, 2, 3, 0]], qpos[:, 7:]
    local_body_pos, body_names = None, None
    try:
        kinematics = KinematicsModel(retargeter.xml_file, device=args.device)
        device = torch.device(args.device)
        with torch.no_grad():
            zeros = torch.zeros((len(qpos), 3), dtype=torch.float32, device=device)
            identity = torch.zeros((len(qpos), 4), dtype=torch.float32, device=device)
            identity[:, -1] = 1.0
            local, _ = kinematics.forward_kinematics(zeros, identity, torch.as_tensor(dof_pos, dtype=torch.float32, device=device))
        local_body_pos, body_names = local.cpu().numpy(), kinematics.body_names
    except Exception as exc:
        print(f"Warning: local FK export skipped: {type(exc).__name__}: {exc}")
    destination = pathlib.Path(args.save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file:
        pickle.dump({"fps": float(fps), "root_pos": root_pos, "root_rot": root_rot, "dof_pos": dof_pos, "local_body_pos": local_body_pos, "link_body_list": body_names}, file)
    print(f"Saved whole-body Omni GMR V2 motion to {destination}")


if __name__ == "__main__":
    main()
