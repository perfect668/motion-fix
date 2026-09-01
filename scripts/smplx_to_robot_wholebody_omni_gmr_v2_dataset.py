"""Recursive batch entry point for the independent WholeBody Omni GMR V2 path."""

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
from scipy.spatial.transform import Rotation, Slerp

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
            result = {name: identity.copy() for name in names}
            index = {name: i for i, name in enumerate(names)}

            def make_basis(left, up):
                left = np.asarray(left, dtype=float)
                up = np.asarray(up, dtype=float)
                if np.linalg.norm(left) < 1e-7 or np.linalg.norm(up) < 1e-7:
                    return None
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
                    quat = Rotation.from_matrix(basis).as_quat()
                    result[name] = np.asarray([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)

            required = ("pelvis", "left_hip", "right_hip", "spine3")
            if all(name in index for name in required):
                hip_center = 0.5 * (frame[index["left_hip"]] + frame[index["right_hip"]])
                assign("pelvis", make_basis(
                    frame[index["left_hip"]] - frame[index["right_hip"]],
                    frame[index["spine3"]] - hip_center,
                ))
                shoulder_left = frame[index["left_shoulder"]] if "left_shoulder" in index else frame[index["left_hip"]]
                shoulder_right = frame[index["right_shoulder"]] if "right_shoulder" in index else frame[index["right_hip"]]
                assign("spine3", make_basis(
                    shoulder_left - shoulder_right,
                    frame[index["spine3"]] - hip_center,
                ))
            return result

        frames = [
            {name: (frame[index].copy(), orientations[name]) for index, name in enumerate(names)}
            for frame in joints
            for orientations in [frame_orientations(frame)]
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


def _expand_root_trajectory_in_object_xy(frames, source_npz, horizontal_scale, frame_fps):
    """Restore scene-scale XY travel without stretching the articulated body."""
    if np.isclose(horizontal_scale, 1.0):
        return frames
    if "object_pos_w" not in source_npz or "object_quat_w" not in source_npz:
        raise ValueError("scene horizontal scaling requires object_pos_w and object_quat_w")

    object_pos = np.asarray(source_npz["object_pos_w"], dtype=np.float64)
    object_quat = np.asarray(source_npz["object_quat_w"], dtype=np.float64)
    source_fps = float(np.asarray(source_npz["mocap_frame_rate"]).item())
    frame_times = np.arange(len(frames), dtype=np.float64) / float(frame_fps)
    source_times = np.arange(len(object_pos), dtype=np.float64) / source_fps
    sampled_pos = np.stack(
        [np.interp(frame_times, source_times, object_pos[:, axis]) for axis in range(3)], axis=-1
    )
    key_rotations = Rotation.from_quat(object_quat[:, [1, 2, 3, 0]])
    if len(object_quat) == 1:
        sampled_quat = Rotation.from_quat(
            np.repeat(key_rotations.as_quat(), len(frames), axis=0)
        )
    else:
        sampled_quat = Slerp(source_times, key_rotations)(
            np.clip(frame_times, source_times[0], source_times[-1])
        )

    expanded = []
    for frame, origin, rotation in zip(frames, sampled_pos, sampled_quat):
        root = np.asarray(frame["pelvis"][0], dtype=np.float64)
        root_local = rotation.inv().apply(root - origin)
        local_delta = np.array(
            [(horizontal_scale - 1.0) * root_local[0], (horizontal_scale - 1.0) * root_local[1], 0.0]
        )
        world_delta = rotation.apply(local_delta)
        expanded.append({
            name: (np.asarray(value[0]) + world_delta, value[1])
            for name, value in frame.items()
        })
    return expanded


def process_file(job):
    source, target, robot, config, fps, device, velocity_limit, scene_horizontal_scale = job
    source, target = pathlib.Path(source), pathlib.Path(target)
    try:
        root = pathlib.Path(__file__).resolve().parent.parent
        frames, aligned_fps, height, orientation_valid = load_motion_frames(source, root, fps)
        source_npz = np.load(source, allow_pickle=True)
        frames = _expand_root_trajectory_in_object_xy(
            frames, source_npz, scene_horizontal_scale, aligned_fps
        )
        retargeter = WholeBodyOmniGMRV2(src_human="smplx", tgt_robot=robot, actual_human_height=height, solver="proxqp", use_velocity_limit=True, velocity_limit=velocity_limit, motion_fps=fps, graph_config_path=config, verbose=False)
        retargeter.set_orientation_valid(orientation_valid)
        retargeter.set_motion_floor(frames)
        schedule = retargeter.build_contact_schedule(frames)
        indices = range(1, len(frames)) if len(frames) > 1 else range(len(frames))
        qpos = np.asarray([retargeter.retarget(frames[i], contact_frame=schedule[i]).copy() for i in indices])
        if len(qpos) == 0:
            return str(source), "empty output"
        root_xy_origin = qpos[0, :2].copy()
        qpos[:, :2] -= root_xy_origin
        # Preserve GRAIL/HoloSoMo interaction-object trajectory alongside the
        # retargeted robot.  The IK remains human-body based; object pose is
        # resampled to the exact exported 50 Hz frame timestamps.
        object_payload = {}
        if "object_pos_w" in source_npz and "object_quat_w" in source_npz:
            op = np.asarray(source_npz["object_pos_w"], dtype=np.float64)
            oq = np.asarray(source_npz["object_quat_w"], dtype=np.float64)
            source_fps = float(np.asarray(source_npz["mocap_frame_rate"]).item())
            out_indices = np.asarray(list(indices), dtype=float)
            times = out_indices / float(fps)
            source_times = np.arange(len(op), dtype=float) / float(source_fps)
            object_pos = np.stack([np.interp(times, source_times, op[:, a]) for a in range(3)], axis=-1)
            # The retargeter scales the human's world translation by the
            # pelvis/root scale, then the exported robot trajectory is shifted
            # so its first XY position is the origin. Apply the same world
            # transform to the interaction object to preserve the source
            # human-object relationship. Z remains in the asset's metric world
            # frame because the robot export does not normalize root Z.
            root_scale = float(retargeter.human_scale_table[retargeter.human_root_name])
            object_pos[:, :2] *= root_scale
            object_pos[:, :2] -= root_xy_origin
            object_payload["object_pos_w"] = object_pos
            # Slerp is used for rotations; GRAIL stores wxyz and so does the
            # HoloSoMo motion schema.
            key_rots = Rotation.from_quat(oq[:, [1, 2, 3, 0]])
            valid_times = np.clip(times, source_times[0], source_times[-1])
            object_payload["object_quat_w"] = Slerp(source_times, key_rots)(valid_times).as_quat()[:, [3, 0, 1, 2]]
            if "object_scale" in source_npz:
                scale = np.asarray(source_npz["object_scale"], dtype=np.float64)
                scene_scale = np.asarray(
                    [scene_horizontal_scale * root_scale, scene_horizontal_scale * root_scale, root_scale],
                    dtype=np.float64,
                ).reshape(3, 1)
                object_payload["object_scale"] = scale * scene_scale
            if "object_asset_path" in source_npz:
                object_payload["object_asset_path"] = str(np.asarray(source_npz["object_asset_path"]).item())
            object_payload["object_fps"] = float(fps)
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
            result = {"fps": 50.0, "root_pos": root_pos, "root_rot": root_rot, "dof_pos": dof_pos, "local_body_pos": local_body_pos, "link_body_list": body_names}
            result.update(object_payload)
            pickle.dump(result, file)
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
    parser.add_argument(
        "--scene_horizontal_scale",
        type=float,
        default=1.0,
        help="Object-local XY trajectory multiplier; 1.0 preserves existing behavior.",
    )
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parent.parent
    source_root, target_root = pathlib.Path(args.src_folder).resolve(), pathlib.Path(args.tgt_folder).resolve()
    config = args.config or str(root / "general_motion_retargeting" / "ik_configs" / "smplx_to_ne01_wholebody_omni_gmr_v2.json")
    jobs = []
    for source in sorted(source_root.rglob("*.npz")):
        if source.name.endswith("_stagei.npz"):
            continue
        target = target_root / source.relative_to(source_root).with_suffix(".pkl")
        if target.exists() and not args.override:
            continue
        jobs.append((
            str(source), str(target), args.robot, config, args.tgt_fps, args.device,
            args.velocity_limit, args.scene_horizontal_scale,
        ))
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
        report = target_root / "wholebody_omni_gmr_v2_failures.txt"
        report.write_text("\n\n".join(f"{source}\n{error}" for source, error in failures), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} motions failed; see {report}")


if __name__ == "__main__":
    main()
