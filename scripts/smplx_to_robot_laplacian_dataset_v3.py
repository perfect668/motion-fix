"""Recursive batch entry point for the independent NE01 V3 pipeline."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import pickle
import traceback

import numpy as np
import torch
from tqdm import tqdm

from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.laplacian_soft_retarget_v3 import (
    LaplacianSoftContactRetargetingV3,
)
from general_motion_retargeting.utils.smpl import (
    get_smplx_data_offline_fast,
    load_smplx_file,
)


def process_file(arguments):
    source_path, target_path, robot, body_model_folder, config, fps, device, velocity_limit = arguments
    source_path = pathlib.Path(source_path)
    target_path = pathlib.Path(target_path)
    try:
        smplx_data, body_model, smplx_output, actual_height = load_smplx_file(
            source_path, pathlib.Path(body_model_folder)
        )
        frame_data, aligned_fps = get_smplx_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=fps
        )
        retargeter = LaplacianSoftContactRetargetingV3(
            src_human="smplx",
            tgt_robot=robot,
            actual_human_height=actual_height,
            solver="proxqp",
            use_velocity_limit=True,
            velocity_limit=velocity_limit,
            motion_fps=fps,
            config_path=config,
            verbose=False,
        )
        retargeter.set_motion_floor(frame_data)
        indices = range(1, len(frame_data)) if len(frame_data) > 1 else range(len(frame_data))
        preview = int(retargeter.v3_config.get("contact", {}).get("future_preview_frames", 8))
        qpos = np.asarray(
            [
                retargeter.retarget(
                    frame_data[index],
                    future_frames=frame_data[index : index + preview + 1],
                ).copy()
                for index in indices
            ],
            dtype=float,
        )
        if len(qpos) == 0:
            return str(source_path), "empty output"
        qpos[:, :2] -= qpos[0, :2]
        root_pos = qpos[:, :3]
        root_rot_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]
        dof_pos = qpos[:, 7:]
        local_body_pos = None
        body_names = None
        try:
            kinematics = KinematicsModel(retargeter.xml_file, device=device)
            resolved_device = torch.device(device)
            with torch.no_grad():
                zero_root = torch.zeros((len(qpos), 3), dtype=torch.float32, device=resolved_device)
                identity_root = torch.zeros((len(qpos), 4), dtype=torch.float32, device=resolved_device)
                identity_root[:, -1] = 1.0
                local_body_pos_t, _ = kinematics.forward_kinematics(
                    zero_root,
                    identity_root,
                    torch.as_tensor(dof_pos, dtype=torch.float32, device=resolved_device),
                )
            local_body_pos = local_body_pos_t.detach().cpu().numpy()
            body_names = kinematics.body_names
        except Exception:
            pass
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as file:
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
        if bool(retargeter.v3_config.get("diagnostics", {}).get("save_per_frame_csv", True)):
            retargeter.save_diagnostics(target_path.with_suffix(".diagnostics.csv"))
        return str(source_path), None
    except Exception as exc:
        return str(source_path), f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def main() -> None:
    parser = argparse.ArgumentParser(description="NE01 V3 staged SMPL-X dataset retargeting")
    parser.add_argument("--src_folder", required=True)
    parser.add_argument("--tgt_folder", required=True)
    parser.add_argument("--robot", default="ne01")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tgt_fps", type=float, default=50.0)
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--velocity_limit", type=float, default=3 * np.pi)
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    source_root = pathlib.Path(args.src_folder).resolve()
    target_root = pathlib.Path(args.tgt_folder).resolve()
    body_model_folder = here.parent / "assets" / "body_models"
    config = args.config
    if config is None and args.robot == "ne01":
        config = str(
            here.parent
            / "general_motion_retargeting"
            / "ik_configs"
            / "smplx_to_ne01_laplacian_soft_v3.json"
        )
    jobs = []
    for source_path in sorted(source_root.rglob("*.npz")):
        if source_path.name.endswith("_stagei.npz"):
            continue
        target_path = target_root / source_path.relative_to(source_root).with_suffix(".pkl")
        if target_path.exists() and not args.override:
            continue
        jobs.append((str(source_path), str(target_path), args.robot, str(body_model_folder), config, args.tgt_fps, args.device, args.velocity_limit))
    print(f"Pending NE01 V3 motions: {len(jobs)}")
    if not jobs:
        return
    failures = []
    if args.num_cpus <= 1:
        iterator = map(process_file, jobs)
        for source_path, error in tqdm(iterator, total=len(jobs)):
            if error:
                failures.append((source_path, error))
    else:
        context = mp.get_context("spawn")
        with context.Pool(args.num_cpus) as pool:
            for source_path, error in tqdm(pool.imap_unordered(process_file, jobs), total=len(jobs)):
                if error:
                    failures.append((source_path, error))
    if failures:
        report = target_root / "laplacian_soft_v3_failures.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n\n".join(f"{path}\n{error}" for path, error in failures), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} motions failed; see {report}")
    print(f"Completed NE01 V3 motions: {len(jobs)} -> {target_root}")


if __name__ == "__main__":
    main()
