"""Retarget HoloSoMo position mocap with the independent Omni-first V3 path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from tqdm import tqdm

from general_motion_retargeting.holosoma_input import (
    build_solver_frames,
    load_holosoma_positions,
    load_joint_map,
    named_source_frames,
    resample_positions,
)
from general_motion_retargeting.terrain_contact_utils import build_terrain_contact_schedule
from general_motion_retargeting.terrain_geometry import SceneTransform, TerrainField
from general_motion_retargeting.wholebody_omni_gmr_v3 import (
    WholeBodyOmniGMRV3,
    sample_terrain_surface_pool,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "general_motion_retargeting/ik_configs/holosoma_to_ne01_wholebody_omni_gmr_v3.json"
DEFAULT_JOINT_MAP = ROOT / "general_motion_retargeting/joint_maps/holosoma_53.json"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _joint_names(model) -> list[str]:
    import mujoco

    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ]


def _scene_transform(config: dict, human_height: float | None) -> SceneTransform:
    scene = config["scene"]
    height = float(human_height or scene["default_human_height"])
    scale = float(scene["robot_height"]) / height
    return SceneTransform(
        rotation=np.asarray(scene["rotation"], dtype=float),
        scale=scale,
        translation=np.asarray(scene["translation"], dtype=float),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--terrain", required=True, type=Path)
    parser.add_argument("--save_path", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--joint_map", type=Path, default=DEFAULT_JOINT_MAP)
    parser.add_argument("--tgt_fps", type=float, default=50.0)
    parser.add_argument("--source_fps", type=float, default=120.0)
    parser.add_argument("--human_height", type=float, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--solver", choices=("daqp", "proxqp"), default="daqp")
    parser.add_argument("--debug_dir", type=Path, default=None)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    joint_map = load_joint_map(args.joint_map)
    positions, source_names, source_fps = load_holosoma_positions(
        args.motion, joint_map, default_fps=args.source_fps
    )
    positions = resample_positions(positions, source_fps, args.tgt_fps)
    if args.max_frames is not None:
        positions = positions[: args.max_frames]
    if len(positions) == 0:
        raise ValueError("Motion contains no frames")

    transform = _scene_transform(config, args.human_height)
    source_terrain = TerrainField.from_file(args.terrain)
    source_terrain.support_normal_min_z = float(
        config["terrain_contact"]["support_normal_min_z"]
    )
    terrain = source_terrain.transform(transform)
    transformed = transform.transform_points(positions)
    source_frames = named_source_frames(transformed, source_names)
    orientation_frames, orientation_valid = build_solver_frames(
        transformed, source_names, joint_map
    )

    raw_frames = named_source_frames(positions, source_names)
    contact_schedule = build_terrain_contact_schedule(
        raw_frames,
        source_terrain,
        transform,
        joint_map,
        args.tgt_fps,
        config["terrain_contact"],
    )
    sampling = config["interaction_graph"]["terrain_sampling"]
    pool = sample_terrain_surface_pool(
        terrain,
        transformed,
        points_per_square_meter=sampling["points_per_square_meter"],
        minimum=sampling["minimum"],
        maximum=sampling["maximum"],
    )
    retargeter = WholeBodyOmniGMRV3(
        args.config,
        terrain,
        pool,
        fps=args.tgt_fps,
        solver=args.solver,
    )

    qpos = []
    for frame, orientations, contact in tqdm(
        zip(source_frames, orientation_frames, contact_schedule),
        total=len(source_frames),
        desc="WholeBody Omni GMR V3",
    ):
        qpos.append(retargeter.retarget(
            frame,
            orientations["pelvis"][1],
            contact,
            chest_quaternion=orientations["spine3"][1],
        ))
    qpos = np.asarray(qpos)

    failures = sum(bool(frame["qp_failures"]) for frame in retargeter.diagnostics)
    minimum_slack = min(
        (frame["minimum_terrain_slack"] for frame in retargeter.diagnostics),
        default=np.inf,
    )
    max_velocity = max(
        (frame["max_velocity"] for frame in retargeter.diagnostics),
        default=0.0,
    )
    summary = {
        "frames": len(qpos),
        "fps": float(args.tgt_fps),
        "scene_scale": transform.scale,
        "minimum_terrain_slack": float(minimum_slack),
        "maximum_joint_velocity": float(max_velocity),
        "frames_with_qp_failure": int(failures),
    }
    payload = {
        "fps": float(args.tgt_fps),
        "root_pos": qpos[:, :3],
        "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": qpos[:, 7:],
        "qpos": qpos,
        "scene_transform": transform.to_dict(),
        "terrain_primitives": terrain.to_spec(),
        "contact_schedule": contact_schedule,
        "terrain_diagnostics": retargeter.diagnostics,
        "contact_metrics": summary,
        "joint_names": _joint_names(retargeter.model),
        "orientation_valid": orientation_valid,
        "source_joint_names": source_names,
        "source_motion": str(args.motion),
        "source_terrain": str(args.terrain),
        "robot_xml": str(retargeter.robot_xml),
        "algorithm": "wholebody_omni_gmr_v3",
    }

    pkl_path = args.save_path.with_suffix(".pkl")
    npz_path = args.save_path.with_suffix(".npz")
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as stream:
        pickle.dump(payload, stream)

    from convert_gmr_pkl_to_holosoma_npz import convert_file

    convert_file(pkl_path, npz_path, retargeter.model)
    debug_dir = args.debug_dir or pkl_path.parent / f"{pkl_path.stem}_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    (debug_dir / "frames.json").write_text(
        json.dumps(_jsonable(retargeter.diagnostics), indent=2)
    )
    (debug_dir / "scene.json").write_text(
        json.dumps(
            _jsonable({
                "scene_transform": transform.to_dict(),
                "terrain": terrain.to_spec(),
                "interaction_environment_pool": pool,
                "semantic_points": config["semantic_points"],
            }),
            indent=2,
        )
    )
    print(f"Saved V3 PKL: {pkl_path}")
    print(f"Saved HoloSoMo NPZ: {npz_path}")
    print(f"Saved diagnostics: {debug_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
