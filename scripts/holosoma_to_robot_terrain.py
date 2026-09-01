"""Retarget HoloSoMo position mocap onto NE01 in an explicit terrain scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from general_motion_retargeting.holosoma_input import (
    build_solver_frames,
    load_holosoma_positions,
    load_joint_map,
    named_source_frames,
    resample_positions,
)
from general_motion_retargeting.terrain_contact_utils import build_terrain_contact_schedule
from general_motion_retargeting.terrain_geometry import SceneTransform, TerrainField
from general_motion_retargeting.wholebody_terrain_omni_gmr_v2 import WholeBodyTerrainOmniGMRV2


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "general_motion_retargeting/ik_configs/smplx_to_ne01_terrain_wholebody_omni_gmr_v2.json"
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


def _scene_transform(config_path: Path, config: dict) -> tuple[SceneTransform, float]:
    scene = config["scene_transform"]
    base_path = Path(config["kinematic_config"])
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    base = json.loads(base_path.read_text())
    human_height = float(scene["actual_human_height"])
    scale = scene.get("scene_scale")
    if scale is None:
        root = base["human_root_name"]
        scale = float(base["human_scale_table"][root]) * human_height / float(base["human_height_assumption"])
    rotation = np.asarray(scene["rotation"], dtype=float)
    translation = np.asarray(scene["translation"], dtype=float)
    source_floor = float(scene["source_floor_z"])
    solver_floor = float(scene["solver_floor_z"])
    transformed_floor = float(scale) * float((rotation @ np.array([0.0, 0.0, source_floor]))[2]) + translation[2]
    translation = translation.copy()
    translation[2] += solver_floor - transformed_floor
    return SceneTransform(rotation, float(scale), translation), human_height


def _joint_names(model) -> list[str]:
    import mujoco

    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ]


def _body_names(model) -> list[str]:
    import mujoco

    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(1, model.nbody)
    ]


def summarize(qpos: np.ndarray, diagnostics: list[dict], fps: float, model, velocity_limit: float) -> dict:
    distances = [
        float(item["signed_distance"])
        for frame in diagnostics for item in frame["slacks"].values()
    ]
    slacks = [float(item["slack"]) for frame in diagnostics for item in frame["slacks"].values()]
    penetrations = np.maximum(0.0, -np.asarray(distances, dtype=float))
    margin_violations = np.maximum(0.0, -np.asarray(slacks, dtype=float))
    region_counts: dict[str, int] = {}
    for frame in diagnostics:
        for name, item in frame["slacks"].items():
            if float(item["signed_distance"]) < 0.0:
                region = name.split(":", 1)[-1].split("_shell", 1)[0]
                region_counts[region] = region_counts.get(region, 0) + 1

    episodes, surface_switches, cumulative_slip = {}, {}, {}
    channels = sorted({name for frame in diagnostics for name in frame.get("contacts", {})})
    for channel in channels:
        active_start = None
        channel_episodes = []
        previous_surface = None
        switches = 0
        slip = 0.0
        for frame_index, frame in enumerate(diagnostics):
            item = frame.get("contacts", {}).get(channel, {"state": "NONE"})
            active = item.get("state") != "NONE"
            if active and active_start is None:
                active_start = frame_index
            if not active and active_start is not None:
                channel_episodes.append([active_start, frame_index - 1])
                active_start = None
            surface = item.get("surface_id") if active else None
            if surface is not None and previous_surface is not None and surface != previous_surface:
                switches += 1
            if surface is not None:
                previous_surface = surface
            if item.get("state") == "STATIC":
                slip += float(item.get("tangential_speed", 0.0)) / fps
        if active_start is not None:
            channel_episodes.append([active_start, len(diagnostics) - 1])
        episodes[channel] = channel_episodes
        surface_switches[channel] = switches
        cumulative_slip[channel] = slip

    articulated = qpos[:, 7:]
    dt = 1.0 / fps
    velocity = np.gradient(articulated, dt, axis=0, edge_order=1) if len(qpos) > 1 else np.zeros_like(articulated)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=1) if len(qpos) > 1 else np.zeros_like(velocity)
    jerk = np.gradient(acceleration, dt, axis=0, edge_order=1) if len(qpos) > 1 else np.zeros_like(acceleration)
    joint_limit_violations = 0
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == 0 or not model.jnt_limited[joint_id]:
            continue
        qadr = int(model.jnt_qposadr[joint_id])
        lower, upper = model.jnt_range[joint_id]
        joint_limit_violations += int(np.count_nonzero((qpos[:, qadr] < lower - 1e-6) | (qpos[:, qadr] > upper + 1e-6)))
    orientation_errors = [
        float(value)
        for frame in diagnostics for value in frame.get("foot_orientation_error_rad", {}).values()
        if value > 0.0
    ]
    return {
        "frames": len(qpos),
        "fps": fps,
        "max_penetration_m": float(penetrations.max(initial=0.0)),
        "p99_penetration_m": float(np.percentile(penetrations, 99)) if len(penetrations) else 0.0,
        "max_margin_violation_m": float(margin_violations.max(initial=0.0)),
        "penetration_counts_by_region": region_counts,
        "contact_episodes": episodes,
        "surface_switches": surface_switches,
        "static_cumulative_slip_m": cumulative_slip,
        "max_foot_orientation_error_deg": float(np.rad2deg(max(orientation_errors, default=0.0))),
        "joint_limit_violations": joint_limit_violations,
        "velocity_limit_violations": int(np.count_nonzero(np.abs(velocity) > velocity_limit + 1e-5)),
        "max_joint_velocity_rad_s": float(np.max(np.abs(velocity), initial=0.0)),
        "max_joint_acceleration_rad_s2": float(np.max(np.abs(acceleration), initial=0.0)),
        "max_joint_jerk_rad_s3": float(np.max(np.abs(jerk), initial=0.0)),
        "qp_failures": int(sum(frame["qp_failures"] for frame in diagnostics)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--terrain", required=True, type=Path)
    parser.add_argument("--joint_map", type=Path, default=DEFAULT_JOINT_MAP)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--robot", default="ne01_desktop_assets_wholebody_omni_gmr_v2")
    parser.add_argument("--tgt_fps", type=float, default=50.0)
    parser.add_argument("--save_path", required=True, type=Path)
    parser.add_argument("--debug_dir", type=Path, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--velocity_limit", type=float, default=3.0 * np.pi)
    parser.add_argument("--solver", choices=("daqp", "proxqp"), default="daqp")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    joint_map = load_joint_map(args.joint_map)
    positions, source_names, source_fps = load_holosoma_positions(args.motion, joint_map)
    positions = resample_positions(positions, source_fps, args.tgt_fps)
    if args.max_frames is not None:
        positions = positions[:args.max_frames]
    if len(positions) == 0:
        raise ValueError("Motion contains no frames after resampling")
    scene_transform, human_height = _scene_transform(args.config, config)
    source_terrain = TerrainField.from_file(args.terrain)
    transformed_positions = scene_transform.transform_points(positions)
    solver_frames, orientation_valid = build_solver_frames(transformed_positions, source_names, joint_map)
    source_frames = named_source_frames(positions, source_names)
    contact_config = dict(config["terrain"])
    contact_config["contact_blend_frames"] = config["terrain_contact_tasks"]["contact_blend_frames"]
    contact_schedule = build_terrain_contact_schedule(
        source_frames, source_terrain, scene_transform, joint_map, args.tgt_fps, contact_config
    )
    retargeter = WholeBodyTerrainOmniGMRV2(
        src_human="smplx",
        tgt_robot=args.robot,
        actual_human_height=human_height,
        solver=args.solver,
        use_velocity_limit=True,
        velocity_limit=args.velocity_limit,
        motion_fps=args.tgt_fps,
        graph_config_path=None,
        verbose=True,
        terrain_spec=source_terrain,
        joint_map=joint_map,
        scene_transform_config=scene_transform,
        terrain_config_path=args.config,
    )
    qpos = np.asarray([
        retargeter.retarget(frame, contact_frame=contact).copy()
        for frame, contact in zip(solver_frames, contact_schedule)
    ])
    metrics = summarize(qpos, retargeter.terrain_diagnostics, args.tgt_fps, retargeter.model, args.velocity_limit)
    transformed_terrain = retargeter.terrain.to_spec()
    surface_ids = np.asarray([
        [frame["contacts"][name]["surface_id"] for name in sorted(frame["contacts"])]
        for frame in contact_schedule
    ])
    payload = {
        "fps": float(args.tgt_fps),
        "root_pos": qpos[:, :3],
        "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": qpos[:, 7:],
        "qpos": qpos,
        "scene_transform": scene_transform.to_dict(),
        "terrain_primitives": transformed_terrain,
        "terrain_surface_ids": surface_ids,
        "contact_schedule": contact_schedule,
        "contact_metrics": metrics,
        "terrain_diagnostics": retargeter.terrain_diagnostics,
        "joint_names": _joint_names(retargeter.model),
        "body_names": _body_names(retargeter.model),
        "orientation_valid": orientation_valid,
        "source_motion": str(args.motion),
        "source_terrain": str(args.terrain),
        "robot_xml": retargeter.xml_file,
    }
    pkl_path = args.save_path if args.save_path.suffix.lower() == ".pkl" else args.save_path.with_suffix(".pkl")
    npz_path = args.save_path if args.save_path.suffix.lower() == ".npz" else args.save_path.with_suffix(".npz")
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as stream:
        pickle.dump(payload, stream)

    from convert_gmr_pkl_to_holosoma_npz import convert_file

    convert_file(pkl_path, npz_path, retargeter.model)
    debug_dir = args.debug_dir or pkl_path.parent / f"{pkl_path.stem}_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "summary.json").write_text(json.dumps(_jsonable(metrics), indent=2))
    (debug_dir / "frames.json").write_text(json.dumps(_jsonable(retargeter.terrain_diagnostics), indent=2))
    (debug_dir / "contact_schedule.json").write_text(json.dumps(_jsonable(contact_schedule), indent=2))
    (debug_dir / "scene.json").write_text(json.dumps({
        "scene_transform": scene_transform.to_dict(),
        "terrain": transformed_terrain,
        "source_joint_names": source_names,
        "joint_map": joint_map,
    }, indent=2))
    print(f"Saved terrain retargeting PKL: {pkl_path}")
    print(f"Saved HoloSoMo NPZ: {npz_path}")
    print(f"Saved diagnostics: {debug_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
