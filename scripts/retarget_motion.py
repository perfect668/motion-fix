"""Unified motion-format detector and WholeBody retargeting entry point."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import pickle

import numpy as np

from general_motion_retargeting.holosoma_input import (
    build_solver_frames,
    named_source_frames,
)
from general_motion_retargeting.motion_adapters import (
    CanonicalMotion,
    load_canonical_motion,
)
from general_motion_retargeting.terrain_contact_utils import build_terrain_contact_schedule
from general_motion_retargeting.terrain_geometry import SceneTransform, TerrainField
from general_motion_retargeting.wholebody_omni_gmr_v4 import WholeBodyOmniGMRV4
from general_motion_retargeting.wholebody_omni_gmr_v3 import sample_terrain_surface_pool


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "general_motion_retargeting/ik_configs/holosoma_to_ne01_wholebody_omni_gmr_v4.json"
DEFAULT_JOINT_MAP = ROOT / "general_motion_retargeting/joint_maps/holosoma_53.json"


def _load_config(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if not raw.get("extends"):
        return raw
    parent = _load_config(path.parent / raw["extends"])
    result = copy.deepcopy(parent)
    for key, value in raw.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    return result


def _scene_transform(config: dict) -> SceneTransform:
    scene = config.get("scene", {})
    scale = float(scene.get("robot_height", 1.316)) / float(scene.get("default_human_height", 1.78))
    return SceneTransform(
        np.asarray(scene.get("rotation", np.eye(3)), dtype=float),
        scale,
        np.asarray(scene.get("translation", [0.0, 0.0, 0.0]), dtype=float),
    )


def _solver_inputs(motion: CanonicalMotion, transform: SceneTransform, config: dict, joint_map: Path | None):
    transformed = transform.transform_points(motion.positions)
    if motion.source_format == "holosoma_global_positions" and joint_map is not None:
        if joint_map is None:
            raise ValueError("HoloSoMo input requires --joint_map")
        from general_motion_retargeting.holosoma_input import load_joint_map

        mapping = load_joint_map(joint_map)
        solver_frames, orientation_valid = build_solver_frames(transformed, motion.joint_names, mapping)
        source_frames = named_source_frames(transformed, motion.joint_names)
        for index, frame in enumerate(solver_frames):
            for solver_name, source_name in mapping["solver_joint_mapping"].items():
                source_frames[index][solver_name] = transformed[index, motion.joint_names.index(source_name)].copy()
            source_frames[index]["pelvis"] = frame["pelvis"][0].copy()
        return source_frames, solver_frames, orientation_valid

    # Blender FBX export carries bone rotations in the FBX author's local
    # basis, which is not guaranteed to match the NE01/GMR basis.  Rebuild the
    # stable pelvis/chest/sole frames from landmarks exactly as for HoloSoMo.
    # This preserves the measured positions while avoiding a root quaternion
    # that can mirror or twist both legs.
    if motion.source_format.startswith("fbx_binary"):
        from general_motion_retargeting.holosoma_input import build_solver_frames as build_frames_from_landmarks, load_joint_map

        mapping_path = joint_map or DEFAULT_JOINT_MAP
        mapping = load_joint_map(mapping_path)
        required = set(mapping["solver_joint_mapping"].values())
        available = set(motion.joint_names)
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"FBX canonical motion is missing required landmarks: {missing}")
        solver_frames, orientation_info = build_frames_from_landmarks(transformed, motion.joint_names, mapping)
        source_frames = named_source_frames(transformed, motion.joint_names)
        for index, frame in enumerate(solver_frames):
            aliases = {
                "Hips": "pelvis", "Spine1": "spine3", "Spine": "spine3",
                "LeftUpLeg": "left_hip", "RightUpLeg": "right_hip",
                "LeftLeg": "left_knee", "RightLeg": "right_knee",
                "LeftFoot": "left_foot", "RightFoot": "right_foot",
                "LeftToeBase": "left_toe", "RightToeBase": "right_toe",
                "LeftArm": "left_shoulder", "RightArm": "right_shoulder",
                "LeftForeArm": "left_elbow", "RightForeArm": "right_elbow",
                "LeftHandMiddle3": "left_wrist", "RightHandMiddle3": "right_wrist",
            }
            for legacy, canonical in aliases.items():
                if legacy in source_frames[index]:
                    source_frames[index][canonical] = source_frames[index][legacy].copy()
            source_frames[index]["pelvis"] = frame["pelvis"][0].copy()
        return source_frames, solver_frames, orientation_info

    # Position/rotation canonical names are adapted to the stable semantic
    # names expected by V4.  Both SMPL-X FK names and BVH names are accepted.
    if {"Hips", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg"} <= set(motion.joint_names):
        source_labels = {
            "pelvis": "Hips", "left_hip": "LeftUpLeg", "right_hip": "RightUpLeg",
            "spine3": "Spine1", "left_knee": "LeftLeg", "right_knee": "RightLeg",
            "left_foot": "LeftFoot", "right_foot": "RightFoot", "left_toe": "LeftToeBase", "right_toe": "RightToeBase",
            "left_shoulder": "LeftArm", "right_shoulder": "RightArm", "left_elbow": "LeftForeArm", "right_elbow": "RightForeArm",
            "left_wrist": "LeftHandMiddle3" if "LeftHandMiddle3" in motion.joint_names else "LeftHand",
            "right_wrist": "RightHandMiddle3" if "RightHandMiddle3" in motion.joint_names else "RightHand",
        }
        index = {name: i for i, name in enumerate(motion.joint_names)}
        required = sorted(set(source_labels.values()) - set(index))
        if required:
            raise ValueError(f"Canonical BVH motion is missing required joints: {required}")
        solver_frames, source_frames = [], []
        target_names = list(source_labels)
        for frame_index in range(motion.frame_count):
            source = {source_labels[name]: transformed[frame_index, index[source_labels[name]]].copy() for name in target_names}
            # V4 semantic configs use the detailed hand marker names.  BVH
            # files with only wrist joints receive an explicit alias.
            if "LeftHandMiddle3" not in source and "LeftHand" in source:
                source["LeftHandMiddle3"] = source["LeftHand"].copy()
            if "RightHandMiddle3" not in source and "RightHand" in source:
                source["RightHandMiddle3"] = source["RightHand"].copy()
            source["pelvis"] = source["Hips"].copy()
            # V4 consumes canonical names; retain BVH labels above only for
            # compatibility with the legacy contact utilities.
            source.update({name: source[source_name].copy() for name, source_name in source_labels.items()})
            source_frames.append(source)
            solver_frame = {}
            identity = np.array([1.0, 0.0, 0.0, 0.0])
            for name in target_names:
                source_name = source_labels[name]
                quat = (motion.orientations[frame_index, index[source_name]].copy()
                        if motion.orientations is not None else identity.copy())
                solver_frame[name] = (source[source_name], quat)
            solver_frames.append(solver_frame)
        return source_frames, solver_frames, bool(motion.orientation_valid)

    # SMPL-X canonical names are already semantic FK names.  Adapt them to the
    # stable semantic names expected by the V4 configuration and reuse the
    # recorded FK orientations instead of inventing identity quaternions.
    aliases = {
        "pelvis": "pelvis", "left_hip": "left_hip", "right_hip": "right_hip",
        "spine3": "spine3", "left_knee": "left_knee", "right_knee": "right_knee",
        "left_foot": "left_foot", "right_foot": "right_foot", "left_toe": "left_toe", "right_toe": "right_toe",
        "left_shoulder": "left_shoulder", "right_shoulder": "right_shoulder",
        "left_elbow": "left_elbow", "right_elbow": "right_elbow",
        "left_wrist": "left_wrist", "right_wrist": "right_wrist",
    }
    index = {name: i for i, name in enumerate(motion.joint_names)}
    required = sorted((set(aliases.values()) - {"left_toe", "right_toe"}) - set(index))
    if required:
        raise ValueError(f"Canonical SMPL-X motion is missing required joints: {required}")
    solver_frames = []
    source_frames = []
    source_labels = {
        "pelvis": "pelvis", "left_hip": "LeftUpLeg", "right_hip": "RightUpLeg",
        "spine3": "Spine1", "left_knee": "LeftLeg", "right_knee": "RightLeg",
        "left_foot": "LeftFoot", "right_foot": "RightFoot", "left_toe": "LeftToeBase", "right_toe": "RightToeBase",
        "left_shoulder": "LeftArm", "right_shoulder": "RightArm", "left_elbow": "LeftForeArm", "right_elbow": "RightForeArm",
        "left_wrist": "LeftHandMiddle3", "right_wrist": "RightHandMiddle3",
    }
    for frame_index in range(motion.frame_count):
        semantic_frame = {}
        for name in aliases.values():
            if name in index:
                semantic_frame[name] = transformed[frame_index, index[name]].copy()
        for side in ("left", "right"):
            toe_name = f"{side}_toe"
            if toe_name not in semantic_frame:
                foot = semantic_frame[f"{side}_foot"]
                ankle = semantic_frame.get(f"{side}_ankle", semantic_frame.get(f"{side}_knee"))
                direction = foot - ankle
                direction /= max(float(np.linalg.norm(direction)), 1e-12)
                semantic_frame[toe_name] = foot + 0.16 * direction
                index.setdefault(toe_name, index.get(f"{side}_foot", 0))
        frame = {source_labels[name]: point for name, point in semantic_frame.items()}
        # Keep canonical semantic names for contact inference while retaining
        # the legacy aliases consumed by inherited interaction configuration.
        frame.update({name: point.copy() for name, point in semantic_frame.items()})
        frame["pelvis"] = semantic_frame["pelvis"].copy()
        source_frames.append(frame)
        oriented = {}
        for name, point in semantic_frame.items():
            orientation_index = index.get(name, index.get(f"{name}"))
            oriented[name] = (point, motion.orientations[frame_index, orientation_index].copy())
        solver_frames.append(oriented)
    return source_frames, solver_frames, bool(motion.orientation_valid)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _contact_frames(motion: CanonicalMotion) -> list[dict[str, np.ndarray]]:
    """Return one stable semantic frame per source frame for contacts."""
    frames = motion.canonical_named_positions()
    for frame in frames:
        for side in ("left", "right"):
            toe = f"{side}_toe"
            if toe in frame:
                continue
            foot = frame.get(f"{side}_foot")
            ankle = frame.get(f"{side}_ankle")
            if foot is None or ankle is None:
                continue
            direction = np.asarray(foot) - np.asarray(ankle)
            norm = float(np.linalg.norm(direction))
            if norm > 1e-9:
                frame[toe] = np.asarray(foot) + 0.16 * direction / norm
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--robot", default="ne01")
    parser.add_argument("--version", choices=("v4",), default="v4")
    parser.add_argument("--joint_map", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--terrain", type=Path, default=None)
    parser.add_argument("--body_models", type=Path, default=ROOT / "assets/body_models")
    parser.add_argument("--source_fps", type=float, default=120.0)
    parser.add_argument("--tgt_fps", type=float, default=50.0)
    parser.add_argument("--save_path", required=True, type=Path)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--solver", choices=("daqp", "proxqp"), default="daqp")
    parser.add_argument("--checkpoint_every", type=int, default=0,
                        help="Write a resumable partial checkpoint every N frames")
    parser.add_argument("--resume", action="store_true", help="Resume from <save_path>.partial.pkl")
    args = parser.parse_args()
    if args.robot.lower() != "ne01":
        raise ValueError("The current unified WholeBody V4 entry is configured for NE01")

    motion = load_canonical_motion(
        args.motion,
        joint_map=args.joint_map or (DEFAULT_JOINT_MAP if args.motion.suffix.lower() in {".npy", ".npz"} else None),
        body_models=args.body_models,
        target_fps=args.tgt_fps,
        default_holosoma_fps=args.source_fps,
    )
    if args.max_frames is not None:
        motion.positions = motion.positions[: args.max_frames]
        if motion.orientations is not None:
            motion.orientations = motion.orientations[: args.max_frames]
    config = _load_config(args.config)
    transform = _scene_transform(config)
    source_terrain = TerrainField.from_file(args.terrain) if args.terrain else TerrainField()
    source_terrain.support_normal_min_z = float(config.get("terrain_contact", {}).get("support_normal_min_z", 0.6))
    terrain = source_terrain.transform(transform)
    source_frames, solver_frames, orientation_valid = _solver_inputs(motion, transform, config, args.joint_map)
    contact_cfg = config.get("terrain_contact", {})
    # Contact inference owns the source-to-solver transform and must receive
    # raw canonical points.  Passing already transformed solver points here
    # applies the similarity transform twice for HoloSoMo inputs.
    contacts = build_terrain_contact_schedule(
        _contact_frames(motion), source_terrain, transform,
        {}, args.tgt_fps, contact_cfg,
    )
    points = np.asarray(motion.positions, dtype=float).reshape(-1, 3)
    pool = sample_terrain_surface_pool(terrain, transform.transform_points(motion.positions), **config["interaction_graph"]["terrain_sampling"])

    # V4's scene backend requires the same terrain mesh to be compiled into a
    # combined robot+scene MuJoCo model.  Build it automatically when a mesh
    # terrain is supplied; the analytic TerrainField remains the query backend.
    effective_config = args.config
    temporary_config = None
    if not args.terrain and motion.scene.get("object_path"):
        raise ValueError(
            "This unified V4 entry detected scene metadata in the motion but no --terrain was supplied. "
            "Pass a supported terrain mesh, or use the GRAIL V4 scene entry for USD assets."
        )
    if not args.terrain and config.get("scene_collision", {}).get("backend") in {"mujoco", "hybrid"}:
        # A floor-only motion has no scene body to compile.  Keep V4's
        # analytic floor collision active without pretending a scene mesh exists.
        effective = copy.deepcopy(config)
        robot_xml = Path(effective["robot_xml"])
        if not robot_xml.is_absolute():
            effective["robot_xml"] = str((ROOT / robot_xml).resolve())
        effective["scene_collision"] = {**effective.get("scene_collision", {}), "backend": "analytic"}
        temporary_config = args.save_path.parent / f".{args.save_path.stem}_effective_config.json"
        temporary_config.parent.mkdir(parents=True, exist_ok=True)
        temporary_config.write_text(json.dumps(effective, indent=2))
        effective_config = temporary_config
    if args.terrain and config.get("scene_collision", {}).get("backend") in {"mujoco", "hybrid"}:
        from general_motion_retargeting.scene_asset_loader import load_scene_asset
        from general_motion_retargeting.scene_mujoco import build_scene_model

        robot_xml = Path(config["robot_xml"])
        if not robot_xml.is_absolute():
            robot_xml = (ROOT / robot_xml).resolve()
        scene_mesh = load_scene_asset(args.terrain, {"object_id": args.terrain.stem})
        pose = np.eye(4)
        pose[:3, :3] = transform.scale * transform.rotation
        pose[:3, 3] = transform.translation
        scene_mesh.object_pose = pose
        output_dir = args.save_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_xml = output_dir / f".{args.save_path.stem}_combined.xml"
        build_scene_model(robot_xml, scene_mesh, combined_xml, cache_root=ROOT / ".cache" / "scene_collision")
        effective = copy.deepcopy(config)
        effective["robot_xml"] = str(combined_xml)
        temporary_config = output_dir / f".{args.save_path.stem}_effective_config.json"
        temporary_config.write_text(json.dumps(effective, indent=2))
        effective_config = temporary_config
    try:
        retargeter = WholeBodyOmniGMRV4(effective_config, terrain, pool, fps=args.tgt_fps, solver=args.solver)
        checkpoint_path = args.save_path.with_suffix(".partial.pkl") if args.checkpoint_every > 0 or args.resume else None
        qpos = retargeter.retarget_canonical(
            motion, solver_frames, contacts, source_frames=source_frames,
            checkpoint_path=checkpoint_path,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
        )
    finally:
        if temporary_config is not None:
            temporary_config.unlink(missing_ok=True)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fps": float(args.tgt_fps), "qpos": qpos,
        "root_pos": qpos[:, :3], "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]], "dof_pos": qpos[:, 7:],
        "algorithm": "wholebody_omni_gmr_v4", "source_format": motion.source_format,
        "source_motion": motion.source_path, "source_joint_names": motion.joint_names,
        "canonical_joint_names": sorted(motion.canonical_named_positions()[0].keys()) if motion.frame_count else [],
        "canonical_provenance": motion.metadata.get("canonical_provenance", []),
        "source_to_canonical": motion.source_to_canonical,
        "human_height": motion.human_height,
        "orientation_valid_mask": motion.orientation_valid_mask,
        "orientation_valid": orientation_valid, "scene": motion.scene,
        "scene_transform": transform.to_dict(), "terrain_primitives": terrain.to_spec(),
        "contact_schedule": contacts, "terrain_diagnostics": retargeter.diagnostics,
        "robot_xml": str(retargeter.robot_xml),
    }
    with args.save_path.with_suffix(".pkl").open("wb") as stream:
        pickle.dump(payload, stream)
    print(f"Detected format: {motion.source_format}")
    print(f"Saved V4 PKL: {args.save_path.with_suffix('.pkl')}")
    print(f"Frames: {len(qpos)}; FPS: {args.tgt_fps:g}; orientation_valid: {orientation_valid}")


if __name__ == "__main__":
    main()
