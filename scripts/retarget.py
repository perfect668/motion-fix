"""WholeBody V5 task-level entry point.

Examples::

    python scripts/retarget.py --job job.json
    python scripts/retarget.py --motion walk.bvh --robot ne01 --output walk_v5.pkl

The resolver decides whether the task is motion-only, paired-scene or scene
replacement before any adapter or solver code runs.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import copy

import numpy as np

from general_motion_retargeting.contact import build_contact_plan
from general_motion_retargeting.core.schemas import FloorPolicy, RetargetTask, RobotModelSpec
from general_motion_retargeting.input import BundleResolver
from general_motion_retargeting.motion_adapters import load_canonical_motion
from general_motion_retargeting.scene import load_scene_model, terrain_from_scene
from general_motion_retargeting.terrain_geometry import SceneTransform
from general_motion_retargeting.wholebody_omni_gmr_v5 import WholeBodyRetargetSolver
from general_motion_retargeting.validation import validate_result

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "general_motion_retargeting/ik_configs/smplx_to_ne01_wholebody_omni_gmr_v5.json"
DEFAULT_JOINT_MAP = ROOT / "general_motion_retargeting/joint_maps/holosoma_53.json"


def _pool(terrain, points):
    reference = np.asarray(points, dtype=float).reshape((-1, 3))
    candidates=[]
    if hasattr(terrain, "points") and len(terrain.points):
        candidates.append(np.asarray(terrain.points, dtype=float))
    elif hasattr(terrain, "centroids") and len(terrain.centroids):
        # MeshSceneField exposes triangle centroids as the deterministic
        # interaction pool.  Omitting them silently turns a paired GRAIL
        # task into motion-only Laplacian tracking even though MuJoCo still
        # contains the chair collision geoms.
        candidates.append(np.asarray(terrain.centroids, dtype=float))
    if terrain.floor_z is not None:
        lower=reference[:,:2].min(axis=0)-.5; upper=reference[:,:2].max(axis=0)+.5
        side=12; x,y=np.meshgrid(np.linspace(lower[0],upper[0],side),np.linspace(lower[1],upper[1],side),indexing="ij")
        candidates.append(np.c_[x.ravel(),y.ravel(),np.full(x.size,terrain.floor_z)])
    for box in terrain.boxes:
        for axis in range(3):
            other=[i for i in range(3) if i != axis]
            grid=np.linspace(-1.,1.,8); u,v=np.meshgrid(grid,grid,indexing="ij"); local=np.zeros((u.size,3)); local[:,axis]=box.half_extents[axis]
            local[:,other[0]]=u.ravel()*box.half_extents[other[0]]; local[:,other[1]]=v.ravel()*box.half_extents[other[1]]
            candidates.append(box.center+local@box.rotation.T)
    return np.concatenate(candidates) if candidates else np.zeros((1,3))


def _config(path: Path):
    raw=json.loads(path.read_text())
    if raw.get("extends"):
        parent=(path.parent/raw["extends"]).resolve(); base=_config(parent)
        for key,value in raw.items():
            if key != "extends":
                if isinstance(value,dict) and isinstance(base.get(key),dict):
                    merged=_deep_merge(base[key],value); base[key]=merged
                else: base[key]=value
        return base
    return raw


def _deep_merge(base, override):
    merged=dict(base)
    for key,value in override.items():
        if isinstance(value,dict) and isinstance(merged.get(key),dict):
            merged[key]=_deep_merge(merged[key],value)
        else:
            merged[key]=value
    return merged


def _resolve_scene_transform(config: dict, motion) -> SceneTransform:
    """Resolve the one source-world -> NE01 solver transform.

    V5 never scales semantic joints independently.  A paired GRAIL scene and
    its human trajectory receive this same transform, which keeps chair/stair
    geometry, source contacts and robot targets in one frame.
    """
    scene_cfg = config.get("scene", {})
    source_height = float(
        motion.human_height
        if motion.human_height is not None
        else scene_cfg.get("default_human_height", 1.78)
    )
    if source_height <= 1e-8:
        raise ValueError(f"Invalid source human height: {source_height}")
    explicit = scene_cfg.get("scene_scale")
    scale = float(explicit) if explicit is not None else (
        float(scene_cfg.get("robot_height", 1.316)) / source_height
    )
    rotation = np.asarray(scene_cfg.get("rotation", np.eye(3)), dtype=float).reshape(3, 3)
    translation = np.asarray(scene_cfg.get("translation", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.0:
        raise ValueError("V5 scene rotation must be a proper orthonormal matrix")
    return SceneTransform(rotation, scale, translation)


def _transform_pose(pose: np.ndarray, transform: SceneTransform) -> np.ndarray:
    pose = np.asarray(pose, dtype=float).reshape(4, 4)
    result = np.eye(4)
    result[:3, :3] = transform.scale * transform.rotation @ pose[:3, :3]
    result[:3, 3] = transform.transform_points(pose[:3, 3])
    return result


def _estimate_auto_floor(motion, transform: SceneTransform) -> float:
    """Estimate only the analytic floor offset when no floor is declared.

    This is not used as a universal scene collision substitute: mesh/box
    surfaces remain authoritative.  It simply prevents the default plane from
    being placed 10--15 cm above a paired GRAIL source world whose floor is
    encoded implicitly in the mocap coordinates.
    """
    frames = motion.canonical_named_positions()
    samples = []
    for frame in frames:
        for side in ("left", "right"):
            for name in (f"{side}_heel", f"{side}_toe"):
                if name in frame:
                    samples.append(transform.transform_points(frame[name]))
    if not samples:
        return 0.0
    points = np.asarray(samples, dtype=float)
    # The lower tail is the only robust estimate available without an explicit
    # floor asset; use a clipped low quantile rather than the absolute minimum
    # so a single erroneous toe marker cannot move the entire scene.
    return float(np.quantile(points[:, 2], 0.05))


def _interaction_anchors(plan, terrain, motion_positions: np.ndarray) -> np.ndarray:
    """Build a fixed semantic environment pool from contact episodes.

    These anchors are source-scene facts, not robot-state nearest points.  In
    a paired interaction this preserves the relevant object relationship for
    the complete episode; plain locomotion falls back to terrain samples.
    """
    anchors = [np.asarray(episode.source_anchor, dtype=float) for episode in plan.episodes]
    if anchors:
        values = np.asarray(anchors, dtype=float).reshape((-1, 3))
        # Preserve deterministic order while avoiding repeated anchors from a
        # static heel/toe pair.
        keys = np.round(values, decimals=6)
        _, indices = np.unique(keys, axis=0, return_index=True)
        return values[np.sort(indices)]
    return _pool(terrain, motion_positions)


def run(args):
    job={}
    if args.job:
        job=json.loads(args.job.read_text(encoding="utf-8"))
        motion_arg=job.get("motion",{}).get("path")
        if motion_arg and not Path(motion_arg).is_absolute(): motion_arg=str((args.job.parent/ motion_arg).resolve())
        motion=Path(motion_arg)
        robot=str(job.get("robot",{}).get("name",args.robot)); output=Path(job.get("output",{}).get("path",args.output or "work/v5_result.pkl"));
        if not output.is_absolute(): output=(args.job.parent/output).resolve()
        config=Path(job.get("solver",{}).get("config",args.config));
        if not config.is_absolute(): config=(args.job.parent/config).resolve()
        manifest=args.job
    else:
        motion=args.motion; robot=args.robot; output=args.output; config=args.config; manifest=args.manifest
    if motion is None or output is None: raise ValueError("--motion/--output or --job is required")
    if robot.lower() != "ne01": raise ValueError("WholeBody V5 currently supports robot ne01")
    bundle=BundleResolver().resolve(motion,manifest=manifest)
    cfg=_config(config)
    map_path=args.joint_map or (DEFAULT_JOINT_MAP if bundle.motion.format.startswith("holosoma") else None)
    canonical=load_canonical_motion(bundle.motion.path,joint_map=map_path,body_models=args.body_models,target_fps=args.tgt_fps)
    if args.max_frames is not None:
        canonical.positions = canonical.positions[:args.max_frames]
        if canonical.orientations is not None:
            canonical.orientations = canonical.orientations[:args.max_frames]
        if canonical.orientation_valid_mask is not None:
            canonical.orientation_valid_mask = canonical.orientation_valid_mask[:args.max_frames]
    relation=job.get("scene_relation") if job else None
    floor_policy=FloorPolicy(str(job.get("scene",{}).get("floor_policy","auto") if job else "auto"))
    # A GRAIL bundle may carry an exact obj_R/obj_t/obj_scale record.  The
    # scene reference remains explicit, but its pose is supplied from the
    # bundle metadata so visual, contact and collision coordinates agree.
    scene_ref = bundle.target_scene
    transform = _resolve_scene_transform(cfg, canonical)
    if scene_ref is not None and canonical.scene:
        obj = canonical.scene.get("obj_data", {})
        if isinstance(obj, dict) and (obj.get("obj_R") is not None or obj.get("obj_t") is not None):
            pose = np.eye(4)
            rotation = np.asarray(obj.get("obj_R", np.eye(3)), dtype=float)
            if rotation.ndim == 3: rotation = rotation[0]
            scale = np.asarray(obj.get("obj_scale", 1.0), dtype=float).reshape(-1)
            pose[:3,:3] = rotation.reshape(3,3) * (float(scale[0]) if len(scale) == 1 else 1.0)
            if len(scale) == 3: pose[:3,:3] = rotation.reshape(3,3) @ np.diag(scale)
            translation = np.asarray(obj.get("obj_t", np.zeros(3)), dtype=float).reshape(-1)
            if len(translation) >= 3: pose[:3,3] = translation[:3]
            from general_motion_retargeting.core.schemas import SceneReference
            pose = _transform_pose(pose, transform)
            scene_ref = SceneReference(path=scene_ref.path, format=scene_ref.format, asset_id=scene_ref.asset_id, pose=pose, metadata=scene_ref.metadata)
        elif scene_ref.pose is not None:
            scene_ref = SceneReference(
                path=scene_ref.path,
                format=scene_ref.format,
                asset_id=scene_ref.asset_id,
                pose=_transform_pose(scene_ref.pose, transform),
                metadata=scene_ref.metadata,
            )
    configured_floor = job.get("scene", {}).get("floor_height") if job else None
    floor_height = (
        float(configured_floor)
        if configured_floor is not None
        else (_estimate_auto_floor(canonical, transform) if floor_policy == FloorPolicy.AUTO else None)
    )
    scene=load_scene_model(scene_ref,floor_policy=floor_policy,floor_height=floor_height)
    terrain=terrain_from_scene(scene)
    from general_motion_retargeting.input.solver_frames import build_solver_inputs
    # CanonicalMotion is in source-world coordinates; apply the single scene
    # transform exactly once before constructing solver targets.
    canonical.positions = transform.transform_points(canonical.positions)
    source_frames,solver_frames,_=build_solver_inputs(canonical)
    # Positions are already in solver coordinates.  Applying the transform a
    # second time here would scale/translate contact references twice and is a
    # direct source of bent legs and detached scene assets.
    plan=build_contact_plan(canonical,terrain,fps=args.tgt_fps,transform=None,config=cfg.get("terrain_contact",{}))
    xml=Path(cfg["robot_xml"]); xml=(config.parent.parent.parent/xml).resolve() if not xml.is_absolute() else xml.resolve()
    # Mesh scenes enter the same MuJoCo model used by the V5 solver.  The
    # visual source mesh, cached convex pieces and object pose are generated
    # from this one SceneReference; no motion-only fallback is permitted.
    effective_config = dict(cfg)
    if scene_ref is not None and scene_ref.path is not None and scene_ref.path.suffix.lower() in {".obj", ".usd", ".usda", ".usdc"}:
        from general_motion_retargeting.scene_asset_loader import load_scene_asset
        from general_motion_retargeting.scene_mujoco import build_scene_model
        scene_mesh = load_scene_asset(scene_ref.path, {"object_id": scene_ref.asset_id, "sample_count": int(cfg.get("scene_geometry", {}).get("scene_surface_points", 512))})
        scene_mesh.object_pose = scene.assets[0].pose
        combined = output.with_name(f".{output.stem}_v5_combined.xml")
        build_scene_model(xml, scene_mesh, combined, cache_root=ROOT / ".cache" / "scene_collision", return_info=True)
        effective_config["robot_xml"] = str(combined)
    effective_config_path = output.with_name(f".{output.stem}_v5_config.json")
    effective_config_path.parent.mkdir(parents=True, exist_ok=True)
    effective_config_path.write_text(json.dumps(effective_config, indent=2))
    robot_xml_for_task = Path(effective_config.get("robot_xml", str(xml))).expanduser()
    if not robot_xml_for_task.is_absolute():
        robot_xml_for_task = (ROOT / robot_xml_for_task).resolve()
    robot_spec=RobotModelSpec(robot,robot_xml_for_task,cfg.get("semantic_points",{}),cfg.get("joint_position_limits",{}),cfg.get("solver",{}))
    task=RetargetTask(bundle,scene,effective_config_path,robot_spec,effective_config)
    solver=WholeBodyRetargetSolver(
        task, terrain, _interaction_anchors(plan, terrain, canonical.positions),
        fps=args.tgt_fps, solver=args.solver,
    )
    result=solver.solve(canonical,solver_frames,list(plan.per_frame_states),source_frames)
    kinematics = solver.forward_kinematics(result.qpos)
    validation = validate_result(result)
    payload={"schema_version":1,"algorithm":"wholebody_omni_gmr_v5","status":validation["status"],"validation":validation,"failure":result.failure,"fps":args.tgt_fps,"qpos":result.qpos,"root_pos":result.qpos[:,:3],"root_rot":result.qpos[:,3:7][:,[1,2,3,0]],"dof_pos":result.qpos[:,7:],"joint_pos":result.qpos[:,7:],"joint_vel":kinematics["joint_vel"],"body_names":kinematics["body_names"],"body_pos_w":kinematics["body_pos_w"],"body_quat_w":kinematics["body_quat_w"],"body_lin_vel_w":kinematics["body_lin_vel_w"],"body_ang_vel_w":kinematics["body_ang_vel_w"],"source_motion":str(bundle.motion.path),"source_format":canonical.source_format,"joint_names":canonical.joint_names,"scene_relation":bundle.scene_relation.value,"scene_transform":transform.to_dict(),"contact_plan":{"episodes":[e.__dict__ for e in plan.episodes],"frames":list(plan.per_frame_states)},"diagnostics":result.diagnostics,"terrain":terrain.to_spec(),"robot_xml":str(solver.robot_xml),"scene":{"assets":[{"asset_id":a.asset_id,"path":str(a.path),"pose":a.pose.tolist(),"collision_enabled":a.collision_enabled} for a in scene.assets],"floor_height":scene.floor_height}}
    output=Path(output).expanduser().resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("wb") as stream: pickle.dump(payload,stream,protocol=pickle.HIGHEST_PROTOCOL)
    if result.status != "VALID":
        raise RuntimeError(result.failure or "WholeBody V5 failed; no complete motion exported")
    np.savez(output.with_suffix(".npz"), fps=np.asarray(args.tgt_fps), qpos=result.qpos,
             root_pos=result.qpos[:,:3], root_rot=result.qpos[:,3:7][:,[1,2,3,0]],
             dof_pos=result.qpos[:,7:], joint_pos=result.qpos[:,7:], joint_vel=kinematics["joint_vel"],
             body_pos_w=kinematics["body_pos_w"], body_quat_w=kinematics["body_quat_w"],
             body_lin_vel_w=kinematics["body_lin_vel_w"], body_ang_vel_w=kinematics["body_ang_vel_w"],
             joint_names=np.asarray(canonical.joint_names), body_names=np.asarray(kinematics["body_names"]),
             source_format=np.asarray(canonical.source_format), scene_relation=np.asarray(bundle.scene_relation.value))
    output.with_suffix(".diagnostics.json").write_text(json.dumps(result.diagnostics, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value), indent=2), encoding="utf-8")
    effective_config_path.unlink(missing_ok=True)
    print(f"WholeBody V5: {result.status}\nFormat: {canonical.source_format}\nFrames: {len(result.qpos)}\nSaved: {output}")
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--job",type=Path); parser.add_argument("--motion",type=Path); parser.add_argument("--robot",default="ne01"); parser.add_argument("--output",type=Path); parser.add_argument("--manifest",type=Path); parser.add_argument("--joint_map",type=Path); parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG); parser.add_argument("--body_models",type=Path,default=ROOT/"assets/body_models"); parser.add_argument("--tgt_fps",type=float,default=50.); parser.add_argument("--solver",choices=("daqp","proxqp"),default="daqp"); parser.add_argument("--max_frames",type=int,default=None); args=parser.parse_args(); run(args)


if __name__ == "__main__": main()
