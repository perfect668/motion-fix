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

from general_motion_retargeting.contact import (
    ContactBinder,
    RobotContactRealizer,
    build_contact_plan,
)
from general_motion_retargeting.core.schemas import (
    ContactEpisode,
    ContactPlan,
    FloorPolicy,
    RetargetJob,
    RetargetTask,
    RobotModelSpec,
    SceneReference,
    SceneRelation,
    PoseTrajectory,
)
from general_motion_retargeting.input import (
    ScopeAdmissionError,
    preflight_scope,
    resolver_for_motion,
    validate_loaded_scope,
)
from general_motion_retargeting.input.eligibility import file_sha256
from general_motion_retargeting.motion_adapters import load_canonical_motion
from general_motion_retargeting.scene import SceneAssembler, alignment_sanity
from general_motion_retargeting.terrain_geometry import SceneTransform
from general_motion_retargeting.wholebody_omni_gmr_v5 import WholeBodyRetargetSolver
from general_motion_retargeting.v5_pipeline import solve_and_validate
from general_motion_retargeting.morphology import (
    build_morphology_targets,
    map_semantic_frame,
    measure_robot_chain_lengths,
)
from general_motion_retargeting.export import export_result, jsonable

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


def _effective_config(base: dict, job: dict, args) -> dict:
    """Resolve the one configuration consumed by every V5 stage."""
    result = copy.deepcopy(base)
    if job:
        mappings = (
            ("transform", "scene"),
            ("morphology", "morphology"),
            ("contact", "terrain_contact"),
            ("solver", "solver"),
            ("validation", "validation"),
        )
        for source_key, target_key in mappings:
            override = job.get(source_key)
            if not isinstance(override, dict):
                continue
            if source_key == "solver":
                override = {key: value for key, value in override.items() if key != "config"}
            result[target_key] = _deep_merge(result.get(target_key, {}), override)
        scene_override = job.get("scene", {})
        if isinstance(scene_override, dict):
            policy_keys = {
                key: value for key, value in scene_override.items()
                if key in {"rotation", "translation", "scene_scale", "floor_inference"}
            }
            result["scene"] = _deep_merge(result.get("scene", {}), policy_keys)
    return result


def _scope_declaration(job: dict, args) -> dict:
    """Merge only explicit scope fields; argparse defaults never masquerade as policy."""
    scope = dict(job.get("scope", {}) if job else {})
    for key in ("task_family", "terrain_kind", "interaction_mode"):
        value = getattr(args, key, None)
        if value is not None:
            scope[key] = value
    return scope


def _raise_unless_admitted(decision) -> None:
    if not decision.admitted:
        raise ScopeAdmissionError(decision)


def _configured_robot_xml(config_path: Path, config: dict) -> Path:
    """Resolve the base robot model before any temporary scene MJCF exists."""
    xml = Path(config["robot_xml"]).expanduser()
    if xml.is_absolute():
        return xml.resolve()
    # A user/job may place a small overriding config outside the repository.
    # Resolve repository-owned robot assets against the project root first,
    # then retain config-relative lookup for self-contained external configs.
    for candidate in (ROOT / xml, config_path.parent / xml, config_path.parent.parent.parent / xml):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"V5 robot_xml {xml!s} was not found relative to {ROOT} or {config_path.parent}"
    )


def _resolve_scene_transform(config: dict, motion, relation: SceneRelation) -> SceneTransform:
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
    if explicit is not None:
        scale = float(explicit)
    elif relation == SceneRelation.MOTION_ONLY:
        # Motion-only morphology is applied to targets, never to a scene.
        # A paired source scene must remain in its metric dataset world.
        scale = 1.0
    else:
        scale = 1.0
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


def _contact_motion_in_query_frame(
    motion,
    source_positions: np.ndarray,
    transform: SceneTransform,
    relation: SceneRelation,
):
    """Return the human timeline in the same world as the contact terrain.

    ``SOURCE_TO_TARGET`` keeps source-world points because the caller passes
    ``contact_transform`` to the detector.  All other relations query a
    terrain already assembled in solver/world coordinates and therefore need
    the shared SceneTransform applied before contact inference.
    """
    result = copy.copy(motion)
    result.positions = (
        np.asarray(source_positions, dtype=float).copy()
        if relation == SceneRelation.SOURCE_TO_TARGET
        else transform.transform_points(source_positions)
    )
    return result


def _asset_scale_baked(scene_ref: SceneReference, canonical) -> bool:
    """Resolve one explicit scale policy for visual/query/collision geometry.

    Processed GRAIL ``mesh_data/model.obj`` exports historically contain the
    reconstruction scale in their vertices, while USD/object-local assets keep
    that scale in ``obj_scale``.  Prefer an authored declaration and only use
    the deterministic path convention as a fallback.  The result is copied
    into ``SceneAsset.metadata`` so every prepared representation consumes the
    same decision.
    """
    scene = getattr(canonical, "scene", {}) or {}
    obj = scene.get("obj_data", {}) if isinstance(scene, dict) else {}
    metadata = dict(getattr(scene_ref, "metadata", {}) or {})
    for value in (
        scene.get("asset_scale_baked") if isinstance(scene, dict) else None,
        obj.get("asset_scale_baked") if isinstance(obj, dict) else None,
        metadata.get("asset_scale_baked"),
    ):
        if value is not None:
            return bool(value)
    path = str(scene_ref.path).replace("\\", "/") if scene_ref.path is not None else ""
    return path.endswith("/mesh_data/model.obj")


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
            # Position-only formats may not carry surface heel/toe markers;
            # use the measured ankle/foot joint only to place a motion-only
            # clip on the analytic floor.  It is never emitted as a contact
            # landmark or used for desired-contact inference.
            if not any(name in frame for name in (f"{side}_heel", f"{side}_toe")):
                for name in (f"{side}_ankle", f"{side}_foot"):
                    if name in frame:
                        samples.append(transform.transform_points(frame[name]))
                        break
    if not samples:
        return 0.0
    points = np.asarray(samples, dtype=float)
    # The lower tail is the only robust estimate available without an explicit
    # floor asset; use a clipped low quantile rather than the absolute minimum
    # so a single erroneous toe marker cannot move the entire scene.
    return float(np.quantile(points[:, 2], 0.05))


def _infer_scene_floor(geometry, config: dict) -> tuple[float | None, dict]:
    """Infer an undeclared floor from the loaded scene geometry.

    GRAIL reconstructions commonly contain metric object poses but no floor
    field.  Using the human heel quantile in that case can put the analytic
    floor below a chair/stair asset (the SMPL-X heel landmark is not a floor
    annotation).  The scene mesh is the common authority for the visual,
    query and collision representations, so use its robust lower extent.  An
    explicit floor always bypasses this helper.
    """
    scene_cfg = config.get("scene", {}) if isinstance(config, dict) else {}
    policy = scene_cfg.get("floor_inference", {})
    if policy is False or (isinstance(policy, dict) and not policy.get("enabled", True)):
        return None, {"method": "disabled"}
    if geometry is None or not geometry.scene.assets:
        return None, {"method": "no_scene"}
    try:
        quantile = float(policy.get("quantile", 0.005)) if isinstance(policy, dict) else 0.005
        floor, assets = geometry.lower_extent(quantile)
        if floor is None:
            return None, {"method": "empty_scene"}
        return floor, {"method": "scene_geometry_lower_extent", "quantile": quantile, "assets": assets}
    except Exception as error:
        # Scene loading later raises the original, actionable error.  Floor
        # inference is only a fallback and must not hide that failure.
        return None, {"method": "scene_geometry_error", "error": str(error)}


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


def _pose_trajectory_from_obj(obj: dict, fps: float, transform: SceneTransform, target_timestamps=None) -> PoseTrajectory | None:
    """Preserve a complete GRAIL object trajectory when metadata contains it."""
    if not isinstance(obj, dict) or obj.get("obj_t") is None:
        return None
    translations = np.asarray(obj["obj_t"], dtype=float)
    rotations = np.asarray(obj.get("obj_R"), dtype=float) if obj.get("obj_R") is not None else None
    if translations.ndim != 2 or translations.shape[1] != 3:
        return None
    if rotations is not None and (rotations.ndim != 3 or rotations.shape[1:] != (3, 3)):
        rotations = None
    count = len(translations)
    source_timestamps = np.arange(count, dtype=float) / float(fps)
    timestamps = source_timestamps if target_timestamps is None else np.asarray(target_timestamps, dtype=float).reshape(-1)
    if len(timestamps) == 0:
        return None
    if len(timestamps) != count or not np.allclose(timestamps, source_timestamps):
        translations = np.asarray([
            np.interp(timestamps, source_timestamps, translations[:, axis]) for axis in range(3)
        ]).T
    translations = transform.transform_points(translations)
    if rotations is not None:
        rotations = np.einsum("ij,njk->nik", transform.rotation, rotations)
        # SceneTransform is uniform scale; pose trajectory stores rotations
        # separately so scale is not accidentally applied twice.
        from scipy.spatial.transform import Rotation
        q_source = Rotation.from_matrix(rotations)
        if len(timestamps) != count:
            from scipy.spatial.transform import Slerp
            # Dataset object tracks are occasionally shorter than the human
            # clip.  Translation interpolation already clamps at the end;
            # use the same zero-order hold for rotations instead of letting
            # SciPy reject out-of-range target timestamps.
            query_timestamps = np.clip(timestamps, source_timestamps[0], source_timestamps[-1])
            q = Slerp(source_timestamps, q_source)(query_timestamps).as_quat(scalar_first=True)
        else:
            q = q_source.as_quat(scalar_first=True)
    else:
        q = None
    return PoseTrajectory(timestamps, translations, q)


def _apply_morphology_inputs(source_frames, solver_frames, morphology):
    if len(source_frames) != len(solver_frames):
        raise ValueError(
            f"V5 morphology input timeline mismatch: {len(source_frames)} != {len(solver_frames)}"
        )
    if morphology.mode == "scene_preserving":
        return source_frames, solver_frames
    mapped_sources = []
    mapped_targets = []
    for source, targets in zip(source_frames, solver_frames):
        mapped = map_semantic_frame(source, morphology)
        mapped_sources.append(mapped)
        mapped_targets.append({
            name: (mapped.get(name, value[0]), value[1])
            for name, value in targets.items()
        })
    return mapped_sources, mapped_targets


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
    requested_format = (job.get("motion", {}).get("format", "auto") if job else args.motion_format)
    bundle=resolver_for_motion(motion, requested_format).resolve(
        motion,
        manifest=manifest,
        motion_format=requested_format,
    )
    cfg=_effective_config(_config(config), job, args)
    scope_decision = preflight_scope(
        bundle.motion.path,
        bundle.motion.format,
        _scope_declaration(job, args),
    )
    _raise_unless_admitted(scope_decision)
    retarget_job = RetargetJob(
        motion=bundle.motion,
        source_scene=bundle.source_scene,
        target_scene=bundle.target_scene,
        scene_relation=bundle.scene_relation,
        robot=robot,
        transform_policy=cfg.get("scene", {}),
        morphology_policy=cfg.get("morphology", {}),
        contact_policy=cfg.get("terrain_contact", {}),
        solver_policy=cfg.get("solver", {}),
        output_policy=(job.get("output", {}) if job else {}),
        task_family=scope_decision.task_family,
        terrain_kind=scope_decision.terrain_kind,
        interaction_mode=scope_decision.interaction_mode,
        scope_evidence=scope_decision.to_dict(),
    )
    map_path=args.joint_map
    if map_path is None and bundle.motion.format.startswith("holosoma"):
        map_path = DEFAULT_JOINT_MAP
    # BundleResolver has already made the format decision (and must honor an
    # explicit manifest declaration).  Pass that decision through so the
    # adapter registry, rather than extension heuristics, owns parsing.
    canonical=load_canonical_motion(bundle.motion.path,
        joint_map=map_path, body_models=args.body_models,
        target_fps=args.tgt_fps, motion_format=bundle.motion.format,
        bvh_format=args.bvh_format)
    scope_decision = validate_loaded_scope(scope_decision, canonical, bundle)
    _raise_unless_admitted(scope_decision)
    relation = bundle.scene_relation
    floor_policy=FloorPolicy(str(job.get("scene",{}).get("floor_policy","auto") if job else "auto"))
    # A GRAIL bundle may carry an exact obj_R/obj_t/obj_scale record.  The
    # scene reference remains explicit, but its pose is supplied from the
    # bundle metadata so visual, contact and collision coordinates agree.
    scene_ref = bundle.target_scene
    transform = _resolve_scene_transform(cfg, canonical, relation)
    if scene_ref is not None and canonical.scene:
        obj = canonical.scene.get("obj_data", {})
        if isinstance(obj, dict) and (obj.get("obj_R") is not None or obj.get("obj_t") is not None):
            pose = np.eye(4)
            rotation = np.asarray(obj.get("obj_R", np.eye(3)), dtype=float)
            if rotation.ndim == 3: rotation = rotation[0]
            scale = np.asarray(obj.get("obj_scale", 1.0), dtype=float).reshape(-1)
            asset_scale_baked = _asset_scale_baked(scene_ref, canonical)
            # ``obj_scale`` belongs either in the object-local vertices or in
            # the logical pose, never both.  Keep the authored/path-derived
            # choice explicit in SceneReference metadata for all downstream
            # geometry consumers.
            pose_scale = np.ones(3, dtype=float) if asset_scale_baked else (
                np.repeat(scale, 3) if len(scale) == 1 else scale
            )
            if len(pose_scale) != 3 or not np.isfinite(pose_scale).all() or np.any(pose_scale <= 0.0):
                raise ValueError("GRAIL obj_scale must contain one or three finite positive values")
            pose[:3,:3] = rotation.reshape(3,3) @ np.diag(pose_scale)
            translation = np.asarray(obj.get("obj_t", np.zeros(3)), dtype=float).reshape(-1)
            if len(translation) >= 3: pose[:3,3] = translation[:3]
            pose = _transform_pose(pose, transform)
            trajectory = _pose_trajectory_from_obj(obj, canonical.fps, transform, canonical.timestamps)
            metadata = dict(scene_ref.metadata)
            metadata.update({
                "asset_scale_baked": bool(asset_scale_baked),
                "asset_space": metadata.get("asset_space", "object_local"),
                "obj_scale_source": "vertices" if asset_scale_baked else "pose",
            })
            scene_ref = SceneReference(path=scene_ref.path, format=scene_ref.format, asset_id=scene_ref.asset_id, pose=pose, pose_trajectory=trajectory, metadata=metadata)
        elif scene_ref.pose is not None:
            scene_ref = SceneReference(
                path=scene_ref.path,
                format=scene_ref.format,
                asset_id=scene_ref.asset_id,
                pose=_transform_pose(scene_ref.pose, transform),
                pose_trajectory=scene_ref.pose_trajectory,
                metadata=scene_ref.metadata,
            )
    scene_sample_count = int(cfg.get("scene_geometry", {}).get("scene_surface_points", 512))
    scene_assembler = SceneAssembler()
    # Decode/normalize all object meshes before floor selection.  The final
    # assembly below reuses this exact object, so visual/query/collision and
    # lower-extent inference cannot drift through separate scale paths.
    prepared_scene = scene_assembler.prepare(scene_ref, sample_count=scene_sample_count)
    configured_floor = job.get("scene", {}).get("floor_height") if job else None
    floor_provenance = {"method": "explicit"} if configured_floor is not None else None
    if configured_floor is not None:
        floor_height = float(configured_floor)
    elif floor_policy != FloorPolicy.AUTO:
        floor_height = 0.0 if relation == SceneRelation.MOTION_ONLY else None
        floor_provenance = {"method": "policy", "floor_policy": floor_policy.value}
    elif relation == SceneRelation.MOTION_ONLY:
        floor_height = 0.0
        floor_provenance = {"method": "solver_world_support_calibration", "source": "canonical_heel_toe"}
    else:
        # For a paired scene, prefer the scene geometry's support baseline.
        # GRAIL's SMPL-X heel markers are body landmarks, not a guaranteed
        # world-floor annotation, and can sit below the generated asset feet.
        inferred_floor, floor_provenance = _infer_scene_floor(prepared_scene.geometry, cfg)
        if inferred_floor is None:
            floor_height = _estimate_auto_floor(canonical, transform)
            floor_provenance = {
                **(floor_provenance or {}),
                "fallback": "motion_support_quantile",
            }
        else:
            floor_height = inferred_floor
    # Motion-only clips are moved onto the analytic z=0 plane.  The plane is
    # never moved to follow the actor.
    if relation == SceneRelation.MOTION_ONLY and floor_policy == FloorPolicy.AUTO:
        floor_offset = _estimate_auto_floor(canonical, transform)
        translation = np.asarray(transform.translation, dtype=float).copy()
        translation[2] -= floor_offset
        transform = SceneTransform(transform.rotation, transform.scale, translation)
        floor_height = 0.0
        floor_provenance.update({
            "solver_world_offset": float(floor_offset),
            "applied_via_scene_transform": True,
        })
    scene_assembly = scene_assembler.with_floor(
        prepared_scene, floor_policy=floor_policy, floor_height=floor_height,
    )
    scene, terrain = scene_assembly.model, scene_assembly.terrain
    # A shared source scene is an input contract: human and assets already
    # occupy one dataset world.  Moving only the human to make noisy foot
    # landmarks meet a floor destroys the authored human-object relation.
    # Landmark-to-surface observation bias is handled by SourceContactDetector
    # without modifying CanonicalMotion or SceneModel.
    motion_scene_alignment = {
        "status": (
            "NOT_APPLICABLE"
            if relation == SceneRelation.MOTION_ONLY
            else "SHARED_SOURCE_WORLD_PRESERVED"
        ),
        "translation_source": [0.0, 0.0, 0.0],
        "translation_solver": [0.0, 0.0, 0.0],
    }
    # Keep a complete canonical copy for contact calibration.  A requested
    # output prefix must not change a source landmark offset model merely
    # because its minimum sample count occurs after that prefix.
    contact_motion_full = copy.copy(canonical)
    contact_motion_full.positions = canonical.positions.copy()
    contact_motion_full.timestamps = canonical.timestamps.copy()
    contact_motion_full.frame_ids = canonical.frame_ids.copy()
    if canonical.orientations is not None:
        contact_motion_full.orientations = canonical.orientations.copy()
    if canonical.orientation_valid_mask is not None:
        contact_motion_full.orientation_valid_mask = canonical.orientation_valid_mask.copy()
    labels_full = canonical.metadata.get("foot_contact_probs")
    if labels_full is not None:
        contact_motion_full.metadata = dict(canonical.metadata)
        contact_motion_full.metadata["foot_contact_probs"] = np.asarray(labels_full).copy()
    if args.max_frames is not None:
        frame_count = max(0, min(int(args.max_frames), canonical.frame_count))
        canonical.positions = canonical.positions[:frame_count]
        canonical.timestamps = canonical.timestamps[:frame_count]
        canonical.frame_ids = canonical.frame_ids[:frame_count]
        if canonical.orientations is not None:
            canonical.orientations = canonical.orientations[:frame_count]
        if canonical.orientation_valid_mask is not None:
            canonical.orientation_valid_mask = canonical.orientation_valid_mask[:frame_count]
        labels = canonical.metadata.get("foot_contact_probs")
        if labels is not None:
            labels = np.asarray(labels)
            if labels.ndim >= 1:
                canonical.metadata["foot_contact_probs"] = labels[:frame_count]
    from general_motion_retargeting.input.solver_frames import build_solver_inputs
    # CanonicalMotion is in source-world coordinates; apply the single scene
    # transform exactly once before constructing solver targets.
    source_positions_for_contacts = contact_motion_full.positions.copy()
    canonical.positions = transform.transform_points(canonical.positions)
    source_evidence_frames,solver_frames,_=build_solver_inputs(canonical)
    morphology_policy = (job.get("morphology", {}) if job else {}) or cfg.get("morphology", {})
    base_robot_xml = _configured_robot_xml(config, cfg)
    robot_chain_lengths, robot_provenance = measure_robot_chain_lengths(
        base_robot_xml, cfg.get("semantic_points", {})
    )
    morphology = build_morphology_targets(
        canonical,
        float(cfg.get("scene", {}).get("robot_height", 1.316)),
        morphology_policy,
        robot_chain_lengths=robot_chain_lengths,
        robot_provenance=robot_provenance,
    )
    robot_reference_frames, solver_frames = _apply_morphology_inputs(
        source_evidence_frames, solver_frames, morphology
    )
    # Positions are already in solver coordinates.  Applying the transform a
    # second time here would scale/translate contact references twice and is a
    # direct source of bent legs and detached scene assets.
    if relation == SceneRelation.TARGET_ONLY_WITH_EXPLICIT_BINDING:
        explicit = (job or {}).get("contacts")
        if not explicit:
            raise ValueError("target_only_with_explicit_binding requires an explicit contacts section")
        plan = ContactPlan(episodes=tuple(), per_frame_states=tuple(explicit), fps=args.tgt_fps, metadata={"explicit_binding": True})
        bound_schedule = ContactBinder().bind(
            plan, relation=relation, binding={"explicit": True}, target_terrain=terrain
        )
    else:
        contact_terrain = terrain
        if relation == SceneRelation.SOURCE_TO_TARGET:
            if not (job or {}).get("contact_binding"):
                raise ValueError("source_to_target requires explicit contact_binding correspondence")
            # Source contact detection must use the source asset pose, while
            # the target scene above uses the explicit source->solver
            # transform.  Reconstruct the source pose from the same GRAIL
            # metadata instead of silently querying an identity-pose mesh.
            source_scene_ref = bundle.source_scene
            obj = canonical.scene.get("obj_data", {}) if isinstance(canonical.scene, dict) else {}
            if source_scene_ref is not None and isinstance(obj, dict) and (obj.get("obj_R") is not None or obj.get("obj_t") is not None):
                source_pose = np.eye(4)
                source_rotation = np.asarray(obj.get("obj_R", np.eye(3)), dtype=float)
                if source_rotation.ndim == 3:
                    source_rotation = source_rotation[0]
                source_scale = np.asarray(obj.get("obj_scale", 1.0), dtype=float).reshape(-1)
                source_pose[:3, :3] = source_rotation.reshape(3, 3) * (float(source_scale[0]) if len(source_scale) == 1 else 1.0)
                if len(source_scale) == 3:
                    source_pose[:3, :3] = source_rotation.reshape(3, 3) @ np.diag(source_scale)
                source_translation = np.asarray(obj.get("obj_t", np.zeros(3)), dtype=float).reshape(-1)
                if len(source_translation) >= 3:
                    source_pose[:3, 3] = source_translation[:3]
                source_scene_ref = SceneReference(
                    path=source_scene_ref.path,
                    format=source_scene_ref.format,
                    asset_id=source_scene_ref.asset_id,
                    pose=source_pose,
                    pose_trajectory=_pose_trajectory_from_obj(
                        obj, canonical.fps, SceneTransform(np.eye(3), 1.0, np.zeros(3)), canonical.timestamps
                    ),
                    metadata=source_scene_ref.metadata,
                )
            source_floor_height = floor_height
            if floor_policy == FloorPolicy.AUTO:
                source_floor_height = _estimate_auto_floor(canonical, SceneTransform(np.eye(3), 1.0, np.zeros(3)))
            source_assembly = SceneAssembler().assemble(
                source_scene_ref, floor_policy=floor_policy, floor_height=source_floor_height,
                sample_count=scene_sample_count,
            )
            source_scene, contact_terrain = source_assembly.model, source_assembly.terrain
        contact_motion = contact_motion_full
        contact_transform = None
        if relation == SceneRelation.SOURCE_TO_TARGET:
            # Source detection runs in the source scene frame; the detector's
            # explicit transform then produces solver-world anchors.
            contact_motion = _contact_motion_in_query_frame(
                contact_motion_full,
                source_positions_for_contacts,
                transform,
                relation,
            )
            contact_transform = transform
        else:
            # Shared-scene and motion-only queries use the terrain already
            # assembled in solver/world coordinates.  Apply the same single
            # SceneTransform to the contact detector's human points; using
            # source-world points here would make a non-unit scene transform
            # silently produce wrong surface ids and support distances.
            contact_motion = _contact_motion_in_query_frame(
                contact_motion_full,
                source_positions_for_contacts,
                transform,
                relation,
            )
        plan=build_contact_plan(contact_motion,contact_terrain,fps=args.tgt_fps,transform=contact_transform,config=cfg.get("terrain_contact",{}))
        if args.max_frames is not None:
            limit = min(int(args.max_frames), len(plan.per_frame_states))
            plan = ContactPlan(
                episodes=tuple(
                    ContactEpisode(
                        episode.body_channel, episode.phase, episode.start_frame,
                        min(episode.end_frame, limit - 1), episode.object_id,
                        episode.surface_id, episode.source_anchor, episode.normal,
                        triangle_id=episode.triangle_id,
                        barycentric=episode.barycentric,
                        asset_local_anchor=episode.asset_local_anchor,
                        confidence=episode.confidence,
                        source_provenance=episode.source_provenance,
                        target_provenance=episode.target_provenance,
                    )
                    for episode in plan.episodes
                    if episode.start_frame < limit
                ),
                per_frame_states=plan.per_frame_states[:limit],
                fps=plan.fps,
                metadata={
                    **plan.metadata,
                    "truncated_from_frame_count": len(plan.per_frame_states),
                },
            )
        bound_schedule = ContactBinder().bind(
            plan,
            relation=relation,
            binding=(job or {}).get("contact_binding", {}),
            target_terrain=terrain,
        )
    xml = base_robot_xml
    # Mesh scenes enter the same MuJoCo model used by the V5 solver.  The
    # visual source mesh, cached convex pieces and object pose are generated
    # from this one SceneReference; no motion-only fallback is permitted.
    effective_config = copy.deepcopy(cfg)
    # The adapter owns the measured human morphology.  Preserve scene metric
    # scale, but make the actual source height available to the root policy
    # instead of silently using the configuration's generic adult default.
    effective_config.setdefault("morphology", {})["human_height"] = float(
        morphology.human_height
        if morphology.human_height is not None
        else morphology.provenance.get("source_height", cfg.get("scene", {}).get("default_human_height", 1.78))
    )
    effective_config["morphology"]["chain_scales"] = dict(morphology.chain_scales)
    scene_alignment = {"status": "NOT_APPLICABLE", "assets": [], "transform_mismatches": []}
    scene_manifest_payload = None
    if scene_ref is not None and (scene_assembly.geometry.meshes or scene_assembly.geometry.boxes):
        from general_motion_retargeting.scene_mujoco import build_scene_model
        combined = output.with_name(f".{output.stem}_v5_combined.xml")
        combined_info = build_scene_model(
            xml, scene_assembly.geometry, combined,
            cache_root=ROOT / ".cache" / "scene_collision",
            # Use the V5-configured decomposition explicitly.  In particular,
            # a cached coarse hull must not be reused for a stair/chair whose
            # desired-contact surface is much more precise than the hull.
            decomposition_config=effective_config.get("scene_collision", {}).get(
                "decomposition", {}
            ),
            floor_z=scene.floor_height, return_info=True,
        )
        scene_alignment = alignment_sanity(
            scene, combined_info.manifest_path, geometry=scene_assembly.geometry
        )
        scene_manifest_payload = json.loads(combined_info.manifest_path.read_text(encoding="utf-8"))
        effective_config["robot_xml"] = str(combined)
    effective_config_path = output.with_name(f".{output.stem}_v5_config.json")
    effective_config_path.parent.mkdir(parents=True, exist_ok=True)
    effective_config_path.write_text(json.dumps(effective_config, indent=2))
    robot_xml_for_task = Path(effective_config.get("robot_xml", str(xml))).expanduser()
    if not robot_xml_for_task.is_absolute():
        robot_xml_for_task = (ROOT / robot_xml_for_task).resolve()
    robot_spec=RobotModelSpec(robot,robot_xml_for_task,cfg.get("semantic_points",{}),cfg.get("joint_position_limits",{}),cfg.get("solver",{}))
    task=RetargetTask(bundle,scene,effective_config_path,robot_spec,effective_config,metadata={"retarget_job": jsonable(retarget_job)})
    realized_schedule = RobotContactRealizer().realize(
        bound_schedule, effective_config.get("contact_tasks", {}).get("robot_points", {})
    )
    solver=WholeBodyRetargetSolver(
        task, terrain, _interaction_anchors(plan, terrain, canonical.positions),
        fps=args.tgt_fps, solver=args.solver, scene_model=scene,
    )
    artifacts = solve_and_validate(
        solver,
        canonical,
        solver_frames,
        realized_schedule,
        robot_reference_frames,
        source_scene_ok=motion_scene_alignment.get("status") != "INCONSISTENT_EVIDENCE",
        scene_alignment_ok=scene_alignment.get("status") != "FAIL",
    )
    result = artifacts.result
    runtime_schedule = artifacts.runtime_schedule
    kinematics = artifacts.kinematics
    validation = artifacts.validation
    # Export names describe robot arrays, never source human landmarks.
    robot_joint_names = list(kinematics["robot_joint_names"])
    payload={
        "schema_version": 2,
        "algorithm": "wholebody_omni_gmr_v5",
        "retarget_job": jsonable(retarget_job),
        "scope_admission": scope_decision.to_dict(),
        "run_identity": {
            "input_sha256": scope_decision.evidence.get("input_sha256"),
            "config_sha256": file_sha256(config),
            "robot_model_sha256": file_sha256(robot_xml_for_task),
            "scene_sha256": file_sha256(scene_ref.path) if scene_ref is not None and scene_ref.path is not None and scene_ref.path.is_file() else None,
        },
        "effective_config": jsonable(effective_config),
        "status": validation["status"],
        "validation": validation,
        "failure": result.failure,
        "fps": args.tgt_fps,
        "timestamps": canonical.timestamps,
        "qpos": result.qpos,
        "root_pos": result.qpos[:,:3],
        "root_rot": result.qpos[:,3:7][:,[1,2,3,0]],
        "root_rot_order": "xyzw",
        "dof_pos": kinematics["joint_pos"],
        "joint_pos": kinematics["joint_pos"],
        "joint_vel": kinematics["joint_vel"],
        "joint_names": robot_joint_names,
        "dof_names": list(kinematics["robot_dof_names"]),
        "body_names": kinematics["body_names"],
        "body_pos_w": kinematics["body_pos_w"],
        "body_quat_w": kinematics["body_quat_w"],
        "body_lin_vel_w": kinematics["body_lin_vel_w"],
        "body_ang_vel_w": kinematics["body_ang_vel_w"],
        "source_motion": str(bundle.motion.path),
        "source_format": canonical.source_format,
        "source_joint_names": canonical.joint_names,
        "canonical_capabilities": canonical.capabilities,
        "canonical_provenance": canonical.landmark_provenance,
        "canonical_metadata": jsonable(canonical.metadata),
        "morphology": jsonable(morphology),
        "robot_reference": jsonable(solver.reference_metadata),
        "scene_relation": bundle.scene_relation.value,
        "scene_alignment": scene_alignment,
        "scene_manifest": scene_manifest_payload,
        "scene_transform": transform.to_dict(),
        "scene_floor_provenance": floor_provenance,
        "motion_scene_alignment": motion_scene_alignment,
        "frame_graph": {"source_motion": "source_motion", "canonical": "canonical", "solver_world": "solver_world", "mujoco_world": "mujoco_world", "asset_local": "asset_local", "robot_link": "robot_link"},
        "contact_plan":{"episodes":[jsonable(e.__dict__) for e in plan.episodes],"frames":list(runtime_schedule.per_frame_states),"metadata":{**plan.metadata, **realized_schedule.metadata, **runtime_schedule.metadata},"source_frames":list(realized_schedule.per_frame_states)},
        "diagnostics":result.diagnostics,
        "terrain":terrain.to_spec(),
        "robot_xml":str(solver.robot_xml),
        "scene":{"assets":[{"asset_id":a.asset_id,"path":str(a.path),"pose":a.pose.tolist(),"pose_trajectory":jsonable(a.pose_trajectory),"collision_enabled":a.collision_enabled,"unit_scale":a.unit_scale,"asset_space":a.asset_space,"asset_scale_baked":a.asset_scale_baked} for a in scene.assets],"floor_height":scene.floor_height},
    }
    arrays={
        "fps":np.asarray(args.tgt_fps), "timestamps":canonical.timestamps,
        "qpos":result.qpos, "root_pos":result.qpos[:,:3],
        "root_rot":result.qpos[:,3:7][:,[1,2,3,0]], "dof_pos":kinematics["joint_pos"],
        "joint_pos":kinematics["joint_pos"], "joint_vel":kinematics["joint_vel"],
        "body_pos_w":kinematics["body_pos_w"], "body_quat_w":kinematics["body_quat_w"],
        "body_lin_vel_w":kinematics["body_lin_vel_w"], "body_ang_vel_w":kinematics["body_ang_vel_w"],
        "joint_names":np.asarray(robot_joint_names), "body_names":np.asarray(kinematics["body_names"]),
        "source_format":np.asarray(canonical.source_format), "scene_relation":np.asarray(bundle.scene_relation.value),
        # Metadata is duplicated as JSON strings in NPZ so downstream
        # HoloSoMo consumers can preserve the scene/contact contract without
        # having to unpickle the diagnostic payload.
        "root_rot_order": np.asarray("xyzw"),
        "scene_transform_json": np.asarray(json.dumps(jsonable(transform.to_dict()), ensure_ascii=False)),
        "scene_floor_provenance_json": np.asarray(json.dumps(jsonable(floor_provenance), ensure_ascii=False)),
        "terrain_spec_json": np.asarray(json.dumps(jsonable(terrain.to_spec()), ensure_ascii=False)),
        "contact_schedule_json": np.asarray(json.dumps(jsonable(runtime_schedule.per_frame_states), ensure_ascii=False)),
        "source_contact_schedule_json": np.asarray(json.dumps(jsonable(realized_schedule.per_frame_states), ensure_ascii=False)),
        "validation_json": np.asarray(json.dumps(jsonable(validation), ensure_ascii=False)),
        "scene_manifest_json": np.asarray(json.dumps(jsonable(scene_manifest_payload), ensure_ascii=False)),
    }
    output, npz_output, diagnostics_output = export_result(
        output, payload, arrays, result.diagnostics,
        allow_invalid=bool(args.save_invalid_debug),
    )
    effective_config_path.unlink(missing_ok=True)
    print(f"WholeBody V5: {result.status}\nFormat: {canonical.source_format}\nFrames: {len(result.qpos)}\nSaved: {output}\nSaved NPZ: {npz_output}\nDiagnostics: {diagnostics_output}")
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--job",type=Path); parser.add_argument("--motion",type=Path); parser.add_argument("--robot",default="ne01"); parser.add_argument("--output",type=Path); parser.add_argument("--manifest",type=Path); parser.add_argument("--joint_map",type=Path); parser.add_argument("--motion_format",default="auto",choices=("auto","canonical_npz","smplx_npz","grail_smplx_recon","holosoma_global_positions","bvh","fbx")); parser.add_argument("--bvh-format",default="lafan1",choices=("lafan1","xsens","nokov"),help="explicit BVH unit/axis profile"); parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG); parser.add_argument("--body_models",type=Path,default=ROOT/"assets/body_models"); parser.add_argument("--tgt_fps",type=float,default=50.); parser.add_argument("--solver",choices=("daqp","proxqp"),default="daqp"); parser.add_argument("--task-family",choices=("flat_motion","terrain_locomotion"),default=None); parser.add_argument("--terrain-kind",choices=("flat","stairs","ramp"),default=None); parser.add_argument("--interaction-mode",choices=("feet_only",),default=None); parser.add_argument("--max_frames",type=int,default=None); parser.add_argument("--save-invalid-debug",action="store_true"); args=parser.parse_args()
    try:
        result=run(args)
    except ScopeAdmissionError as error:
        output = args.output
        if args.job is not None:
            job = json.loads(args.job.read_text(encoding="utf-8"))
            output = Path(job.get("output", {}).get("path", output or "work/v5_result.pkl"))
            if not output.is_absolute():
                output = (args.job.parent / output).resolve()
        output = Path(output or "work/v5_result.pkl").expanduser().resolve()
        report_path = output.with_suffix(".status.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            **error.decision.to_dict(),
            "motion": str(args.motion) if args.motion is not None else None,
            "job": str(args.job) if args.job is not None else None,
            "formal_motion_written": False,
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"WholeBody V5: {error.decision.status.value}\nReport: {report_path}")
        raise SystemExit(3 if error.decision.status.value == "EXCLUDED" else 4)
    except (FileNotFoundError, ValueError) as error:
        # Input/scene contract failures are first-class batch outcomes, not
        # unstructured tracebacks.  Keep the report next to the requested
        # output while preserving a non-zero exit code.
        output = args.output or Path("work/v5_result.pkl")
        if args.job is not None:
            try:
                job = json.loads(args.job.read_text(encoding="utf-8"))
                output = Path(job.get("output", {}).get("path", output))
                if not output.is_absolute():
                    output = (args.job.parent / output).resolve()
            except Exception:
                pass
        output = Path(output).expanduser().resolve()
        report_path = output.with_suffix(".status.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({
            "status": "INPUT_ERROR", "reason_code": type(error).__name__,
            "error": str(error), "formal_motion_written": False,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"WholeBody V5: INPUT_ERROR\nReport: {report_path}")
        raise SystemExit(4)
    except RuntimeError as error:
        if "V5 validation failed" not in str(error):
            raise
        print(f"WholeBody V5: INVALID\n{error}")
        raise SystemExit(2)
    # A debug artifact may be written for an invalid solve, but the CLI must
    # still be machine-detectable as failed.  Callers can explicitly opt into
    # inspecting the ``.INVALID_DEBUG`` files without mistaking them for a
    # formal retarget result.
    if result.status != "VALID":
        raise SystemExit(2)


if __name__ == "__main__": main()
