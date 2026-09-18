"""Retarget one GRAIL reconstruction with the standalone WholeBody Omni V4 solver.

GRAIL reconstructions contain SMPL-X parameters and an optional interaction
object.  This adapter performs SMPL-X FK, maps the resulting semantic points
to the V4 configuration, and keeps object metadata in the exported payload.
The companion ``grail_retarget_scene.py`` entry resolves complex USD/OBJ
assets into both the interaction pool and MuJoCo collision model.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.terrain_geometry import SceneTransform, TerrainField
from general_motion_retargeting.utils.smpl import (
    get_smplx_data_offline_fast,
    load_smplx_file,
)
from general_motion_retargeting.omni_solver import WholeBodyV4
from general_motion_retargeting.omni_solver_core import sample_terrain_surface_pool


DEFAULT_CONFIG = ROOT / "general_motion_retargeting/ik_configs/ne01_v4.json"


def _load_grail(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            value = pickle.load(stream)
    except ModuleNotFoundError as error:
        if "numpy._core" not in str(error):
            raise
        import importlib
        for target, source in {
            "numpy._core": "numpy.core",
            "numpy._core.numeric": "numpy.core.numeric",
            "numpy._core.multiarray": "numpy.core.multiarray",
            "numpy._core.umath": "numpy.core.umath",
            "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
        }.items():
            sys.modules.setdefault(target, importlib.import_module(source))
        with path.open("rb") as stream:
            value = pickle.load(stream)
    if not isinstance(value, dict) or not isinstance(value.get("human_data"), dict):
        raise ValueError(f"{path} is not a GRAIL reconstruction pickle")
    return value


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _config_for_smplx(config: dict) -> dict:
    """Translate source labels to the names emitted by SMPL-X FK."""
    if "extends" in config:
        base_path = Path(__file__).resolve().parent.parent / "general_motion_retargeting/ik_configs" / config["extends"]
        base = json.loads(base_path.read_text())
        # Merge nested sections instead of replacing them wholesale.  V4
        # overrides only a few semantic/contact entries; a shallow update
        # silently discarded inherited heel/toe/knee/palm robot points.
        def merge(parent: dict, override: dict) -> dict:
            result = copy.deepcopy(parent)
            for key, value in override.items():
                if isinstance(value, dict) and isinstance(result.get(key), dict):
                    result[key] = merge(result[key], value)
                else:
                    result[key] = copy.deepcopy(value)
            return result
        config = merge(base, {k: v for k, v in config.items() if k != "extends"})
    labels = {
        "Spine1": "pelvis",
        "pelvis": "pelvis",
        "LeftUpLeg": "left_hip", "RightUpLeg": "right_hip",
        "LeftLeg": "left_knee", "RightLeg": "right_knee",
        "LeftFoot": "left_ankle", "RightFoot": "right_ankle",
        "LeftToeBase": "left_toe", "RightToeBase": "right_toe",
        "LeftArm": "left_shoulder", "RightArm": "right_shoulder",
        "LeftForeArm": "left_elbow", "RightForeArm": "right_elbow",
        "LeftHandMiddle3": "left_wrist", "RightHandMiddle3": "right_wrist",
        "left_hip": "left_hip", "right_hip": "right_hip",
        "left_knee": "left_knee", "right_knee": "right_knee",
        "left_foot": "left_ankle", "right_foot": "right_ankle",
        "left_toe": "left_toe", "right_toe": "right_toe",
        "left_shoulder": "left_shoulder", "right_shoulder": "right_shoulder",
        "left_elbow": "left_elbow", "right_elbow": "right_elbow",
        "left_wrist": "left_wrist", "right_wrist": "right_wrist",
    }
    adapted = copy.deepcopy(config)
    for item in adapted["semantic_points"].values():
        item["source"] = labels[item["source"]]
    adapted["global_anchor"]["source"] = "pelvis"
    return adapted


def _source_frame(frame: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value[0], dtype=float).copy() for name, value in frame.items()}


def _contact_aliases(points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Provide the HoloSoMo labels expected by the shared contact utility."""
    aliases = dict(points)
    required = {
        "LeftFoot": "left_foot", "RightFoot": "right_foot",
        "LeftToeBase": "left_toe", "RightToeBase": "right_toe",
        "LeftHand": "left_wrist", "RightHand": "right_wrist",
        "LeftLeg": "left_knee", "RightLeg": "right_knee",
    }
    for alias, source in required.items():
        if source not in points:
            raise ValueError(f"SMPL-X FK did not emit required point {source}")
        aliases[alias] = points[source]
    # Preserve measured surface landmarks for the generic terrain contact
    # scheduler; these are positions only and carry no fabricated rotation.
    for side in ("left", "right"):
        aliases[f"{side}_heel"] = points[f"{side}_heel"]
        aliases[f"{side}_big_toe"] = points[f"{side}_big_toe"]
        aliases[f"{side}_small_toe"] = points[f"{side}_small_toe"]
    return aliases


def _anatomical_quaternion(points: dict[str, np.ndarray], center: str = "pelvis") -> np.ndarray:
    """Build a stable anatomical frame from landmarks, independent of SMPLX axes."""
    if center == "pelvis":
        up_vec = points["spine3"] - points["pelvis"]
    else:
        up_vec = points["spine3"] - points["pelvis"]
    left_vec = points["left_hip"] - points["right_hip"]
    up_norm = np.linalg.norm(up_vec)
    left_norm = np.linalg.norm(left_vec)
    if up_norm < 1e-8 or left_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    up = up_vec / up_norm
    left = left_vec / left_norm
    forward = np.cross(left, up)
    if np.linalg.norm(forward) < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    left /= max(np.linalg.norm(left), 1e-12)
    return Rotation.from_matrix(np.column_stack((forward, left, up))).as_quat(scalar_first=True)


def main(
    argv: list[str] | None = None,
    *,
    retargeter_cls=WholeBodyV4,
    pool_builder=sample_terrain_surface_pool,
    contact_provider=None,
) -> None:
    """Run GRAIL conversion with explicit dependencies, never global injection.

    Scene-aware callers supply a combined-model solver class, a terrain plus
    scene interaction-pool builder, and a mesh contact augmentation callback.
    All three dependencies are local to this invocation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--save_path", required=True, type=Path)
    parser.add_argument("--terrain", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tgt_fps", type=float, default=50.0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--solver", choices=("daqp", "proxqp"), default="daqp")
    args = parser.parse_args(argv)

    record = _load_grail(args.motion)
    human = record["human_data"]
    temporary = args.save_path.parent / f".{args.save_path.stem}_smplx_input.npz"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        temporary,
        root_orient=np.asarray(human["poses"], dtype=np.float32)[:, :3],
        pose_body=np.asarray(human["poses"], dtype=np.float32)[:, 3:66],
        trans=np.asarray(human["trans"], dtype=np.float32),
        betas=np.asarray(human.get("betas", np.zeros(10)), dtype=np.float32).reshape(-1)[:10],
        gender=np.asarray(str(human.get("gender", "neutral"))),
        mocap_frame_rate=np.asarray(float(human.get("mocap_frame_rate", 50.0))),
    )
    try:
        data, model, output, human_height = load_smplx_file(
            temporary, ROOT / "assets" / "body_models"
        )
        frames, fps = get_smplx_data_offline_fast(
            data, model, output, tgt_fps=args.tgt_fps
        )
    finally:
        temporary.unlink(missing_ok=True)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    if not frames:
        raise ValueError("GRAIL motion contains no frames")

    config = _config_for_smplx(json.loads(args.config.read_text()))
    terrain = TerrainField.from_file(args.terrain) if args.terrain else TerrainField()
    terrain.support_normal_min_z = float(config["terrain_contact"]["support_normal_min_z"])
    # Use explicit V4 scene scale when a complete scene already shares the
    # source world; otherwise derive a consistent human-to-robot scale.
    scene_scale_value = config["scene"].get("scene_scale")
    scene_scale = (float(scene_scale_value) if scene_scale_value is not None
                   else float(config["scene"]["robot_height"]) / float(human_height))
    source_reference_height = config["scene"].get("source_reference_height")
    if source_reference_height is not None:
        scene_scale = float(config["scene"].get("robot_height", 1.316)) / float(source_reference_height)
    transform = SceneTransform(
        rotation=np.asarray(config["scene"]["rotation"], dtype=float),
        scale=scene_scale,
        translation=np.asarray(config["scene"]["translation"], dtype=float),
    )
    source_frames = []
    orientation_frames = []
    contacts_input = []
    for frame in frames:
        points = _source_frame(frame)
        # Use measured SMPL-X landmarks.  The canonical foot is the ankle;
        # toe is the measured big/small-toe midpoint and heel is measured.
        for side in ("left", "right"):
            points[f"{side}_foot"] = points[f"{side}_ankle"].copy()
            points[f"{side}_toe"] = 0.5 * (
                points[f"{side}_big_toe"] + points[f"{side}_small_toe"]
            )
            points[f"{side}_heel"] = points[f"{side}_heel"].copy()
        transformed = {name: transform.transform_points(point) for name, point in points.items()}
        source_frames.append(transformed)
        # Source SMPL-X pelvis axes are model-specific.  Use the landmark-
        # derived anatomical frame so GRAIL follows the same convention as
        # the other adapters and cannot introduce a global 90/180 degree leg
        # twist.  The torso target shares this calibrated frame.
        anatomical_quat = _anatomical_quaternion(points)
        orientation_frames.append((anatomical_quat, anatomical_quat))
        # Contact inference owns the source->solver transform.  Passing the
        # already transformed points here would apply the similarity twice.
        contacts_input.append(_contact_aliases(points))

    # Contact schedule is computed from human points only; it never sees qpos.
    from general_motion_retargeting.terrain_contact_utils import build_terrain_contact_schedule

    solver_terrain = terrain.transform(transform)
    schedule = build_terrain_contact_schedule(
        contacts_input,
        terrain,
        transform,
        {},
        float(args.tgt_fps),
        config["terrain_contact"],
    )
    if contact_provider is not None:
        schedule = contact_provider(schedule, source_frames, config)
    transformed_points = np.asarray(
        [[*frame.values()] for frame in source_frames], dtype=float
    ).reshape(-1, 3)
    pool = pool_builder(solver_terrain, transformed_points)
    adapted_config = args.save_path.parent / f".{args.save_path.stem}_v4_config.json"
    adapted_config.write_text(json.dumps(config, indent=2))
    try:
        retargeter = retargeter_cls(adapted_config, solver_terrain, pool, fps=args.tgt_fps, solver=args.solver)
        qpos = [retargeter.retarget(frame, ori[0], contact, chest_quaternion=ori[1]) for frame, ori, contact in zip(source_frames, orientation_frames, schedule)]
    finally:
        adapted_config.unlink(missing_ok=True)
    qpos = np.asarray(qpos, dtype=float)
    # The solver and every scene representation use the same origin.  Keep
    # source-world coordinates here; shifting only qpos would detach the chair
    # from the robot.  Consumers that normalize XY must transform the scene
    # metadata with the identical shift.
    object_data = record.get("obj_data") or {}
    payload = {
        "fps": float(args.tgt_fps), "qpos": qpos,
        "root_pos": qpos[:, :3], "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": qpos[:, 7:],
        "algorithm": "wholebody_omni_v4",
        "source_motion": str(args.motion), "robot_xml": str(retargeter.robot_xml),
        "scene_transform": transform.to_dict(), "terrain_primitives": solver_terrain.to_spec(),
        "contact_schedule": schedule, "terrain_diagnostics": retargeter.diagnostics,
        "grail_object_path": record.get("object_path", ""),
        "grail_object_data": object_data,
    }
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    with args.save_path.with_suffix(".pkl").open("wb") as stream:
        pickle.dump(payload, stream)
    np.savez_compressed(args.save_path.with_suffix(".npz"), **{k: _jsonable(v) for k, v in payload.items() if k not in {"contact_schedule", "terrain_diagnostics", "grail_object_data"}})
    print(f"Saved {retargeter.__class__.__name__} GRAIL PKL: {args.save_path.with_suffix('.pkl')}")
    print(f"Saved {retargeter.__class__.__name__} GRAIL NPZ: {args.save_path.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
