"""Convert NE01 GMR PKL motions to HoloSoMo whole-body-tracking NPZ.

HoloSoMo stores root DOFs inside joint_pos/joint_vel and expects body poses
and velocities for the named robot links.  The FK and body velocities are
recomputed from the final exported qpos rather than inferred from local_body_pos.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


JOINT_NAMES = [
    "WAIST_ROLL_JOINT", "TORSO_JOINT",
    "SHOULDER_PITCH_R_JOINT", "SHOULDER_ROLL_R_JOINT", "SHOULDER_YAW_R_JOINT",
    "ELBOW_PITCH_R_JOINT", "HAND_YAW_R_JOINT",
    "SHOULDER_PITCH_L_JOINT", "SHOULDER_ROLL_L_JOINT", "SHOULDER_YAW_L_JOINT",
    "ELBOW_PITCH_L_JOINT", "HAND_YAW_L_JOINT",
    "HIP_PITCH_R_JOINT", "HIP_ROLL_R_JOINT", "HIP_YAW_R_JOINT",
    "KNEE_PITCH_R_JOINT", "ANKLE_PITCH_R_JOINT", "ANKLE_ROLL_R_JOINT",
    "HIP_PITCH_L_JOINT", "HIP_ROLL_L_JOINT", "HIP_YAW_L_JOINT",
    "KNEE_PITCH_L_JOINT", "ANKLE_PITCH_L_JOINT", "ANKLE_ROLL_L_JOINT",
]

CANONICAL_BODY_NAMES = [
    "BASE_LINK", "WAIST_ROLL_LINK", "TORSO_LINK",
    "SHOULDER_PITCH_R_LINK", "SHOULDER_ROLL_R_LINK", "SHOULDER_YAW_R_LINK",
    "ELBOW_PITCH_R_LINK", "HAND_YAW_R_LINK",
    "SHOULDER_PITCH_L_LINK", "SHOULDER_ROLL_L_LINK", "SHOULDER_YAW_L_LINK",
    "ELBOW_PITCH_L_LINK", "HAND_YAW_L_LINK",
    "HIP_PITCH_R_LINK", "HIP_ROLL_R_LINK", "HIP_YAW_R_LINK",
    "KNEE_PITCH_R_LINK", "ANKLE_PITCH_R_LINK", "ANKLE_ROLL_R_LINK",
    "HIP_PITCH_L_LINK", "HIP_ROLL_L_LINK", "HIP_YAW_L_LINK",
    "KNEE_PITCH_L_LINK", "ANKLE_PITCH_L_LINK", "ANKLE_ROLL_L_LINK",
]

# GMR's first two joints/bodies have the same physical role as HoloSoMo's
# differently named NE01 entries.  The remaining names are identical.
BODY_TO_MUJOCO = {"BASE_LINK": "base_link", "WAIST_ROLL_LINK": "WAIST_YAW"}
BODY_TO_MUJOCO.update({name: name for name in CANONICAL_BODY_NAMES if name not in BODY_TO_MUJOCO})


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


def _model_joint_names(model: mujoco.MjModel, expected_dofs: int) -> list[str]:
    """Return the actual articulated-joint names in qpos order.

    The desktop NE01 asset intentionally uses ``base_link`` and ``WAIST_YAW``;
    HoloSoMo's packaged asset uses a different spelling.  The exported motion
    schema must name the bodies and DOFs of the XML used for its FK.
    """
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(names) != expected_dofs:
        raise ValueError(f"Expected {expected_dofs} articulated joints, found {len(names)} in {model!r}")
    return names


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw), axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from wxyz quaternions by centered differences."""
    result = np.zeros((len(quat_wxyz), 3), dtype=np.float64)
    if len(quat_wxyz) < 2:
        return result
    for i in range(len(quat_wxyz)):
        if i == 0:
            qa, qb, denom = quat_wxyz[0], quat_wxyz[1], dt
        elif i == len(quat_wxyz) - 1:
            qa, qb, denom = quat_wxyz[-2], quat_wxyz[-1], dt
        else:
            qa, qb, denom = quat_wxyz[i - 1], quat_wxyz[i + 1], 2.0 * dt
        delta = quat_mul(qb, quat_conj(qa))
        if delta[0] < 0.0:
            delta = -delta
        result[i] = Rotation.from_quat(delta[[1, 2, 3, 0]]).as_rotvec() / denom
    return result


def convert_file(source: pathlib.Path, target: pathlib.Path, model: mujoco.MjModel) -> None:
    with source.open("rb") as f:
        motion = pickle.load(f)
    fps = float(motion["fps"])
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3 or dof_pos.ndim != 2 or dof_pos.shape[1] != 24:
        raise ValueError(f"Unexpected motion shapes in {source}: root={root_pos.shape}, dof={dof_pos.shape}")
    if len(root_pos) != len(dof_pos) or len(root_pos) != len(root_xyzw):
        raise ValueError(f"Frame count mismatch in {source}")

    # HoloSoMo's NE01 asset uses the canonical uppercase link names.  Keep
    # compatibility with the older GMR XML, whose root/waist bodies were
    # named base_link/WAIST_YAW.
    try:
        has_holosoma_names = model.body("BASE_LINK").id > 0
    except (KeyError, ValueError):
        has_holosoma_names = False
    if has_holosoma_names:
        body_mapping = {name: name for name in CANONICAL_BODY_NAMES}
    else:
        body_mapping = BODY_TO_MUJOCO
    exported_body_names = [body_mapping[name] for name in CANONICAL_BODY_NAMES]
    body_ids = [model.body(name).id for name in exported_body_names]
    exported_joint_names = _model_joint_names(model, dof_pos.shape[1])
    qpos = np.concatenate((root_pos, root_xyzw[:, [3, 0, 1, 2]], dof_pos), axis=1)
    body_pos = np.empty((len(qpos), len(exported_body_names), 3), dtype=np.float64)
    body_quat = np.empty((len(qpos), len(exported_body_names), 4), dtype=np.float64)
    data = mujoco.MjData(model)
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos[body_ids]
        body_quat[frame] = data.xquat[body_ids]  # MuJoCo/HoloSoMo file convention: wxyz

    dt = 1.0 / fps
    root_lin_vel = np.gradient(root_pos, dt, axis=0, edge_order=1) if len(qpos) > 1 else np.zeros_like(root_pos)
    root_ang_vel = angular_velocity(root_quat := qpos[:, 3:7], dt)
    dof_vel = np.gradient(dof_pos, dt, axis=0, edge_order=1) if len(qpos) > 1 else np.zeros_like(dof_pos)
    body_lin_vel = np.gradient(body_pos, dt, axis=0, edge_order=1) if len(qpos) > 1 else np.zeros_like(body_pos)
    body_ang_vel = np.stack([angular_velocity(body_quat[:, i], dt) for i in range(len(exported_body_names))], axis=1)

    joint_pos = np.concatenate((qpos[:, :7], dof_pos), axis=1)
    joint_vel = np.concatenate((root_lin_vel, root_ang_vel, dof_vel), axis=1)
    object_payload = {}
    if "object_pos_w" in motion and "object_quat_w" in motion:
        object_pos = np.asarray(motion["object_pos_w"], dtype=np.float64)
        object_quat = np.asarray(motion["object_quat_w"], dtype=np.float64)
        if object_pos.shape != (len(qpos), 3) or object_quat.shape != (len(qpos), 4):
            raise ValueError(f"Object trajectory shape mismatch in {source}: {object_pos.shape}, {object_quat.shape}")
        object_payload.update(
            object_pos_w=object_pos,
            object_quat_w=object_quat,
            object_lin_vel_w=np.gradient(object_pos, dt, axis=0, edge_order=1) if len(qpos) > 1 else np.zeros_like(object_pos),
            object_ang_vel_w=np.stack([angular_velocity(object_quat, dt)], axis=0)[0],
        )
        if "object_scale" in motion:
            object_payload["object_scale"] = np.asarray(motion["object_scale"])
        if "object_asset_path" in motion:
            object_payload["object_asset_path"] = np.asarray(str(motion["object_asset_path"]))
    terrain_payload = {}
    for key in ("scene_transform", "terrain_primitives", "contact_schedule", "contact_metrics", "terrain_diagnostics", "orientation_valid"):
        if key in motion:
            terrain_payload[key] = np.asarray(json.dumps(_jsonable(motion[key]), separators=(",", ":")))
    if "terrain_surface_ids" in motion:
        terrain_payload["terrain_surface_ids"] = np.asarray(motion["terrain_surface_ids"], dtype=str)
    for key in ("source_motion", "source_terrain", "robot_xml"):
        if key in motion:
            terrain_payload[key] = np.asarray(str(motion[key]))
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        fps=np.asarray(round(fps), dtype=np.int64),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
        joint_names=np.asarray(exported_joint_names),
        body_names=np.asarray(exported_body_names),
        **object_payload,
        **terrain_payload,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", type=pathlib.Path, required=True)
    parser.add_argument("--tgt_folder", type=pathlib.Path, required=True)
    parser.add_argument("--xml", type=pathlib.Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    sources = sorted(args.src_folder.glob("*.pkl"))
    if not sources:
        raise RuntimeError(f"No PKL files found in {args.src_folder}")
    for source in sources:
        convert_file(source, args.tgt_folder / f"{source.stem}.npz", model)
        print(f"Converted {source.name}")
    print(f"Converted {len(sources)} motions to {args.tgt_folder}")


if __name__ == "__main__":
    main()
