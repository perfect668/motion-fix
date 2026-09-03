import numpy as np
import smplx
import torch
from pathlib import Path
from types import SimpleNamespace
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES
from scipy.interpolate import interp1d

import general_motion_retargeting.utils.lafan_vendor.utils as utils

def load_smpl_file(smpl_file):
    smpl_data = np.load(smpl_file, allow_pickle=True)
    return smpl_data


def _get_betas_and_num_betas(smplx_data):
    betas = np.asarray(smplx_data["betas"])
    if betas.ndim == 2 and betas.shape[0] == 1:
        betas = betas.reshape(-1)
    if betas.ndim != 1:
        raise ValueError(f"Expected betas with shape (B,) or (1, B), got {betas.shape}")
    return betas, int(betas.shape[0])


def _scalar_to_str(value):
    value = np.asarray(value).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).lower()


def _has_smplx_model_file(smplx_body_model_path, gender):
    model_dir = Path(smplx_body_model_path) / "smplx"
    stem = f"SMPLX_{gender.upper()}"
    return any((model_dir / f"{stem}{suffix}").exists() for suffix in [".npz", ".pkl"])


def _resolve_smplx_gender(smplx_body_model_path, requested_gender):
    requested_gender = requested_gender.lower()
    if _has_smplx_model_file(smplx_body_model_path, requested_gender):
        return requested_gender
    if requested_gender != "neutral" and _has_smplx_model_file(smplx_body_model_path, "neutral"):
        print(
            f"Warning: SMPL-X {requested_gender} body model not found in "
            f"{Path(smplx_body_model_path) / 'smplx'}, falling back to neutral."
        )
        return "neutral"
    return requested_gender


def load_smplx_file(smplx_file, smplx_body_model_path):
    smplx_data = np.load(smplx_file, allow_pickle=True)
    betas, num_betas = _get_betas_and_num_betas(smplx_data)
    gender = _resolve_smplx_gender(
        smplx_body_model_path,
        _scalar_to_str(smplx_data["gender"]),
    )
    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=gender,
        use_pca=False,
        num_betas=num_betas,
    )
    # print(smplx_data["pose_body"].shape)
    # print(smplx_data["betas"].shape)
    # print(smplx_data["root_orient"].shape)
    # print(smplx_data["trans"].shape)
    
    num_frames = smplx_data["pose_body"].shape[0]
    betas_tensor = torch.tensor(betas).float().view(1, -1)
    root_orient = torch.tensor(smplx_data["root_orient"]).float()
    pose_body = torch.tensor(smplx_data["pose_body"]).float()
    trans = torch.tensor(smplx_data["trans"]).float()
    output_parts = {"global_orient": [], "full_pose": [], "joints": []}

    body_model.eval()
    with torch.no_grad():
        # SMPL-X materializes per-frame vertices internally. Chunking prevents
        # long AMASS recordings such as LARa from exhausting host memory.
        for start in range(0, num_frames, 512):
            end = min(start + 512, num_frames)
            batch_size = end - start
            output = body_model(
                betas=betas_tensor,
                global_orient=root_orient[start:end],
                body_pose=pose_body[start:end],
                transl=trans[start:end],
                left_hand_pose=torch.zeros(batch_size, 45).float(),
                right_hand_pose=torch.zeros(batch_size, 45).float(),
                jaw_pose=torch.zeros(batch_size, 3).float(),
                leye_pose=torch.zeros(batch_size, 3).float(),
                reye_pose=torch.zeros(batch_size, 3).float(),
                return_full_pose=True,
                return_verts=False,
            )
            for name in output_parts:
                output_parts[name].append(getattr(output, name).detach().cpu())

    smplx_output = SimpleNamespace(
        **{name: torch.cat(parts, dim=0) for name, parts in output_parts.items()}
    )
    
    human_height = 1.66 + 0.1 * betas[0]
    
    return smplx_data, body_model, smplx_output, human_height


def load_gvhmr_pred_file(gvhmr_pred_file, smplx_body_model_path):
    gvhmr_pred = torch.load(gvhmr_pred_file)
    smpl_params_global = gvhmr_pred['smpl_params_global']
    # print(smpl_params_global['body_pose'].shape)
    # print(smpl_params_global['betas'].shape)
    # print(smpl_params_global['global_orient'].shape)
    # print(smpl_params_global['transl'].shape)
    
    betas = smpl_params_global['betas'][0].numpy()
    
    # correct rotations
    # rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    # rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    
    # smpl_params_global['body_pose'] = smpl_params_global['body_pose'] @ rotation_matrix
    # smpl_params_global['global_orient'] = smpl_params_global['global_orient'] @ rotation_quat
    
    smplx_data = {
        'pose_body': smpl_params_global['body_pose'].numpy(),
        'betas': betas,
        'root_orient': smpl_params_global['global_orient'].numpy(),
        'trans': smpl_params_global['transl'].numpy(),
        "mocap_frame_rate": torch.tensor(30),
    }

    gender = _resolve_smplx_gender(smplx_body_model_path, "neutral")
    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=gender,
        use_pca=False,
        num_betas=betas.shape[0],
    )
    
    num_frames = smpl_params_global['body_pose'].shape[0]
    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1), # (16,)
        global_orient=torch.tensor(smplx_data["root_orient"]).float(), # (N, 3)
        body_pose=torch.tensor(smplx_data["pose_body"]).float(), # (N, 63)
        transl=torch.tensor(smplx_data["trans"]).float(), # (N, 3)
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        # expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )
    
    if len(smplx_data['betas'].shape)==1:
        human_height = 1.66 + 0.1 * smplx_data['betas'][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data['betas'][0, 0]
    
    return smplx_data, body_model, smplx_output, human_height


def get_smplx_data(smplx_data, body_model, smplx_output, curr_frame):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    global_orient = smplx_output.global_orient[curr_frame].squeeze()
    full_body_pose = smplx_output.full_pose[curr_frame].reshape(-1, 3)
    joints = smplx_output.joints[curr_frame].detach().numpy().squeeze()
    # SMPL-X returns extended surface landmarks beyond the kinematic parent
    # array.  Build names from the actual joint tensor, while computing
    # orientations only for joints with valid parents.
    joint_names = JOINT_NAMES[: len(joints)]
    kinematic_joint_names = joint_names[: len(body_model.parents)]
    parents = body_model.parents

    result = {}
    joint_orientations = []
    for i, joint_name in enumerate(kinematic_joint_names):
        if i == 0:
            rot = R.from_rotvec(global_orient)
        else:
            rot = joint_orientations[parents[i]] * R.from_rotvec(
                full_body_pose[i].squeeze()
            )
        joint_orientations.append(rot)
        result[joint_name] = (joints[i], rot.as_quat(scalar_first=True))

    for landmark in ("left_ankle", "right_ankle", "left_big_toe", "right_big_toe",
                     "left_small_toe", "right_small_toe", "left_heel", "right_heel"):
        if landmark in joint_names:
            result[landmark] = (joints[joint_names.index(landmark)], None)

  
    return result


def slerp(rot1, rot2, t):
    """Spherical linear interpolation between two rotations."""
    # Convert to quaternions
    q1 = rot1.as_quat()
    q2 = rot2.as_quat()
    
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product
    dot = np.sum(q1 * q2)
    
    # If the dot product is negative, slerp won't take the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If the inputs are too close, linearly interpolate
    if dot > 0.9995:
        return R.from_quat(q1 + t * (q2 - q1))
    
    # Perform SLERP
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = s0 * q1 + s1 * q2
    
    return R.from_quat(q)

def get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    src_fps = float(smplx_data["mocap_frame_rate"].item())
    if src_fps <= 0 or tgt_fps <= 0:
        raise ValueError(f"FPS values must be positive, got src={src_fps}, target={tgt_fps}")
    num_frames = smplx_data["pose_body"].shape[0]
    global_orient = smplx_output.global_orient.detach().cpu().numpy().reshape(num_frames, 3)
    full_body_pose = smplx_output.full_pose.detach().cpu().numpy().reshape(num_frames, -1, 3)
    joints = smplx_output.joints.detach().cpu().numpy().reshape(num_frames, -1, 3)
    joint_names = JOINT_NAMES[: joints.shape[1]]
    kinematic_joint_names = joint_names[: len(body_model.parents)]
    parents = body_model.parents
    
    if num_frames == 1:
        aligned_fps = tgt_fps
    elif not np.isclose(tgt_fps, src_fps):
        # Sample by time so non-integer ratios such as 120 FPS -> 50 FPS keep
        # the requested output rate instead of silently falling back to 60 FPS.
        new_num_frames = max(1, int(np.floor((num_frames - 1) * tgt_fps / src_fps)) + 1)
        original_time = np.arange(num_frames)
        target_time = np.arange(new_num_frames) * (src_fps / tgt_fps)
        
        # Interpolate global orientation using SLERP
        global_orient_interp = []
        for i in range(len(target_time)):
            t = target_time[i]
            idx1 = int(np.floor(t))
            idx2 = min(idx1 + 1, num_frames - 1)
            alpha = t - idx1
            
            rot1 = R.from_rotvec(global_orient[idx1])
            rot2 = R.from_rotvec(global_orient[idx2])
            interp_rot = slerp(rot1, rot2, alpha)
            global_orient_interp.append(interp_rot.as_rotvec())
        global_orient = np.stack(global_orient_interp, axis=0)
        
        # Interpolate full body pose using SLERP
        full_body_pose_interp = []
        for i in range(full_body_pose.shape[1]):  # For each joint
            joint_rots = []
            for j in range(len(target_time)):
                t = target_time[j]
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1
                
                rot1 = R.from_rotvec(full_body_pose[idx1, i])
                rot2 = R.from_rotvec(full_body_pose[idx2, i])
                interp_rot = slerp(rot1, rot2, alpha)
                joint_rots.append(interp_rot.as_rotvec())
            full_body_pose_interp.append(np.stack(joint_rots, axis=0))
        full_body_pose = np.stack(full_body_pose_interp, axis=1)
        
        # Interpolate joint positions using linear interpolation
        joints_interp = []
        for i in range(joints.shape[1]):  # For each joint
            for j in range(3):  # For each coordinate
                interp_func = interp1d(original_time, joints[:, i, j], kind='linear')
                joints_interp.append(interp_func(target_time))
        joints = np.stack(joints_interp, axis=1).reshape(new_num_frames, -1, 3)
        
        aligned_fps = tgt_fps
    else:
        aligned_fps = src_fps
        
    smplx_data_frames = []
    for curr_frame in range(len(global_orient)):
        result = {}
        single_global_orient = global_orient[curr_frame]
        single_full_body_pose = full_body_pose[curr_frame]
        single_joints = joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(kinematic_joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))

        # These are measured surface landmarks, not rigid bodies.  Preserve
        # their positions while deliberately leaving orientation undefined.
        for landmark in ("left_ankle", "right_ankle", "left_big_toe", "right_big_toe",
                         "left_small_toe", "right_small_toe", "left_heel", "right_heel"):
            if landmark in joint_names:
                index = joint_names.index(landmark)
                result[landmark] = (single_joints[index], None)
        smplx_data_frames.append(result)

    return smplx_data_frames, aligned_fps



def get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    src_fps = smplx_data["mocap_frame_rate"].item()
    frame_skip = int(src_fps / tgt_fps)
    num_frames = smplx_data["pose_body"].shape[0]
    global_orient = smplx_output.global_orient.squeeze()
    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)
    joints = smplx_output.joints.detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents
    
    if tgt_fps < src_fps:
        # perform fps alignment with proper interpolation
        new_num_frames = num_frames // frame_skip
        
        # Create time points for interpolation
        original_time = np.arange(num_frames)
        target_time = np.linspace(0, num_frames-1, new_num_frames)
        
        # Interpolate global orientation using SLERP
        global_orient_interp = []
        for i in range(len(target_time)):
            t = target_time[i]
            idx1 = int(np.floor(t))
            idx2 = min(idx1 + 1, num_frames - 1)
            alpha = t - idx1
            
            rot1 = R.from_rotvec(global_orient[idx1])
            rot2 = R.from_rotvec(global_orient[idx2])
            interp_rot = slerp(rot1, rot2, alpha)
            global_orient_interp.append(interp_rot.as_rotvec())
        global_orient = np.stack(global_orient_interp, axis=0)
        
        # Interpolate full body pose using SLERP
        full_body_pose_interp = []
        for i in range(full_body_pose.shape[1]):  # For each joint
            joint_rots = []
            for j in range(len(target_time)):
                t = target_time[j]
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1
                
                rot1 = R.from_rotvec(full_body_pose[idx1, i])
                rot2 = R.from_rotvec(full_body_pose[idx2, i])
                interp_rot = slerp(rot1, rot2, alpha)
                joint_rots.append(interp_rot.as_rotvec())
            full_body_pose_interp.append(np.stack(joint_rots, axis=0))
        full_body_pose = np.stack(full_body_pose_interp, axis=1)
        
        # Interpolate joint positions using linear interpolation
        joints_interp = []
        for i in range(joints.shape[1]):  # For each joint
            for j in range(3):  # For each coordinate
                interp_func = interp1d(original_time, joints[:, i, j], kind='linear')
                joints_interp.append(interp_func(target_time))
        joints = np.stack(joints_interp, axis=1).reshape(new_num_frames, -1, 3)
        
        aligned_fps = len(global_orient) / num_frames * src_fps
    else:
        aligned_fps = tgt_fps
        
    smplx_data_frames = []
    for curr_frame in range(len(global_orient)):
        result = {}
        single_global_orient = global_orient[curr_frame]
        single_full_body_pose = full_body_pose[curr_frame]
        single_joints = joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))


        smplx_data_frames.append(result)
        
    # add correct rotations
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    for result in smplx_data_frames:
        for joint_name in result.keys():
            orientation = utils.quat_mul(rotation_quat, result[joint_name][1])
            position = result[joint_name][0] @ rotation_matrix.T
            result[joint_name] = (position, orientation)
            

    return smplx_data_frames, aligned_fps
