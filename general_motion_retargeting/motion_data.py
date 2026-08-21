import numpy as np
import torch

from .kinematics_model import KinematicsModel
from .params import ROBOT_XML_DICT


def resolve_torch_device(device):
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "cuda:0"
    return device


def enrich_robot_motion_with_fk(motion_data, robot, device="auto"):
    resolved_device = resolve_torch_device(device)
    kinematics_model = KinematicsModel(str(ROBOT_XML_DICT[robot]), device=resolved_device)
    dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float32)
    frame_count = dof_pos.shape[0]
    root_pos = torch.zeros((frame_count, 3), device=resolved_device)
    root_rot = torch.zeros((frame_count, 4), device=resolved_device)
    root_rot[:, -1] = 1.0
    with torch.no_grad():
        local_body_pos, _ = kinematics_model.forward_kinematics(
            root_pos,
            root_rot,
            torch.from_numpy(dof_pos).to(device=resolved_device),
        )
    enriched_motion = dict(motion_data)
    enriched_motion["local_body_pos"] = local_body_pos.detach().cpu().numpy()
    enriched_motion["link_body_list"] = list(kinematics_model.body_names)
    return enriched_motion
