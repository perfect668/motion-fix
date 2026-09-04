
import pickle

def load_robot_motion(motion_file):
    """
    Load robot motion data from a pickle file.
    """
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)
        motion_fps = float(motion_data.get("fps", 50.0) or 50.0)
        motion_root_pos = motion_data["root_pos"]
        motion_root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]] # from xyzw to wxyz
        motion_dof_pos = motion_data["dof_pos"]
        # V4 exports the canonical qpos fields and does not need the legacy
        # derived-body arrays used by older viewers. Keep the return contract
        # stable for callers that still unpack these values.
        motion_local_body_pos = motion_data.get("local_body_pos")
        motion_link_body_list = motion_data.get(
            "link_body_list", motion_data.get("body_names", []))
    return motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list

