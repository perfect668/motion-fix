import numpy as np

from general_motion_retargeting.terrain_contact_utils import build_terrain_contact_schedule
from general_motion_retargeting.terrain_geometry import BoxPrimitive, SceneTransform, TerrainField


def _frame(left_foot_z, right_foot_z):
    frame = {}
    for side, title, y, z in (("left", "Left", 0.2, left_foot_z), ("right", "Right", -0.2, right_foot_z)):
        frame[f"{title}Foot"] = np.array([0.0 if side == "left" else 1.0, y, z])
        frame[f"{title}ToeBase"] = frame[f"{title}Foot"] + np.array([0.2, 0, 0])
        frame[f"{title}Hand"] = np.array([0, y, 1.0])
        frame[f"{title}Leg"] = frame[f"{title}Foot"] + np.array([0, 0, 0.4])
    return frame


def test_floor_and_step_contacts_keep_independent_surfaces():
    terrain = TerrainField([BoxPrimitive("step", [0, 0, 0.25], [0.4, 0.4, 0.25], np.eye(3))], floor_z=0.0)
    frames = [_frame(0.50, 0.0) for _ in range(10)]
    schedule = build_terrain_contact_schedule(
        frames, terrain, SceneTransform(np.eye(3), 1.0, np.zeros(3)), {}, 50.0,
        {"contact_enter_distance": 0.03, "contact_exit_distance": 0.055, "contact_blend_frames": 3},
    )
    contacts = schedule[-1]["contacts"]
    assert contacts["left_toe"]["surface_id"] == "step:z+"
    assert contacts["right_toe"]["surface_id"] == "floor"
    assert contacts["left_heel"]["surface_id"] == "step:z+"


def test_surface_episode_hysteresis_is_stable_at_edge():
    terrain = TerrainField([BoxPrimitive("step", [0, 0, 0.25], [0.5, 0.5, 0.25], np.eye(3))], floor_z=0.0)
    frames = []
    for x in [0.45, 0.49, 0.501, 0.499, 0.502, 0.498]:
        frame = _frame(0.5, 0.0)
        frame["LeftFoot"][0] = x - 0.2
        frame["LeftToeBase"][0] = x
        frames.append(frame)
    schedule = build_terrain_contact_schedule(
        frames, terrain, SceneTransform(np.eye(3), 1.0, np.zeros(3)), {}, 50.0,
        {"surface_switch_hysteresis": 0.02, "contact_blend_frames": 1},
    )
    ids = [item["contacts"]["left_toe"]["surface_id"] for item in schedule]
    assert len(set(ids)) == 1
