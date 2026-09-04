import numpy as np
import importlib.util
from pathlib import Path

from general_motion_retargeting.terrain_contact_utils import (
    augment_mesh_contact_schedule,
    build_terrain_contact_schedule,
)
from general_motion_retargeting.terrain_geometry import BoxPrimitive, SceneTransform, TerrainField

_spec = importlib.util.spec_from_file_location(
    "grail_v4_entry", Path(__file__).parents[1] / "scripts" / "grail_to_robot_wholebody_v4.py"
)
_grail = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_grail)


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


def test_generic_mesh_provider_keeps_surface_anchor_metadata():
    terrain = TerrainField([], floor_z=0.0)
    frame = _frame(0.0, 0.0)
    schedule = build_terrain_contact_schedule(
        [frame], terrain, SceneTransform(np.eye(3), 1.0, np.zeros(3)), {}, 50.0,
        {"contact_blend_frames": 1},
    )
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    schedule[0]["contacts"]["left_butt"]["human_point_solver"] = np.array([0.2, 0.2, 0.01])
    result = augment_mesh_contact_schedule(
        schedule, [frame],
        vertices, np.array([[0, 1, 2]]), "surface", np.eye(4),
        {"channels": ["left_butt"], "object_contact_distance": 0.05},
    )
    contact = result[0]["contacts"]["left_butt"]
    assert contact["object_id"] == "surface"
    assert contact["surface_triangle_index"] == 0
    np.testing.assert_allclose(np.asarray(contact["surface_barycentric"]), [0.6, 0.2, 0.2])


def test_generic_mesh_provider_holds_surface_at_edge():
    frame = _frame(0.0, 0.0)
    terrain = TerrainField([], floor_z=0.0)
    schedule = build_terrain_contact_schedule(
        [frame, frame], terrain, SceneTransform(np.eye(3), 1.0, np.zeros(3)), {}, 50.0,
        {"contact_blend_frames": 1},
    )
    # Two coplanar triangles with distinct IDs meet at x=0.5.
    vertices = np.array([
        [0, 0, 0], [0.5, 0, 0], [0, 1, 0],
        [0.5, 0, 0], [1, 0, 0], [1, 1, 0],
    ], dtype=float)
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    for i, x in enumerate((0.49, 0.51)):
        schedule[i]["contacts"]["left_butt"]["human_point_solver"] = np.array([x, 0.2, 0.01])
    result = augment_mesh_contact_schedule(
        schedule, [frame, frame], vertices, faces, "surface", np.eye(4),
        {"channels": ["left_butt"], "object_contact_distance": 0.05,
         "surface_switch_hysteresis": 0.03},
    )
    assert result[0]["contacts"]["left_butt"]["surface_id"] == result[1]["contacts"]["left_butt"]["surface_id"]


def test_point_calibration_does_not_move_solver_floor():
    terrain = TerrainField([], floor_z=0.0)
    frames = [_frame(0.20, 0.20) for _ in range(3)]
    scene = SceneTransform(np.eye(3), 1.0, np.zeros(3))
    human = SceneTransform(np.eye(3), 1.0, np.array([0.0, 0.0, -0.20]))
    schedule = build_terrain_contact_schedule(
        frames, terrain, scene, {}, 50.0,
        {"contact_blend_frames": 1}, point_transform=human,
    )
    item = schedule[-1]["contacts"]["left_heel"]
    np.testing.assert_allclose(item["human_point_solver"][2], 0.0)
    np.testing.assert_allclose(item["surface_point_solver"][2], 0.0)
    np.testing.assert_allclose(item["signed_distance"], 0.0)


def test_stair_support_surface_prefers_highest_upward_face():
    triangles = np.asarray([
        [[0.0, 0.0, 0.25], [1.0, 0.0, 0.25], [0.0, 1.0, 0.25]],
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.50], [2.0, 0.0, 0.50], [1.0, 1.0, 0.50]],
    ])
    normals = np.asarray([[0, 0, 1], [1, 0, 0], [0, 0, 1]], dtype=float)
    centers = triangles.mean(axis=1)
    result = _grail._highest_support_surface(
        np.array([0.2, 0.2, 0.30]), triangles, centers, normals,
    )
    assert result is not None
    index, surface, normal, signed = result
    assert index == 0
    np.testing.assert_allclose(normal, [0, 0, 1])
    np.testing.assert_allclose(surface[2], 0.25)
    np.testing.assert_allclose(signed, 0.05)


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


def test_foot_frame_uses_solver_rotation_for_all_axes():
    frame = _frame(0.0, 0.0)
    frame.update({
        "left_hip": np.array([0.0, 0.2, 1.0]),
        "right_hip": np.array([0.0, -0.2, 1.0]),
        "pelvis": np.array([0.0, 0.0, 0.9]),
        "spine3": np.array([0.0, 0.0, 1.8]),
        "left_big_toe": frame["LeftToeBase"] + np.array([0.0, 0.04, 0.0]),
        "left_small_toe": frame["LeftToeBase"] - np.array([0.0, 0.04, 0.0]),
        "right_big_toe": frame["RightToeBase"] + np.array([0.0, 0.04, 0.0]),
        "right_small_toe": frame["RightToeBase"] - np.array([0.0, 0.04, 0.0]),
    })
    angle = np.pi / 2.0
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    schedule = build_terrain_contact_schedule(
        [frame] * 3, TerrainField(), SceneTransform(rotation, 1.0, np.zeros(3)), {}, 50.0,
        {"contact_enter_distance": 0.03, "contact_exit_distance": 0.055, "contact_blend_frames": 1},
    )
    item = schedule[-1]["contacts"]["left_heel"]
    forward = item["human_foot_forward_solver"]
    normal = item["human_foot_normal_solver"]
    np.testing.assert_allclose(np.linalg.norm(forward), 1.0)
    np.testing.assert_allclose(np.linalg.norm(normal), 1.0)
    np.testing.assert_allclose(float(forward @ normal), 0.0, atol=1e-7)
    body_up_solver = rotation @ np.array([0.0, 0.0, 0.9])
    assert float(normal @ body_up_solver) > 0.0


def test_foot_frame_recovers_normal_after_turn_degeneracy():
    first = _frame(0.0, 0.0)
    second = _frame(0.0, 0.0)
    for frame in (first, second):
        frame.update({
            "left_ankle": np.array([0.0, 0.2, 0.0]),
            "right_ankle": np.array([1.0, -0.2, 0.0]),
            "left_hip": np.array([0.0, 0.2, 1.0]),
            "right_hip": np.array([0.0, -0.2, 1.0]),
            "pelvis": np.array([0.0, 0.0, 0.9]),
            "spine3": np.array([0.0, 0.0, 1.8]),
        })
    first["LeftToeBase"] = first["left_ankle"] + np.array([1.0, 0.0, 0.0])
    second["LeftToeBase"] = second["left_ankle"] + np.array([0.0, 1.0, 0.0])
    schedule = build_terrain_contact_schedule(
        [first, second], TerrainField(), SceneTransform(np.eye(3), 1.0, np.zeros(3)), {}, 50.0,
        {"contact_blend_frames": 1},
    )
    item = schedule[-1]["contacts"]["left_heel"]
    forward = item["human_foot_forward_solver"]
    normal = item["human_foot_normal_solver"]
    lateral = np.cross(normal, forward)
    assert np.isclose(np.linalg.norm(normal), 1.0)
    assert np.isclose(np.linalg.norm(lateral), 1.0)
    np.testing.assert_allclose(forward @ normal, 0.0, atol=1e-7)
