import numpy as np

from general_motion_retargeting.terrain_native_geometry import TerrainPatchMap
from general_motion_retargeting.terrain_native_planner import (
    apply_terrain_native_plan,
    build_support_plan,
)


def _step_patch_map():
    vertices = np.array([
        [1.0, -0.5, 0.20],
        [2.0, -0.5, 0.20],
        [2.0, 0.5, 0.20],
        [1.0, 0.5, 0.20],
    ])
    # Reverse one triangle winding on purpose. Support semantics must not depend
    # on asset triangle order.
    faces = np.array([[0, 2, 1], [0, 3, 2]])
    patch_map = TerrainPatchMap.from_mesh(
        vertices, faces, np.eye(4), "stairs",
        {"support_normal_min_z": 0.65},
    )
    patch_map.add_horizontal_patch("floor", 0.0, [-2.0, -2.0], [3.0, 2.0])
    return patch_map


def _foot_points(x, z):
    heel = np.array([x - 0.08, 0.0, z])
    toe = np.array([x + 0.08, 0.0, z])
    return {
        "heel": heel,
        "toe": toe,
        "big_toe": toe + np.array([0.0, -0.04, 0.0]),
        "small_toe": toe + np.array([0.0, 0.04, 0.0]),
    }


def _frame(left_x, left_z, right_x=0.2, right_z=0.01):
    left = _foot_points(left_x, left_z)
    right = _foot_points(right_x, right_z)
    frame = {}
    for side, points in (("left", left), ("right", right)):
        frame[f"{side}_heel"] = points["heel"]
        frame[f"{side}_toe"] = points["toe"]
        frame[f"{side}_big_toe"] = points["big_toe"]
        frame[f"{side}_small_toe"] = points["small_toe"]
    return frame


def test_support_patch_query_chooses_top_surface_and_ignores_winding():
    patch_map = _step_patch_map()
    hit = patch_map.support_at(
        np.array([1.5, 0.0, 0.23]),
        edge_margin=0.0,
        above_tolerance=0.04,
        max_gap=0.04,
    )
    assert hit is not None
    patch, surface, gap = hit
    assert patch.patch_id.startswith("stairs:support_")
    np.testing.assert_allclose(surface[2], 0.20)
    np.testing.assert_allclose(patch.normal, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(gap, 0.03)


def test_whole_sequence_planner_assigns_floor_swing_and_next_tread():
    patch_map = _step_patch_map()
    trajectory = [
        (0.20, 0.01), (0.20, 0.01), (0.20, 0.01), (0.20, 0.01),
        (0.45, 0.07), (0.70, 0.14), (0.95, 0.24), (1.20, 0.28),
        (1.40, 0.21), (1.40, 0.21), (1.40, 0.21), (1.40, 0.21),
    ]
    frames = [_frame(x, z) for x, z in trajectory]
    plan = build_support_plan(
        frames,
        patch_map,
        fps=10.0,
        config={
            "source_contact_distance": 0.04,
            "source_contact_speed": 5.0,
            "minimum_stance_frames": 3,
            "contact_gap_fill_frames": 1,
            "swing_clearance": 0.05,
        },
    )

    assert all(plan[i]["left"]["mode"] == "stance" for i in range(4))
    assert all(plan[i]["left"]["patch_id"] == "floor" for i in range(4))
    assert all(plan[i]["left"]["mode"] == "swing" for i in range(4, 8))
    landing_ids = {plan[i]["left"]["landing_patch_id"] for i in range(4, 8)}
    assert len(landing_ids) == 1
    landing_id = next(iter(landing_ids))
    assert landing_id.startswith("stairs:support_")
    assert all(plan[i]["left"]["mode"] == "stance" for i in range(8, 12))
    assert all(plan[i]["left"]["patch_id"] == landing_id for i in range(8, 12))
    assert min(plan[i]["left"]["clearance_floor_z"] for i in range(4, 8)) > 0.20


def test_schedule_uses_planned_patch_instead_of_robot_proximity_contact():
    patch_map = _step_patch_map()
    frames = [_frame(1.4, 0.21) for _ in range(4)]
    schedule = [
        {
            "contacts": {
                "left_heel": {"state": "NONE", "score": 0.0},
                "left_toe": {"state": "NONE", "score": 0.0},
                "right_heel": {"state": "NONE", "score": 0.0},
                "right_toe": {"state": "NONE", "score": 0.0},
            },
            "flat_foot": {"left": 0.0, "right": 0.0},
        }
        for _ in frames
    ]
    apply_terrain_native_plan(
        schedule, frames, patch_map, fps=30.0,
        config={"source_contact_distance": 0.04, "source_contact_speed": 5.0},
    )
    item = schedule[0]["contacts"]["left_heel"]
    assert item["state"] == "STATIC"
    assert item["surface_type"] == "support_patch"
    assert item["surface_id"].startswith("stairs:support_")
    assert schedule[0]["flat_foot"]["left"] == 1.0


def test_duplicate_face_vertices_merge_into_one_support_patch():
    # Same physical square, but each triangle owns independent vertex indices
    # as frequently happens after USD/OBJ export.
    vertices = np.array([
        [0.0, 0.0, 0.4], [1.0, 0.0, 0.4], [1.0, 1.0, 0.4],
        [0.0, 0.0, 0.4], [1.0, 1.0, 0.4], [0.0, 1.0, 0.4],
    ])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    patch_map = TerrainPatchMap.from_mesh(
        vertices, faces, np.eye(4), "stairs",
        {"support_normal_min_z": 0.65},
    )
    assert len(patch_map.patches) == 1
    assert len(patch_map.patches[0].face_indices) == 2
