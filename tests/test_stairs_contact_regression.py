"""Finite tread contacts: failures observed in the stairs_0037 delivery."""

import copy

import numpy as np

from general_motion_retargeting.terrain_contact_utils import (
    augment_mesh_contact_schedule, refresh_flat_foot,
)
from general_motion_retargeting.terrain_tasks import FootFrameTask, TerrainPointContactTask
from test_terrain_tasks import _configuration, _contact


def _mesh_schedule(points, channel="left_heel"):
    return [{"contacts": {channel: {
        **_contact("NONE", 0.0), "human_point_solver": np.asarray(p, dtype=float),
        "tangential_speed": 0.0,
    }}, "flat_foot": {"left": 0.0, "right": 0.0}} for p in points]


def _augment(schedule, vertices, faces, channel="left_heel"):
    return augment_mesh_contact_schedule(
        schedule, [{} for _ in schedule], np.asarray(vertices, dtype=float),
        np.asarray(faces), "stairs", np.eye(4),
        {"channels": [channel], "object_contact_distance": 0.05},
    )


def test_near_infinite_plane_is_not_contact_with_distant_triangle():
    vertices = [[0, 0, .5], [1, 0, .5], [0, 1, .5]]
    for channel in ("left_heel", "left_knee"):
        result = _augment(_mesh_schedule([[2, 0, .51]], channel), vertices, [[0, 1, 2]], channel)
        assert result[0]["contacts"][channel]["state"] == "NONE"


def test_hysteresis_releases_distant_old_triangle_for_nearby_new_tread():
    vertices = [[0, 0, .5], [1, 0, .5], [0, 1, .5],
                [2, 0, .52], [3, 0, .52], [2, 1, .52]]
    result = _augment(_mesh_schedule([[.2, .2, .51], [2.2, .2, .51]]),
                      vertices, [[0, 1, 2], [3, 4, 5]])
    assert result[1]["contacts"]["left_heel"]["surface_triangle_index"] == 1
    assert result[1]["contacts"]["left_heel"]["surface_distance"] < .011


def test_contact_at_triangle_edge_uses_full_distance_for_confidence():
    result = _augment(_mesh_schedule([[.2, -.03, .52]]),
                      [[0, 0, .5], [1, 0, .5], [0, 1, .5]], [[0, 1, 2]])
    c = result[0]["contacts"]["left_heel"]
    np.testing.assert_allclose(c["score"], 1 - np.hypot(.03, .02) / .05)


def test_mesh_contact_does_not_inherit_unrelated_floor_confidence():
    schedule = _mesh_schedule([[.2, .2, .54]])
    schedule[0]["contacts"]["left_heel"]["score"] = .99
    result = _augment(schedule, [[0, 0, .5], [1, 0, .5], [0, 1, .5]], [[0, 1, 2]])
    np.testing.assert_allclose(result[0]["contacts"]["left_heel"]["score"], .2)


def test_mesh_provider_refreshes_flat_foot_across_coplanar_triangles():
    frame = _mesh_schedule([[.2, .2, .51]])[0]
    frame["contacts"]["left_toe"] = {
        **copy.deepcopy(frame["contacts"]["left_heel"]),
        "human_point_solver": np.array([.8, .8, .51]),
    }
    augment_mesh_contact_schedule(
        [frame], [{}], np.array([[0, 0, .5], [1, 0, .5], [0, 1, .5], [1, 1, .5]]),
        np.array([[0, 1, 2], [1, 3, 2]]), "stairs", np.eye(4),
        {"channels": ["left_heel", "left_toe"], "object_contact_distance": .05},
    )
    assert frame["flat_foot"]["left"] > .79
    frame["contacts"]["left_toe"]["surface_point_solver"][2] += .15
    refresh_flat_foot(frame, {})
    assert frame["flat_foot"]["left"] == 0.0


def test_anchor_rebinds_on_new_tread_but_not_coplanar_triangle_seam():
    model, configuration = _configuration()
    task = TerrainPointContactTask(model, {"left_heel": {"sites": ["heel"]}}, 35, 18, .004)
    contact = {**_contact(), "object_id": "stairs", "surface_id": "stairs:face_0"}
    task.set_contacts(configuration, {"left_heel": contact})
    anchor = task.anchors["left_heel"].copy()
    seam = {**contact, "surface_id": "stairs:face_1", "surface_point_solver": np.array([.2, 0, 0])}
    task.set_contacts(configuration, {"left_heel": seam})
    np.testing.assert_allclose(task.anchors["left_heel"], anchor)
    step = {**seam, "surface_point_solver": np.array([.2, 0, .15])}
    task.set_contacts(configuration, {"left_heel": step})
    np.testing.assert_allclose(task.anchors["left_heel"], [.2, 0, .154])


def test_toe_support_is_not_airborne_and_true_flight_keeps_source_orientation():
    model, _ = _configuration()
    task = FootFrameTask(model, {"left": "foot"}, [0, 0, 1], .12)
    heel = {**_contact("NONE", 0), "human_foot_normal_solver": np.array([0, 1., 0])}
    toe = {**_contact(), "human_point_solver": np.array([.2, 0, 0])}
    task.set_contacts({"left_heel": heel, "left_toe": toe}, {"left": 0.0})
    assert task.targets["left"]["mode"] == "partial"
    assert task.targets["left"]["activation"] == .15
    np.testing.assert_allclose(task.targets["left"]["normal"], [0, 1, 0])
    task.set_contacts({"left_heel": heel, "left_toe": {**toe, "state": "NONE"}}, {"left": 0.0})
    np.testing.assert_allclose(task.targets["left"]["normal"], [0, 1, 0])
    assert task.targets["left"]["activation"] == .15
