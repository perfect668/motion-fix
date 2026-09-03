import numpy as np

from general_motion_retargeting.terrain_geometry import BoxPrimitive, SceneTransform, TerrainField


def test_floor_signed_distance_and_normal():
    terrain = TerrainField([], floor_z=0.0)
    above = terrain.nearest_surface(np.array([1.0, 2.0, 0.4]))
    below = terrain.nearest_surface(np.array([1.0, 2.0, -0.1]))
    assert np.isclose(above.signed_distance, 0.4)
    assert np.isclose(below.signed_distance, -0.1)
    np.testing.assert_allclose(above.normal, [0.0, 0.0, 1.0])


def test_aabb_top_side_and_inside_queries():
    box = BoxPrimitive("step", [0, 0, 0.5], [1, 1, 0.5], np.eye(3), "aabb")
    terrain = TerrainField([box], floor_z=None)
    top = terrain.nearest_surface([0, 0, 1.2])
    side = terrain.nearest_surface([1.2, 0, 0.5])
    inside = terrain.nearest_surface([0, 0, 0.8])
    assert top.surface_id == "step:z+"
    assert side.surface_id == "step:x+"
    assert np.isclose(inside.signed_distance, -0.2)
    np.testing.assert_allclose(top.normal, [0, 0, 1])
    np.testing.assert_allclose(side.normal, [1, 0, 0])


def test_obb_query_and_scene_transform_roundtrip():
    angle = np.deg2rad(30.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    box = BoxPrimitive("rotated", [1, 2, 0.5], [0.5, 0.25, 0.5], rotation, "obb")
    terrain = TerrainField([box], floor_z=None)
    point = box.center + box.rotation @ np.array([0.7, 0.0, 0.0])
    hit = terrain.nearest_surface(point)
    assert hit.surface_id == "rotated:x+"
    assert np.isclose(hit.signed_distance, 0.2)
    np.testing.assert_allclose(hit.normal, rotation[:, 0])
    transform = SceneTransform(rotation, 0.7, [0.2, -0.3, 0.1])
    sample = np.array([[0.1, 0.2, 0.3], [-1.0, 2.0, 0.0]])
    np.testing.assert_allclose(transform.inverse().transform_points(transform.transform_points(sample)), sample)


def test_batch_matches_scalar_and_support_rejects_sidewall():
    box = BoxPrimitive("step", [0, 0, 0.5], [0.5, 0.5, 0.5], np.eye(3))
    terrain = TerrainField([box], floor_z=0.0)
    points = np.array([[0, 0, 1.1], [0.7, 0, 0.5], [2, 2, 0.2]])
    batch = terrain.nearest_surface_batch(points)
    scalar = [terrain.nearest_surface(point) for point in points]
    assert [item.surface_id for item in batch] == [item.surface_id for item in scalar]
    np.testing.assert_allclose([item.signed_distance for item in batch], [item.signed_distance for item in scalar])
    support = terrain.support_surface([0.7, 0.0, 0.5])
    assert support.surface_id == "floor"
    assert support.normal[2] > 0.6


def test_two_feet_can_use_floor_and_step():
    box = BoxPrimitive("step", [0, 0, 0.25], [0.5, 0.5, 0.25], np.eye(3))
    terrain = TerrainField([box], floor_z=0.0)
    assert terrain.support_surface([0, 0, 0.52]).surface_id == "step:z+"
    assert terrain.support_surface([1, 0, 0.02]).surface_id == "floor"


def test_vectorized_array_query_matches_scalar_fields():
    box = BoxPrimitive("step", [0, 0, 0.25], [0.5, 0.5, 0.25], np.eye(3))
    terrain = TerrainField([box], floor_z=0.0)
    points = np.array([[0.0, 0.0, 0.8], [0.7, 0.0, 0.5], [2.0, 2.0, -0.1]])
    arrays = terrain.nearest_surface_batch_arrays(points)
    scalar = terrain.nearest_surface_batch(points)
    np.testing.assert_allclose(arrays["signed_distance"], [item.signed_distance for item in scalar])
    np.testing.assert_allclose(arrays["closest_point"], [item.closest_point for item in scalar])
    np.testing.assert_allclose(arrays["normal"], [item.normal for item in scalar])
    assert arrays["surface_id"].tolist() == [item.surface_id for item in scalar]
