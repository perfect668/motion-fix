import numpy as np

from general_motion_retargeting.scene.loader import MeshSceneField


def _box_mesh():
    vertices = np.array([
        [-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.],
        [-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ])
    return vertices, faces


def test_mesh_support_uses_top_triangle_not_sidewall():
    field = MeshSceneField(*_box_mesh(), "box", floor_z=-2.0)
    hit = field.support_surface(np.array([0.2, 0.3, 1.4]))
    assert hit.supportable
    assert hit.normal[2] > 0.6
    assert np.isclose(hit.closest_point[2], 1.0)


def test_mesh_nearest_query_is_triangle_not_vertex_projection():
    field = MeshSceneField(*_box_mesh(), "box", floor_z=None)
    hit = field.nearest_surface(np.array([0.1, 0.2, 1.3]))
    assert np.allclose(hit.closest_point, [0.1, 0.2, 1.0], atol=1e-6)
    assert hit.surface_id.startswith("box:patch_")
