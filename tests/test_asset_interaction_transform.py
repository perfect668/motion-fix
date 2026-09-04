import numpy as np

from general_motion_retargeting.asset_interaction_transform import AssetInteractionTransform


def test_asset_transform_points_and_inverse_transpose_normals():
    transform = AssetInteractionTransform(
        translation=[1.0, 2.0, 3.0], yaw=np.pi / 2,
        scale_axis_1=2.0, scale_axis_2=1.0, scale_up=0.5,
    )
    point = transform.transform_points([[1.0, 0.0, 0.0]])[0]
    np.testing.assert_allclose(point, [1.0, 4.0, 3.0])
    normal = transform.transform_normals([[0.0, 0.0, 1.0]])[0]
    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0])


def test_asset_transform_default_is_identity():
    transform = AssetInteractionTransform()
    np.testing.assert_allclose(transform.transform_points([[1, 2, 3]]), [[1, 2, 3]])
    np.testing.assert_allclose(transform.transform_normals([[0, 0, 1]]), [[0, 0, 1]])


def test_asset_transform_rejects_out_of_bounds_scale():
    try:
        AssetInteractionTransform.from_config({"scale_axis_1": 1.2, "scale_max": 1.15})
    except ValueError:
        return
    raise AssertionError("out-of-bounds asset scale was accepted")


def test_asset_transform_coordinate_descent_respects_bounds():
    transform = AssetInteractionTransform()

    def objective(value):
        return float(np.sum((value.translation - np.array([0.04, -0.02, 0.01])) ** 2)
                     + (value.scale_axis_1 - 1.04) ** 2)

    result, score = transform.optimize(
        objective, max_iterations=4, translation_step=0.02, scale_step=0.03,
        scale_min=0.95, scale_max=1.05,
    )
    assert score < objective(transform)
    assert 0.95 <= result.scale_axis_1 <= 1.05
