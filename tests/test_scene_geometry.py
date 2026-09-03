import mujoco as mj
import numpy as np

from general_motion_retargeting.scene_geometry import SceneGeometry, SceneObject, sample_mesh_surface
from general_motion_retargeting.scene_limits import AutomaticSceneCollisionLimit


def test_scene_sampling_and_pose_are_geometry_agnostic():
    obj = SceneObject("chair", np.array([[0., 0., 0.], [1., 0., 0.]]), pose=np.diag([2., 2., 2., 1.]))
    scene = SceneGeometry([obj], floor_z=0.0)
    assert np.allclose(obj.transformed_samples()[1], [2., 0., 0.])
    assert scene.interaction_samples(2).shape == (2, 3)


def test_triangle_surface_sampling_is_deterministic():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    a = sample_mesh_surface(v, [[0, 1, 2]], 8)
    b = sample_mesh_surface(v, [[0, 1, 2]], 8)
    np.testing.assert_allclose(a, b)
    assert np.all(a[:, 0] + a[:, 1] <= 1.0 + 1e-9)


def test_scene_collision_limit_accepts_combined_mujoco_model():
    xml = '''<mujoco><worldbody><geom name="floor" type="plane" size="1 1 0.1"/><body name="robot" pos="0 0 0.25"><freejoint/><geom name="robot_geom" type="sphere" size="0.1"/></body><body name="scene_chair"><geom name="chair_geom" type="box" pos="0 0 0" size="0.1 0.1 0.1"/></body></worldbody></mujoco>'''
    model = mj.MjModel.from_xml_string(xml)
    limit = AutomaticSceneCollisionLimit(model, {"activate_distance": 0.2, "scene_body_prefix": "scene_"})
    import mink
    configuration = mink.Configuration(model)
    assert model.geom("floor").id not in limit.robot_geoms
    assert model.geom("chair_geom").id in limit.scene_geoms
    assert model.geom("robot_geom").id in limit.robot_geoms
    limit.prepare_active_set(configuration)
    assert limit.active_pairs
    constraint = limit.compute_qp_inequalities(configuration, 0.02)
    assert constraint.G.shape[1] == model.nv
    measurement = limit.measure_current_distances(configuration)
    assert np.isfinite(measurement["minimum_scene_distance"])
    assert measurement["closest_scene_collision_pair"]["robot_geom"] == "robot_geom"
