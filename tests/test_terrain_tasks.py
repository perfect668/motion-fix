import mujoco
import mink
import numpy as np

from general_motion_retargeting.terrain_tasks import (
    TerrainFootOrientationTask,
    TerrainFootTemporalTask,
    TerrainPointContactTask,
)


MODEL_XML = """
<mujoco><worldbody><body name="foot" pos="0 0 0.2"><freejoint/>
  <geom type="box" size="0.1 0.05 0.02" mass="1"/>
  <site name="heel" pos="-0.08 0 -0.02" size="0.005"/>
  <site name="toe" pos="0.08 0 -0.02" size="0.005"/>
</body></worldbody></mujoco>
"""


def _configuration():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    return model, mink.Configuration(model)


def _contact(state="STATIC", score=1.0, normal=(0, 0, 1)):
    return {
        "state": state,
        "score": score,
        "surface_normal_solver": np.asarray(normal, dtype=float),
        "surface_point_solver": np.zeros(3),
        "human_point_solver": np.zeros(3),
        "surface_id": "floor",
    }


def test_inactive_terrain_point_task_has_zero_error_and_jacobian():
    model, configuration = _configuration()
    task = TerrainPointContactTask(model, {"left_palm": {"sites": ["heel"]}}, 80, 25, 0.004)
    task.set_contacts(configuration, {"left_palm": _contact("NONE", 0.0)})
    np.testing.assert_allclose(task.compute_error(configuration), 0.0)
    np.testing.assert_allclose(task.compute_jacobian(configuration), 0.0)


def test_static_anchors_while_sliding_tracks_human_tangent_target():
    model, configuration = _configuration()
    task = TerrainPointContactTask(model, {"left_palm": {"sites": ["heel"]}}, 80, 25, 0.004)
    static = _contact("STATIC")
    static["human_point_solver"] = np.array([1.0, 0.0, 0.0])
    task.set_contacts(configuration, {"left_palm": static})
    static_error = task.compute_error(configuration)
    sliding = {**static, "state": "SLIDING"}
    task.set_contacts(configuration, {"left_palm": sliding})
    sliding_error = task.compute_error(configuration)
    np.testing.assert_allclose(static_error[1:], 0.0, atol=1e-12)
    assert np.linalg.norm(sliding_error[1:]) > 0.5


def test_foot_orientation_residual_is_zero_for_matching_horizontal_normal():
    model, configuration = _configuration()
    task = TerrainFootOrientationTask(model, {"left": "foot"}, [0, 0, 1], 0.12)
    contacts = {"left_heel": _contact(), "left_toe": _contact()}
    task.set_contacts(contacts, {"left": 1.0})
    np.testing.assert_allclose(task.compute_error(configuration), 0.0, atol=1e-12)
    contacts["left_heel"]["surface_normal_solver"] = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    contacts["left_toe"]["surface_normal_solver"] = contacts["left_heel"]["surface_normal_solver"]
    task.set_contacts(contacts, {"left": 1.0})
    assert np.linalg.norm(task.compute_error(configuration)) > 0.5


def test_temporal_task_only_sticks_static_contacts():
    model, configuration = _configuration()
    specs = {
        "left_heel": {"sites": ["heel"]}, "left_toe": {"sites": ["toe"]},
        "right_heel": {"sites": ["heel"]}, "right_toe": {"sites": ["toe"]},
    }
    task = TerrainFootTemporalTask(model, specs, 0.02, 0.04)
    contacts = {name: _contact("SLIDING") for name in specs}
    task.begin_frame(configuration, contacts)
    np.testing.assert_allclose(task.compute_error(configuration), 0.0)
    np.testing.assert_allclose(task.compute_jacobian(configuration), 0.0)
