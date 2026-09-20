"""Directed sole orientation and its actual IK correction, not just flags."""
import mink
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from general_motion_retargeting.terrain_tasks import FootFrameTask
from test_terrain_tasks import _configuration, _contact


def _task(model):
    task = FootFrameTask(model, {"left": "foot"}, [0, 0, 1], .12)
    heel = {**_contact(), "human_point_solver": np.array([-.08, 0, 0])}
    toe = {**_contact(), "human_point_solver": np.array([.08, 0, 0])}
    task.set_contacts({"left_heel": heel, "left_toe": toe}, {"left": 1.0})
    return task


def _rotate(configuration, axis, degrees):
    q = configuration.q.copy()
    quat = Rotation.from_euler(axis, degrees, degrees=True).as_quat()
    q[3:7] = quat[[3, 0, 1, 2]]
    configuration.update(q)


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("degrees", [90, 180])
def test_vertical_and_upside_down_soles_have_nonzero_angle_and_recover(axis, degrees):
    model, configuration = _configuration()
    task = _task(model)
    _rotate(configuration, axis, degrees)
    np.testing.assert_allclose(np.linalg.norm(task.compute_error(configuration)),
                               np.deg2rad(degrees), atol=1e-9)
    diagnostics = task.orientation_diagnostics(configuration)["left"]
    np.testing.assert_allclose(diagnostics["sole_target_angle_deg"], degrees)
    initial_position = configuration.q[:3].copy()
    for _ in range(60):
        velocity = mink.solve_ik(configuration, [task], .02, "daqp", damping=1e-5)
        configuration.integrate_inplace(velocity, .02)
    assert np.linalg.norm(task.compute_error(configuration)) < 1e-4
    np.testing.assert_allclose(configuration.q[:3], initial_position, atol=1e-10)


def test_rotation_jacobian_matches_mujoco_tangent_finite_difference():
    model, configuration = _configuration()
    task = _task(model)
    _rotate(configuration, "xyz", [70, -35, 48])
    q = configuration.q.copy()
    jac = task.compute_jacobian(configuration)
    numeric = np.zeros_like(jac)
    eps = 1e-6
    for i in range(model.nv):
        direction = np.eye(model.nv)[i]
        plus, minus = q.copy(), q.copy()
        mujoco.mj_integratePos(model, plus, direction, eps)
        mujoco.mj_integratePos(model, minus, direction, -eps)
        configuration.update(plus)
        first = task.compute_error(configuration)
        configuration.update(minus)
        numeric[:, i] = (first - task.compute_error(configuration)) / (2 * eps)
    configuration.update(q)
    np.testing.assert_allclose(jac, numeric, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("mode", ["flat", "partial", "airborne"])
def test_slope_and_source_tilt_are_preserved_without_flattening_every_phase(mode):
    model, configuration = _configuration()
    task = _task(model)
    rotation = Rotation.from_euler("y", 35, degrees=True).as_matrix()
    normal, forward = rotation[:, 2], rotation[:, 0]
    heel = {**_contact("STATIC" if mode == "flat" else "NONE"),
            "human_point_solver": -.08 * forward,
            "human_foot_normal_solver": normal,
            "human_foot_forward_solver": forward,
            "surface_normal_solver": normal if mode == "flat" else np.array([0., 0, 1])}
    toe = {**heel, "human_point_solver": .08 * forward,
           "state": "NONE" if mode == "airborne" else "STATIC"}
    task.set_contacts({"left_heel": heel, "left_toe": toe},
                      {"left": 1.0 if mode == "flat" else 0.0})
    assert task.targets["left"]["mode"] == mode
    _rotate(configuration, "y", 35)
    np.testing.assert_allclose(task.compute_error(configuration), 0.0, atol=1e-9)
    _rotate(configuration, "y", 0)
    assert np.linalg.norm(task.compute_error(configuration)) > .08


def test_airborne_upside_down_source_is_preserved_for_acrobatics():
    model, configuration = _configuration()
    task = _task(model)
    heel = {**_contact("NONE", 0),
            "human_foot_normal_solver": np.array([0., 0, -1]),
            "human_foot_forward_solver": np.array([1., 0, 0])}
    task.set_contacts({"left_heel": heel, "left_toe": {**heel,
        "human_point_solver": np.array([.2, 0, 0])}}, {"left": 0.0})
    _rotate(configuration, "x", 180)
    np.testing.assert_allclose(task.compute_error(configuration), 0.0, atol=1e-9)
    assert task.orientation_diagnostics(configuration)["left"]["mode"] == "airborne"
