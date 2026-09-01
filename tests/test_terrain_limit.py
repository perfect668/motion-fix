import mujoco
import mink
import numpy as np

from general_motion_retargeting.terrain_geometry import TerrainField
from general_motion_retargeting.terrain_limits import TerrainNonPenetrationLimit


MODEL_XML = """
<mujoco><worldbody><body name="base" pos="0 0 0.4"><freejoint/>
  <geom type="box" size="0.1 0.1 0.1" mass="1"/>
  <site name="guard" pos="0.05 0 -0.1" size="0.005"/>
</body></worldbody></mujoco>
"""


def _configuration():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    return model, mink.Configuration(model)


def test_terrain_limit_normal_jacobian_matches_finite_difference():
    model, configuration = _configuration()
    limit = TerrainNonPenetrationLimit(
        model, TerrainField([], floor_z=0.0), {}, {"guard": 0.008}, {}, {}, {},
        {"adaptive_activation": False},
    )
    limit.prepare_active_set(configuration, 0.02, {})
    constraint = limit.compute_qp_inequalities(configuration, 0.02)
    direction = np.linspace(-0.2, 0.3, model.nv)
    qpos = configuration.q.copy()
    epsilon = 1e-7
    perturbed = qpos.copy()
    mujoco.mj_integratePos(model, perturbed, direction, epsilon)
    configuration.update(perturbed)
    plus = limit.measure_current_slacks(configuration)["guard"]["signed_distance"]
    configuration.update(qpos)
    base = limit.measure_current_slacks(configuration)["guard"]["signed_distance"]
    finite_difference = (plus - base) / epsilon
    np.testing.assert_allclose(-constraint.G[0] @ direction, finite_difference, rtol=2e-5, atol=2e-6)


def test_floor_constraint_direction_matches_vertical_ground_limit():
    model, configuration = _configuration()
    limit = TerrainNonPenetrationLimit(
        model, TerrainField([], floor_z=0.0), {}, {"guard": 0.008}, {}, {}, {},
        {"adaptive_activation": False},
    )
    limit.prepare_active_set(configuration, 0.02, {})
    constraint = limit.compute_qp_inequalities(configuration, 0.02)
    np.testing.assert_allclose(constraint.G[0, :3], [0.0, 0.0, -1.0], atol=1e-9)
    assert np.isclose(constraint.h[0], 0.292)
