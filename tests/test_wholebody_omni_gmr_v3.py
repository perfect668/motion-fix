import json
from pathlib import Path

import mujoco as mj
import mink
import numpy as np
from scipy.spatial.transform import Rotation

from general_motion_retargeting.terrain_geometry import BoxPrimitive, TerrainField
from general_motion_retargeting.wholebody_omni_gmr_v3 import (
    AutomaticMeshTerrainLimit,
    AutomaticSelfCollisionLimit,
    InteractionLaplacianTask,
    TorsoPelvisCoherenceTask,
    WholeBodyOmniGMRV3,
    sample_terrain_surface_pool,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "general_motion_retargeting/ik_configs/holosoma_to_ne01_wholebody_omni_gmr_v3.json"


def test_v3_config_uses_native_holosoma_semantics_without_smpl_tables():
    config = json.loads(CONFIG.read_text())
    text = CONFIG.read_text()
    assert "smplx_to_ne01" not in text
    assert "ik_match_table" not in text
    assert "human_scale_table" not in text
    mapping = config["semantic_points"]
    assert mapping["left_hip"]["source"] == "LeftUpLeg"
    assert mapping["left_hip"]["robot"]["body"] == "HIP_PITCH_L_LINK"
    assert mapping["left_foot"]["robot"]["body"] == "ANKLE_PITCH_L_LINK"
    assert mapping["left_shoulder"]["robot"]["body"] == "SHOULDER_ROLL_L_LINK"


def test_interaction_laplacian_is_translation_invariant():
    rng = np.random.default_rng(7)
    vertices = rng.normal(size=(20, 3))
    matrix = InteractionLaplacianTask._laplacian(vertices)
    translated = vertices + np.array([2.0, -1.0, 0.7])
    np.testing.assert_allclose(matrix @ vertices, matrix @ translated, atol=1e-10)
    np.testing.assert_allclose(matrix.sum(axis=1), 0.0, atol=1e-12)


def test_terrain_pool_depends_on_geometry_and_is_deterministic():
    terrain = TerrainField([
        BoxPrimitive("box", [0, 0, 0.25], [0.4, 0.3, 0.25], np.eye(3))
    ])
    reference = np.array([[-0.5, -0.5, 0.5], [0.5, 0.5, 1.2]])
    first = sample_terrain_surface_pool(terrain, reference, 48)
    second = sample_terrain_surface_pool(terrain, reference, 48)
    assert first.shape == (48, 3)
    np.testing.assert_allclose(first, second)
    assert np.any(np.isclose(first[:, 2], 0.5))
    assert np.any(np.isclose(first[:, 2], 0.0))


def test_automatic_collision_limit_discovers_mesh_shells_without_point_config():
    config = json.loads(CONFIG.read_text())
    xml = ROOT / config["robot_xml"]
    model = mj.MjModel.from_xml_path(str(xml))
    terrain = TerrainField()
    limit = AutomaticMeshTerrainLimit(model, terrain, config["terrain_nonpenetration"])
    assert "ANKLE_ROLL_L_LINK" in limit.shells
    assert "KNEE_PITCH_L_LINK" in limit.shells
    assert "HAND_YAW_L_LINK" in limit.shells
    assert "TORSO_LINK" in limit.shells
    assert all(item["proxies"].shape[1] == 3 for item in limit.shells.values())


def test_self_collision_pairs_are_automatic_and_exclude_adjacent_links():
    config = json.loads(CONFIG.read_text())
    model = mj.MjModel.from_xml_path(str(ROOT / config["robot_xml"]))
    limit = AutomaticSelfCollisionLimit(model, config["self_collision"])
    assert limit.geom_pairs
    for first, second in limit.geom_pairs:
        body_first = int(model.geom_bodyid[first])
        body_second = int(model.geom_bodyid[second])
        assert body_first != body_second
        assert limit._body_distance(body_first, body_second) > 2


def test_v3_applies_ne01_torso_roll_safety_range():
    retargeter = WholeBodyOmniGMRV3(
        CONFIG,
        TerrainField(),
        np.array([
            [-0.5, -0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
        ]),
    )
    joint_id = retargeter.model.joint("TORSO_ROLL_JOINT").id
    np.testing.assert_allclose(retargeter.model.jnt_range[joint_id], [-0.3, 0.3])


def test_torso_pelvis_task_tracks_and_clamps_relative_orientation():
    config = json.loads(CONFIG.read_text())
    model = mj.MjModel.from_xml_path(str(ROOT / config["robot_xml"]))
    configuration = mink.Configuration(model)
    task = TorsoPelvisCoherenceTask(model, config["torso_pelvis_coherence"])
    pelvis = Rotation.identity().as_quat(scalar_first=True)
    chest = Rotation.from_euler("zyx", [0.2, 0.0, 0.6]).as_quat(scalar_first=True)
    task.set_target(configuration, pelvis, chest)
    assert np.isclose(task.targets["waist_yaw"], 0.2)
    assert np.isclose(task.targets["torso_roll"], 0.3)

    chest = Rotation.from_euler("zyx", [-0.2, 0.0, -0.6]).as_quat(scalar_first=True)
    task.set_target(configuration, pelvis, chest)
    assert -0.2 < task.targets["waist_yaw"] < 0.2
    assert -0.3 < task.targets["torso_roll"] < 0.3
