from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from general_motion_retargeting.core.schemas import MorphologyTargets
from general_motion_retargeting.morphology import (
    build_morphology_targets,
    map_semantic_frame,
    measure_robot_chain_lengths,
)


def _motion(frame: dict[str, np.ndarray]):
    return SimpleNamespace(
        human_height=1.70,
        canonical_named_positions=lambda: [frame],
    )


def test_chain_morphology_uses_measured_lengths_and_keeps_scene_metric():
    frame = {
        "pelvis": np.array([1.0, 2.0, 1.1]),
        "left_hip": np.array([1.0, 2.2, 1.1]),
        "left_knee": np.array([1.0, 2.2, 0.6]),
        "left_foot": np.array([1.0, 2.2, 0.1]),
        "left_toe": np.array([1.2, 2.2, 0.1]),
        "left_shoulder": np.array([1.0, 2.3, 1.6]),
        "left_elbow": np.array([1.0, 2.3, 1.3]),
        "left_wrist": np.array([1.0, 2.3, 1.0]),
    }
    targets = build_morphology_targets(
        _motion(frame), 1.316,
        {"mode": "chain"},
        robot_chain_lengths={
            "left_hip_offset": 0.1,
            "left_upper_leg": 0.25,
            "left_lower_leg": 0.25,
            "left_foot_length": 0.1,
            "left_shoulder_offset": 0.2,
            "left_upper_arm": 0.15,
            "left_lower_arm": 0.15,
        },
    )
    mapped = map_semantic_frame(frame, targets)
    np.testing.assert_allclose(mapped["pelvis"], frame["pelvis"])
    np.testing.assert_allclose(np.linalg.norm(mapped["left_hip"] - mapped["pelvis"]), 0.1)
    np.testing.assert_allclose(np.linalg.norm(mapped["left_knee"] - mapped["left_hip"]), 0.25)
    np.testing.assert_allclose(np.linalg.norm(mapped["left_foot"] - mapped["left_knee"]), 0.25)
    # The source scene/anchor is deliberately not part of map_semantic_frame.
    np.testing.assert_allclose(frame["pelvis"], [1.0, 2.0, 1.1])


def test_ne01_chain_measurement_uses_hand_as_wrist_alias():
    import json
    from pathlib import Path

    root = Path(__file__).parents[1]
    config = json.loads((root / "general_motion_retargeting/ik_configs/smplx_to_ne01_wholebody_omni_gmr_v5.json").read_text())
    lengths, provenance = measure_robot_chain_lengths(
        root / config["robot_xml"], config["semantic_points"]
    )
    assert lengths["left_upper_leg"] > 0.0
    assert lengths["left_lower_leg"] > 0.0
    assert lengths["left_lower_arm"] > 0.0
    assert "left_wrist" in provenance["semantic_points_measured"]


def test_ne01_profile_exports_all_scalar_joints_across_mujoco_enum_versions():
    import mujoco as mj
    from pathlib import Path
    from general_motion_retargeting.robot_profile import joint_names

    root = Path(__file__).parents[1]
    model = mj.MjModel.from_xml_path(str(root / "assets/ne01/ne01_desktop_assets_wholebody_omni_gmr_v2.xml"))
    names = joint_names(model)
    assert len(names) == 24
    assert "KNEE_PITCH_L_JOINT" in names


def test_recursive_mapping_does_not_mix_raw_and_mapped_proximals():
    frame = {
        "pelvis": np.zeros(3),
        "left_hip": np.array([0.0, 1.0, 0.0]),
        "left_knee": np.array([0.0, 2.0, 0.0]),
        "left_foot": np.array([0.0, 3.0, 0.0]),
        "left_toe": np.array([1.0, 3.0, 0.0]),
    }
    targets = MorphologyTargets(
        human_height=1.7, robot_height=1.316, mode="chain",
        chain_scales={
            "left_hip_offset": 0.5,
            "left_upper_leg": 0.5,
            "left_lower_leg": 0.5,
            "left_foot_length": 0.5,
        },
    )
    mapped = map_semantic_frame(frame, targets)
    np.testing.assert_allclose(mapped["left_foot"], [0.0, 1.5, 0.0])
    np.testing.assert_allclose(mapped["left_toe"], [0.5, 1.5, 0.0])


def test_known_ne01_pose_has_consistent_root_and_interaction_reference():
    import json
    from pathlib import Path
    from types import SimpleNamespace

    import mink
    import mujoco as mj
    from general_motion_retargeting.wholebody_omni_gmr_v5 import (
        _ContactTask,
        _InteractionTask,
        _RootTask,
        WholeBodyRetargetSolver,
    )

    root = Path(__file__).parents[1]
    config = json.loads((
        root
        / "general_motion_retargeting/ik_configs/smplx_to_ne01_wholebody_omni_gmr_v5.json"
    ).read_text())
    model = mj.MjModel.from_xml_path(str(root / config["robot_xml"]))
    configuration = mink.Configuration(model)
    contact_config = config["contact_tasks"]
    contact = _ContactTask(
        model, contact_config["robot_points"], contact_config
    )
    interaction = _InteractionTask(
        model, config["semantic_points"], np.empty((0, 3)),
        config["interaction_graph"],
    )
    reference = {
        name: point.value(configuration).copy()
        for name, point in interaction.points.items()
    }
    surface_z = float(np.median([
        contact.support_value(configuration, name, np.array([0.0, 0.0, 1.0]))[2]
        for name in ("left_heel", "left_toe", "right_heel", "right_toe")
    ]))
    contacts = [{
        "support_state": "SUPPORTED",
        "contacts": {
            name: {
                "state": "STATIC", "activation": 1.0,
                "surface_point_solver": [0.0, 0.0, surface_z],
                "surface_normal_solver": [0.0, 0.0, 1.0],
            }
            for name in ("left_heel", "left_toe", "right_heel", "right_toe")
        },
    }]
    pelvis_quaternion = configuration.data.xquat[model.body("base_link").id].copy()
    targets = [{
        name: (value.copy(), pelvis_quaternion.copy())
        for name, value in reference.items()
    }]
    solver = WholeBodyRetargetSolver.__new__(WholeBodyRetargetSolver)
    solver.model = model
    solver.configuration = configuration
    solver.contact = contact
    solver.root = SimpleNamespace(body_id=int(model.body("base_link").id))
    solver.terrain = SimpleNamespace(floor_z=surface_z)
    solver.root_policy = "support_aware"
    solver._root_vertical_scale = 1.0
    solver._root_reference_pelvis_z = None
    solver._root_reference_surface_z = None
    solver._robot_root_support_height = None
    solver.reference_metadata = {}

    references, aligned_targets = solver._prepare_robot_reference_motion(
        [reference], targets, contacts
    )
    np.testing.assert_allclose(references[0]["pelvis"], reference["pelvis"])
    interaction.set_target(references[0])
    np.testing.assert_allclose(
        interaction.compute_error(configuration), 0.0, atol=1e-10
    )
    root_task = _RootTask(model, "base_link", [1.0] * 6)
    root_task.set_target(*aligned_targets[0]["pelvis"])
    np.testing.assert_allclose(root_task.compute_error(configuration), 0.0, atol=1e-10)
