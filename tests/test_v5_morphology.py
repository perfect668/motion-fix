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
