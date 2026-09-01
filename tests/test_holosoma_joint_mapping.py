import json

import numpy as np
import pytest

from general_motion_retargeting.holosoma_input import build_solver_frames, load_holosoma_positions, load_joint_map


def _mapping_path():
    from pathlib import Path
    return Path(__file__).parents[1] / "general_motion_retargeting/joint_maps/holosoma_53.json"


def test_verified_53_mapping_and_quaternion_continuity(tmp_path):
    mapping = load_joint_map(_mapping_path())
    names = mapping["source_joint_names"]
    positions = np.zeros((3, len(names), 3), dtype=np.float32)
    index = {name: i for i, name in enumerate(names)}
    for frame in range(3):
        positions[frame, index["LeftUpLeg"]] = [0, 0.1, 0.9]
        positions[frame, index["RightUpLeg"]] = [0, -0.1, 0.9]
        positions[frame, index["Spine1"]] = [0, 0, 1.3]
        positions[frame, index["Neck"]] = [0, 0, 1.5]
        positions[frame, index["LeftArm"]] = [0, 0.3, 1.4]
        positions[frame, index["RightArm"]] = [0, -0.3, 1.4]
        positions[frame, index["LeftFoot"]] = [0, 0.1, 0.1]
        positions[frame, index["LeftToeBase"]] = [0.2, 0.1, 0.1]
        positions[frame, index["RightFoot"]] = [0, -0.1, 0.1]
        positions[frame, index["RightToeBase"]] = [0.2, -0.1, 0.1]
    path = tmp_path / "motion.npy"
    np.save(path, positions)
    loaded, loaded_names, _ = load_holosoma_positions(path, mapping)
    frames, validity = build_solver_frames(loaded, loaded_names, mapping)
    assert len(frames) == 3
    assert not validity["measured"]
    quaternions = np.asarray([frame["pelvis"][1] for frame in frames])
    assert np.all(np.sum(quaternions[:-1] * quaternions[1:], axis=1) >= 0.0)


def test_wrong_point_count_fails_explicitly(tmp_path):
    mapping = load_joint_map(_mapping_path())
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((2, 52, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="does not match"):
        load_holosoma_positions(path, mapping)


def test_duplicate_npz_joint_names_fail(tmp_path):
    mapping = load_joint_map(_mapping_path())
    names = mapping["source_joint_names"].copy()
    names[-1] = names[-2]
    path = tmp_path / "bad.npz"
    np.savez(path, joint_positions=np.zeros((2, 53, 3)), joint_names=np.asarray(names))
    with pytest.raises(ValueError, match="Duplicate"):
        load_holosoma_positions(path, mapping)
