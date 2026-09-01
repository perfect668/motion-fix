from pathlib import Path

import numpy as np

from general_motion_retargeting.wholebody_omni_gmr_v2 import WholeBodyOmniGMRV2
from general_motion_retargeting.wholebody_terrain_omni_gmr_v2 import WholeBodyTerrainOmniGMRV2


def test_terrain_class_is_independent_subclass_and_flat_entry_remains():
    assert issubclass(WholeBodyTerrainOmniGMRV2, WholeBodyOmniGMRV2)
    root = Path(__file__).parents[1]
    assert (root / "scripts/smplx_to_robot_wholebody_omni_gmr_v2.py").is_file()
    assert (root / "scripts/holosoma_to_robot_terrain.py").is_file()


def test_real_climb_source_has_verified_53_points_when_available():
    motion = Path("/home/user/桌面/holosoma/holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/climb/mocap_climb_seq_0/mocap_climb_seq_0_joint_positions_f900-3700.npy")
    if motion.exists():
        positions = np.load(motion, mmap_mode="r")
        assert positions.shape == (2801, 53, 3)
