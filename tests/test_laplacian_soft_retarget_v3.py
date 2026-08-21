from pathlib import Path
import unittest

import mink
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from general_motion_retargeting.laplacian_soft_retarget_v3 import (
    FootAnchorTask,
    LaplacianSoftContactRetargetingV3,
    QposSubsetTask,
    SwingClearanceTask,
)


MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "ne01" / "ne01_v3.xml"


def make_model_and_configuration():
    model = mj.MjModel.from_xml_path(str(MODEL_PATH))
    return model, mink.Configuration(model)


class LaplacianSoftRetargetV3Tests(unittest.TestCase):
    def test_qpos_subset_maps_hinge_after_free_joint_to_correct_dof(self):
        model, _ = make_model_and_configuration()
        waist_qpos = int(model.jnt_qposadr[model.joint("WAIST_YAW_JOINT").id])
        task = QposSubsetTask(model, [waist_qpos], [0.0])
        self.assertEqual(task.dof_indices.tolist(), [6])

    def test_deactivated_swing_clearance_has_zero_error_and_jacobian(self):
        model, configuration = make_model_and_configuration()
        geom_ids = [
            model.geom(f"left_foot_{point}_collision").id
            for point in ("rear_left", "rear_right", "front_left", "front_right")
        ]
        task = SwingClearanceTask(model, geom_ids, cost=20.0)
        task.set_state(0.0, range(4), clearance=0.10)
        np.testing.assert_allclose(task.compute_error(configuration), 0.0)
        np.testing.assert_allclose(task.compute_jacobian(configuration), 0.0)

    def test_anchor_deadband_suppresses_submillimeter_correction(self):
        model, configuration = make_model_and_configuration()
        geom_ids = [
            model.geom(f"right_foot_{point}_collision").id
            for point in ("rear_left", "rear_right", "front_left", "front_right")
        ]
        task = FootAnchorTask(model, geom_ids, deadband=0.003)
        target = task._point(configuration)[:2] + np.array([0.001, 0.0])
        task.set_target(target, 1.0)
        np.testing.assert_allclose(task.compute_error(configuration), 0.0)

    def test_total_frame_budget_is_shared_by_all_stages(self):
        retargeter = LaplacianSoftContactRetargetingV3(
            src_human="smplx",
            tgt_robot="ne01",
            motion_fps=50.0,
            velocity_limit=3.0 * np.pi,
            verbose=False,
        )
        q_start = retargeter.configuration.data.qpos.copy()
        candidate = q_start.copy()
        candidate[7:] += 1.0
        candidate[2] += 1.0
        candidate[3:7] = Rotation.from_euler("x", 1.0).as_quat(scalar_first=True)
        clipped = retargeter._clip_total_budget(q_start, candidate)
        self.assertLessEqual(
            np.max(np.abs(clipped[7:] - q_start[7:])),
            3.0 * np.pi / 50.0 + 1e-12,
        )
        self.assertLessEqual(abs(clipped[2] - q_start[2]), 0.25 / 50.0 + 1e-12)
        root_angle = (
            Rotation.from_quat(clipped[3:7], scalar_first=True)
            * Rotation.from_quat(q_start[3:7], scalar_first=True).inv()
        ).magnitude()
        self.assertLessEqual(root_angle, 3.0 * np.pi / 50.0 + 1e-12)

    def test_v3_collision_proxy_pairs_are_active_and_nonpenetrating_at_rest(self):
        retargeter = LaplacianSoftContactRetargetingV3(
            src_human="smplx",
            tgt_robot="ne01",
            motion_fps=50.0,
            verbose=False,
        )
        self.assertTrue(retargeter.xml_file.endswith("ne01_v3.xml"))
        self.assertEqual(len(retargeter._collision_geom_pairs), 8)
        distance = retargeter._min_collision_distance(
            retargeter.configuration.data.qpos.copy()
        )
        self.assertGreater(distance, 0.01)


if __name__ == "__main__":
    unittest.main()
