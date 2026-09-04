import numpy as np
from types import SimpleNamespace
import importlib.util
from pathlib import Path

from general_motion_retargeting.terrain_geometry import SceneTransform

_spec = importlib.util.spec_from_file_location("retarget_motion_entry", Path(__file__).parents[1] / "scripts" / "retarget_motion.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_solver_inputs = _module._solver_inputs


def test_smplx_foot_semantics():
    names = ["pelvis", "left_hip", "right_hip", "spine1", "spine2", "spine3", "neck",
             "head", "left_collar", "right_collar", "left_shoulder", "right_shoulder",
             "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_knee", "right_knee",
             "left_ankle", "right_ankle", "left_big_toe", "right_big_toe", "left_small_toe",
             "right_small_toe", "left_heel", "right_heel"]
    points = {name: np.array([float(i), float(i % 3), 1.0 + i * 0.01]) for i, name in enumerate(names)}
    points.update({"left_ankle": np.array([1., 0., .2]), "left_foot": np.array([9., 9., 9.]),
                   "left_big_toe": np.array([1., .2, .1]), "left_small_toe": np.array([1., -.2, .1]),
                   "left_heel": np.array([.7, 0., .1]), "right_ankle": np.array([-1., 0., .2]),
                   "right_big_toe": np.array([-1., .2, .1]), "right_small_toe": np.array([-1., -.2, .1]),
                   "right_heel": np.array([-.7, 0., .1])})
    positions = np.asarray([[points[n] for n in names]])
    motion = SimpleNamespace(source_format="smplx_npz", joint_names=names, positions=positions,
                             frame_count=1, orientations=np.tile([1., 0., 0., 0.], (1, len(names), 1)))
    source, solver, _ = _solver_inputs(motion, SceneTransform(np.eye(3), 1., np.zeros(3)), {}, None)
    assert np.allclose(source[0]["left_foot"], points["left_ankle"])
    assert np.allclose(source[0]["left_toe"], (points["left_big_toe"] + points["left_small_toe"]) / 2)
    assert np.allclose(solver[0]["left_foot"][0], points["left_ankle"])
    assert np.allclose(solver[0]["left_toe"][0], source[0]["left_toe"])
