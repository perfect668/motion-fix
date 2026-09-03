import numpy as np

from general_motion_retargeting.motion_adapters import CanonicalMotion
from general_motion_retargeting.terrain_geometry import SceneTransform
from general_motion_retargeting.terrain_contact_utils import _proxy_points
from general_motion_retargeting.wholebody_omni_gmr_v4 import LimbPlaneTask


def test_measured_foot_landmarks_are_preferred_and_ankle_is_foot():
    names = ["pelvis", "spine3", "left_hip", "right_hip", "left_knee", "right_knee",
             "left_ankle", "right_ankle", "left_big_toe", "left_small_toe",
             "right_big_toe", "right_small_toe", "left_heel", "right_heel"]
    values = np.zeros((1, len(names), 3), dtype=float)
    for i, name in enumerate(names):
        values[0, i] = [float(i), 0.0, 0.5]
    motion = CanonicalMotion(values, names, orientations=None, orientation_valid=False, fps=50.0)
    semantic = motion.canonical_named_positions()[0]
    np.testing.assert_allclose(semantic["left_foot"], semantic["left_ankle"])
    np.testing.assert_allclose(semantic["left_toe"], 0.5 * (values[0, 8] + values[0, 9]))
    np.testing.assert_allclose(semantic["left_heel"], values[0, 12])
    assert motion.metadata["canonical_provenance"][0]["left_heel"] == "measured_heel"


def test_scene_scale_is_applied_once_to_object_transform():
    transform = SceneTransform(np.eye(3), 1.0, np.zeros(3))
    point = np.array([0.2, -0.1, 0.4])
    np.testing.assert_allclose(transform.transform_points(point), point)


def test_scene_aabb_transform_is_deterministic_at_scale_one():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    translation = np.array([2.0, -1.0, 0.5])
    transformed = vertices @ rotation.T + translation
    np.testing.assert_allclose(transformed.min(axis=0), [-0.0, -1.0, 0.5])
    np.testing.assert_allclose(transformed.max(axis=0), [2.0, 0.0, 3.5])



def test_contact_uses_measured_heel_and_toe_midpoint():
    frame = {
        "left_ankle": np.array([0.0, 0.1, 0.1]), "right_ankle": np.array([0.0, -0.1, 0.1]),
        "left_big_toe": np.array([0.2, 0.11, 0.1]), "left_small_toe": np.array([0.2, 0.09, 0.1]),
        "right_big_toe": np.array([0.2, -0.09, 0.1]), "right_small_toe": np.array([0.2, -0.11, 0.1]),
        "left_heel": np.array([-0.1, 0.1, 0.1]), "right_heel": np.array([-0.1, -0.1, 0.1]),
    }
    points = _proxy_points(frame, {})
    np.testing.assert_allclose(points["left_heel"][0], frame["left_heel"])
    np.testing.assert_allclose(points["left_toe"][0], [0.2, 0.1, 0.1])
    assert points["left_heel"][1] == "measured_heel"


class _Model:
    nv = 1


class _Point:
    def __init__(self, fn, jac):
        self.fn, self.jac = fn, np.asarray(jac, dtype=float).reshape(3, 1)
    def point(self, _configuration):
        return np.asarray(self.fn(), dtype=float)
    def jacobian(self, _configuration):
        return self.jac


def test_limb_plane_jacobian_matches_finite_difference_and_mirror_is_nonzero():
    q = [0.0]
    points = {
        "left_hip": _Point(lambda: [0, 0, 0], [[0], [0], [0]]),
        "left_knee": _Point(lambda: [1, 0, 0], [[0], [0], [0]]),
        "left_foot": _Point(lambda: [0, -1, q[0]], [[0], [0], [1]]),
    }
    task = LimbPlaneTask(_Model(), points, [("left_hip", "left_knee", "left_foot")], 1.0)
    task.set_source({"left_hip": np.array([0, 0, 0]), "left_knee": np.array([1, 0, 0]), "left_foot": np.array([0, 1, 0])})
    class C: pass
    configuration = C()
    analytic = task.compute_jacobian(configuration)[:, 0]
    eps = 1e-6
    q[0] = eps
    plus = task.compute_error(configuration)
    q[0] = -eps
    minus = task.compute_error(configuration)
    numeric = (plus - minus) / (2 * eps)
    q[0] = 0.0
    assert np.linalg.norm(analytic) > 1e-6
    np.testing.assert_allclose(analytic, numeric, rtol=2e-4, atol=2e-5)
