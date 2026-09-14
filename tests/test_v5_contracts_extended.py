import numpy as np
import pytest

from general_motion_retargeting.core.schemas import PoseTrajectory, SceneAsset, SceneRelation
from general_motion_retargeting.motion_adapters import CanonicalMotion, get_motion_adapter
from general_motion_retargeting.contact import SourceContactDetector
from general_motion_retargeting.terrain_geometry import BoxPrimitive
from general_motion_retargeting.input import resolver_for_motion
from general_motion_retargeting.scene.loader import CompositeSceneField, DynamicSceneField, MeshSceneField
from general_motion_retargeting.terrain_geometry import TerrainField


def test_adapter_registry_exposes_all_v5_formats():
    for name in ("smplx_npz", "grail_smplx_recon", "holosoma_global_positions", "bvh", "fbx"):
        assert get_motion_adapter(name).name == name


def test_grail_declared_human_scale_is_body_local_not_a_scene_scale():
    from general_motion_retargeting.motion_adapters import _human_geometry_scale

    assert _human_geometry_scale({"scale": np.array([0.75])}, "grail_smplx_recon") == 0.75
    assert _human_geometry_scale({"scale": np.array([0.75])}, "smplx_npz") == 1.0
    with pytest.raises(Exception, match="one finite positive scalar"):
        _human_geometry_scale({"scale": np.array([1.0, 1.0])}, "grail_smplx_recon")


def test_canonical_npz_adapter_is_not_treated_as_holosoma(tmp_path):
    path = tmp_path / "canonical.npz"
    np.savez(path, positions=np.zeros((1, 2, 3)), joint_names=np.asarray(["pelvis", "spine3"]), fps=np.asarray(50.0))
    from general_motion_retargeting.motion_adapters import detect_motion_format, load_canonical_motion
    assert detect_motion_format(path) == "canonical_npz"
    loaded = load_canonical_motion(path)
    assert loaded.source_format == "canonical_npz"


def test_position_only_canonical_motion_does_not_fabricate_toe():
    names = ["pelvis", "left_ankle", "right_ankle", "left_knee", "right_knee"]
    positions = np.zeros((1, len(names), 3), dtype=float)
    positions[0, names.index("pelvis")] = [0, 0, 1]
    motion = CanonicalMotion(positions, names, 50.0)
    frame = motion.canonical_named_positions()[0]
    assert "left_toe" not in frame
    assert "right_toe" not in frame


def test_v5_solver_inputs_use_canonical_ankle_for_foot_alias():
    from general_motion_retargeting.input.solver_frames import build_solver_inputs

    names = [
        "pelvis", "spine3", "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_foot", "right_foot", "left_toe", "right_toe",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
    ]
    values = np.zeros((1, len(names), 3), dtype=float)
    for index, name in enumerate(names):
        values[0, index] = [float(index), 0.0, 1.0]
    values[0, names.index("pelvis")] = [0.0, 0.0, 1.0]
    values[0, names.index("spine3")] = [0.0, 0.0, 1.3]
    values[0, names.index("left_hip")] = [0.0, 0.1, 1.0]
    values[0, names.index("right_hip")] = [0.0, -0.1, 1.0]
    values[0, names.index("left_ankle")] = [1.0, 0.1, 0.1]
    values[0, names.index("right_ankle")] = [1.0, -0.1, 0.1]
    values[0, names.index("left_foot")] = [9.0, 9.0, 9.0]
    values[0, names.index("right_foot")] = [9.0, -9.0, 9.0]
    motion = CanonicalMotion(values, names, 50.0, orientation_valid=False)
    _, solver_frames, _ = build_solver_inputs(motion)
    np.testing.assert_allclose(solver_frames[0]["left_foot"][0], [1.0, 0.1, 0.1])
    np.testing.assert_allclose(solver_frames[0]["right_foot"][0], [1.0, -0.1, 0.1])


def test_contact_motion_uses_shared_scene_transform_for_shared_scene():
    import importlib.util
    from pathlib import Path

    module_spec = importlib.util.spec_from_file_location(
        "v5_retarget_entry", Path(__file__).parents[1] / "scripts" / "retarget.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    _contact_motion_in_query_frame = module._contact_motion_in_query_frame
    from general_motion_retargeting.terrain_geometry import SceneTransform

    motion = type("Motion", (), {})()
    motion.positions = np.zeros((1, 1, 3), dtype=float)
    source = np.asarray([[[1.0, 2.0, 3.0]]])
    transform = SceneTransform(np.eye(3), 2.0, np.array([0.5, -1.0, 0.25]))
    result = _contact_motion_in_query_frame(
        motion, source, transform, SceneRelation.SHARED_SCENE
    )
    np.testing.assert_allclose(result.positions, [[[2.5, 3.0, 6.25]]])
    # The source motion object is not mutated; this is important because the
    # same untransformed timeline is retained for provenance/export.
    np.testing.assert_allclose(motion.positions, np.zeros((1, 1, 3)))


def test_v5_root_task_constrains_full_se3_orientation():
    import mink
    import mujoco as mj
    from general_motion_retargeting.wholebody_omni_gmr_v5 import _RootTask

    model = mj.MjModel.from_xml_path(
        "assets/ne01/ne01_desktop_assets_wholebody_omni_gmr_v2.xml"
    )
    configuration = mink.Configuration(model)
    task = _RootTask(model, "base_link", [60.0, 60.0, 12.0, 8.0])
    task.set_target(
        configuration.data.xpos[model.body("base_link").id],
        configuration.data.xquat[model.body("base_link").id],
    )

    assert np.asarray(task.cost).shape == (6,)
    np.testing.assert_allclose(task.compute_error(configuration), np.zeros(6), atol=1e-10)
    assert task.compute_jacobian(configuration).shape == (6, model.nv)


def test_v5_qpos_interpolation_uses_mujoco_joint_manifold():
    import mujoco as mj
    from scipy.spatial.transform import Rotation
    from general_motion_retargeting.wholebody_omni_gmr_v5 import _interpolate_qpos

    model = mj.MjModel.from_xml_path(
        "assets/ne01/ne01_desktop_assets_wholebody_omni_gmr_v2.xml"
    )
    start = model.qpos0.copy()
    end = start.copy()
    end[:3] += [0.4, -0.2, 0.1]
    end[3:7] = Rotation.from_euler("xyz", [0.2, -0.3, 0.5]).as_quat(
        scalar_first=True
    )
    end[7:] += 0.2

    np.testing.assert_allclose(_interpolate_qpos(model, start, end, 0.0), start)
    np.testing.assert_allclose(_interpolate_qpos(model, start, end, 1.0), end)
    middle = _interpolate_qpos(model, start, end, 0.5)
    np.testing.assert_allclose(middle[:3], 0.5 * (start[:3] + end[:3]))
    np.testing.assert_allclose(np.linalg.norm(middle[3:7]), 1.0)
    np.testing.assert_allclose(middle[7:], 0.5 * (start[7:] + end[7:]))


def test_v5_frame_displacement_limit_is_expressed_in_delta_q():
    import mink
    import mujoco as mj
    from general_motion_retargeting.wholebody_omni_gmr_v5 import (
        _FrameDisplacementLimit,
    )

    model = mj.MjModel.from_xml_path(
        "assets/ne01/ne01_desktop_assets_wholebody_omni_gmr_v2.xml"
    )
    configuration = mink.Configuration(model)
    limit = _FrameDisplacementLimit(model, {
        "base_linear_velocity_limit": 1.0,
        "base_angular_velocity_limit": 4.0,
        "joint_velocity_limit": 3.0,
    })
    limit.set_frame(configuration.data.qpos, 0.02, enabled=True)
    constraint = limit.compute_qp_inequalities(configuration, 0.005)

    assert constraint.G.shape[1] == model.nv
    assert constraint.h.shape == (constraint.G.shape[0],)
    # Mink's QP variable is delta-q, so the frame budget is 1 m/s * 20 ms;
    # the internal IK substep dt must not divide this bound again.
    translation_rows = constraint.G[:, :3]
    x_row = np.flatnonzero(np.all(translation_rows == [1.0, 1.0, 1.0], axis=1))[0]
    assert constraint.h[x_row] == pytest.approx(0.02)

    moved = configuration.data.qpos.copy()
    moved[0] += 0.015
    configuration.update(moved)
    updated = limit.compute_qp_inequalities(configuration, 0.005)
    assert updated.h[x_row] == pytest.approx(0.005)


def test_pose_trajectory_rejects_duplicate_timestamps():
    with pytest.raises(ValueError, match="strictly increasing"):
        PoseTrajectory(np.array([0.0, 0.0]), np.zeros((2, 3)))


def test_scene_asset_pose_at_interpolates_translation():
    asset = SceneAsset(
        "chair", "/tmp/chair.obj",
        pose_trajectory=PoseTrajectory(np.array([0.0, 1.0]), np.array([[0, 0, 0], [1, 2, 3.]])),
    )
    pose = asset.pose_at(0.5)
    np.testing.assert_allclose(pose[:3, 3], [0.5, 1.0, 1.5])


def test_scene_asset_pose_at_preserves_metric_scale():
    from general_motion_retargeting.core.schemas import PoseTrajectory, SceneAsset
    base = np.eye(4)
    base[:3, :3] *= 0.8
    asset = SceneAsset(
        "scaled", "/tmp/scaled.obj", pose=base,
        pose_trajectory=PoseTrajectory(
            np.array([0.0, 1.0]), np.zeros((2, 3)),
            np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1)),
        ),
    )
    np.testing.assert_allclose(np.linalg.norm(asset.pose_at(0.5)[:3, :3], axis=0), 0.8)


def test_dynamic_scene_field_updates_query_time():
    from types import SimpleNamespace
    from pathlib import Path
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    loaded = SimpleNamespace(vertices=vertices, faces=np.array([[0, 1, 2]]))
    asset = SceneAsset(
        "moving", Path("/tmp/moving.obj"),
        pose_trajectory=PoseTrajectory(np.array([0.0, 1.0]), np.array([[0, 0, 0], [0, 0, 1.]])),
    )
    field = DynamicSceneField([(asset, loaded)], floor_z=None)
    field.set_time(0.0)
    first = field.nearest_surface(np.array([0.2, 0.2, 0.1]))
    field.set_time(1.0)
    second = field.nearest_surface(np.array([0.2, 0.2, 0.1]))
    assert first.signed_distance < second.signed_distance


def test_composite_scene_field_queries_all_assets():
    first = TerrainField(floor_z=0.0)
    vertices = np.array([[-1, -1, 1], [1, -1, 1], [0, 1, 1]], dtype=float)
    mesh = MeshSceneField(vertices, np.array([[0, 1, 2]]), "platform", floor_z=None)
    field = CompositeSceneField([first, mesh])
    assert field.nearest_surface(np.array([0.0, 0.0, 1.1])).surface_id.startswith("platform:")
    assert field.support_surface(np.array([0.0, 0.0, 1.2])).surface_id.startswith("platform:")


def test_prepared_scene_reuses_asset_geometry_and_transforms_box_primitives(tmp_path):
    import json
    from general_motion_retargeting.core.schemas import FloorPolicy, SceneModel

    spec = tmp_path / "terrain.json"
    spec.write_text(json.dumps({"boxes": [{
        "surface_id": "step", "center": [0.0, 0.0, 0.5],
        "half_extents": [0.5, 0.5, 0.5],
    }]}), encoding="utf-8")
    pose = np.eye(4)
    pose[:3, :3] *= 2.0
    pose[:3, 3] = [1.0, -2.0, 3.0]
    asset = SceneAsset("step", spec, pose=pose)
    source = SceneModel(assets=(asset,), floor_policy=FloorPolicy.DISABLED, floor_height=None)
    # Build directly through PreparedSceneGeometry because this fixture already
    # has a resolved SceneModel, then verify with_scene only changes floor.
    from general_motion_retargeting.scene.prepared import PreparedSceneGeometry
    prepared = PreparedSceneGeometry.from_scene(source)
    box = prepared.boxes[0]
    np.testing.assert_allclose(box.center, [1.0, -2.0, 4.0])
    np.testing.assert_allclose(box.half_extents, [1.0, 1.0, 1.0])
    final = prepared.with_scene(SceneModel(assets=(asset,), floor_policy=FloorPolicy.EXPLICIT, floor_height=0.0))
    assert final.mesh_assets is prepared.mesh_assets
    assert final.boxes is prepared.boxes
    hit = final.terrain_field().support_surface(np.array([1.0, -2.0, 5.2]))
    assert hit.surface_id == "step:z+"
    np.testing.assert_allclose(hit.closest_point[2], 5.0)


def test_support_replay_rejects_an_expected_support_frame_that_floats():
    from types import SimpleNamespace
    from general_motion_retargeting.validation.kinematic import _replay_support

    class Configuration:
        def update(self, qpos):
            self.qpos = np.asarray(qpos, dtype=float)

    class Point:
        def value(self, configuration):
            return np.array([0.0, 0.0, configuration.qpos[2]])

    class Terrain:
        @staticmethod
        def support_surface(point):
            return SimpleNamespace(supportable=True, signed_distance=float(point[2]))

    solver = SimpleNamespace(
        configuration=Configuration(),
        contact=SimpleNamespace(points={"left_heel": Point()}),
        terrain=Terrain(),
        config={"validation": {"max_support_gap": 0.005, "max_support_penetration": 0.002}},
    )
    plan = SimpleNamespace(per_frame_states=({
        "support_expected": True,
        "support_state": "SUPPORTED",
        "contacts": {"left_heel": {}},
    },))
    metrics = _replay_support(solver, plan, np.array([[0.0, 0.0, 0.10]]))
    assert metrics["expected_frames"] == 1
    assert metrics["failed_frames"] == 1
    assert metrics["support_state_frames"]["SUPPORTED"] == 1


def test_support_replay_rejects_one_floating_active_foot_even_if_other_is_grounded():
    from types import SimpleNamespace
    from general_motion_retargeting.validation.kinematic import _replay_support

    class Configuration:
        def update(self, qpos):
            self.qpos = np.asarray(qpos, dtype=float)

    class Point:
        def __init__(self, z):
            self.z = float(z)

        def value(self, configuration):
            return np.array([0.0, 0.0, self.z + configuration.qpos[2]])

    solver = SimpleNamespace(
        configuration=Configuration(),
        contact=SimpleNamespace(
            points={"left_heel": Point(0.004), "right_heel": Point(0.020)}
        ),
        terrain=SimpleNamespace(),
        config={"validation": {"max_support_gap": 0.005, "max_support_penetration": 0.002}},
    )
    plan = {"frames": [{
        "support_expected": True,
        "support_state": "SUPPORTED",
        "contacts": {
            "left_heel": {"state": "STATIC", "activation": 1.0,
                          "surface_point_solver": [0.0, 0.0, 0.0],
                          "anchor_normal_solver": [0.0, 0.0, 1.0]},
            "right_heel": {"state": "STATIC", "activation": 1.0,
                           "surface_point_solver": [0.0, 0.0, 0.0],
                           "anchor_normal_solver": [0.0, 0.0, 1.0]},
        },
    }]}
    metrics = _replay_support(solver, plan, np.array([[0.0, 0.0, 0.0]]))
    assert metrics["covered_frames"] == 0
    assert metrics["failed_frames"] == 1


def test_support_replay_fails_closed_when_contact_schedule_is_missing():
    from types import SimpleNamespace
    from general_motion_retargeting.validation.kinematic import _replay_support

    class Configuration:
        def update(self, qpos):
            self.qpos = np.asarray(qpos, dtype=float)

    class Point:
        def value(self, configuration):
            return np.array([0.0, 0.0, configuration.qpos[2]])

    class Terrain:
        @staticmethod
        def support_surface(point):
            return SimpleNamespace(supportable=True, signed_distance=float(point[2]))

    solver = SimpleNamespace(
        configuration=Configuration(),
        contact=SimpleNamespace(points={name: Point() for name in ("left_heel", "left_toe")}),
        terrain=Terrain(),
        config={"validation": {"max_support_gap": 0.005, "max_support_penetration": 0.002}},
    )
    metrics = _replay_support(solver, None, np.array([[0.0, 0.0, 0.10]]))
    assert metrics["expected_frames"] == 1
    assert metrics["failed_frames"] == 1
    assert metrics["coverage_ratio"] == 0.0


def test_support_replay_accepts_exported_schedule_dictionary():
    from types import SimpleNamespace
    from general_motion_retargeting.validation.kinematic import _replay_support

    class Configuration:
        def update(self, qpos):
            self.qpos = np.asarray(qpos, dtype=float)

    class Point:
        def value(self, configuration):
            return np.array([0.0, 0.0, configuration.qpos[2]])

    class Terrain:
        @staticmethod
        def support_surface(point):
            return SimpleNamespace(supportable=True, signed_distance=float(point[2]))

    solver = SimpleNamespace(
        configuration=Configuration(),
        contact=SimpleNamespace(points={"left_heel": Point()}),
        terrain=Terrain(),
        config={"validation": {"max_support_gap": 0.005, "max_support_penetration": 0.002}},
    )
    exported = {
        "frames": [{
            "support_expected": True,
            "support_state": "SUPPORTED",
            "contacts": {"left_heel": {"state": "STATIC", "activation": 1.0}},
        }]
    }
    metrics = _replay_support(solver, exported, np.array([[0.0, 0.0, 0.0]]))
    assert metrics["covered_frames"] == 1
    assert metrics["failed_frames"] == 0


def test_support_replay_uses_locked_contact_surface_at_stair_edge():
    from types import SimpleNamespace
    from general_motion_retargeting.validation.kinematic import _replay_support

    class Configuration:
        def update(self, qpos):
            self.qpos = np.asarray(qpos, dtype=float)

    class Point:
        def value(self, configuration):
            return np.asarray(configuration.qpos[:3], dtype=float)

    class Terrain:
        @staticmethod
        def support_surface(point):
            # The nearest-triangle query intentionally returns the lower
            # tread, modeling the ambiguous edge case the locked schedule
            # must avoid during final replay.
            return SimpleNamespace(supportable=True, signed_distance=0.20)

    solver = SimpleNamespace(
        configuration=Configuration(),
        contact=SimpleNamespace(points={"left_toe": Point()}),
        terrain=Terrain(),
        config={"validation": {"max_support_gap": 0.012, "max_support_penetration": 0.002}},
    )
    plan = {"frames": [{
        "support_expected": True,
        "support_state": "SUPPORTED",
        "contacts": {"left_toe": {
            "state": "STATIC",
            "activation": 1.0,
            "tangent_anchor_solver": [0.0, 0.0, 0.0],
            "anchor_normal_solver": [0.0, 0.0, 1.0],
        }},
    }]}
    metrics = _replay_support(solver, plan, np.array([[0.0, 0.0, 0.004]]))
    assert metrics["covered_frames"] == 1
    assert metrics["failed_frames"] == 0


def test_support_replay_skips_unreachable_surface_transition_when_other_foot_supports():
    from types import SimpleNamespace
    from general_motion_retargeting.validation.kinematic import _replay_support

    class Configuration:
        def update(self, qpos):
            self.qpos = np.asarray(qpos, dtype=float)

    class Point:
        def value(self, configuration):
            return np.asarray([0.0, 0.0, configuration.qpos[2]], dtype=float)

    class Terrain:
        @staticmethod
        def support_surface(point):
            return SimpleNamespace(supportable=True, signed_distance=0.0)

    solver = SimpleNamespace(
        configuration=Configuration(),
        contact=SimpleNamespace(points={"left_heel": Point(), "right_toe": Point()}),
        terrain=Terrain(),
        config={
            "validation": {
                "max_support_gap": 0.005,
                "max_support_penetration": 0.002,
                "support_activation_min": 0.5,
            }
        },
    )
    plan = {"frames": [{
        "support_expected": True,
        "support_state": "SUPPORTED",
        "contacts": {
            "left_heel": {
                "state": "STATIC",
                "activation": 1.0,
                "surface_point_solver": [0.0, 0.0, 0.0],
                "surface_normal_solver": [0.0, 0.0, 1.0],
            },
            # A new stair tread is 10 cm higher in this frame.  The stable
            # left heel is still the load-bearing support for this frame.
            "right_toe": {
                "state": "SLIDING",
                "activation": 1.0,
                "surface_transition": True,
                "surface_point_solver": [0.0, 0.0, 0.10],
                "surface_normal_solver": [0.0, 0.0, 1.0],
            },
        },
    }]}
    metrics = _replay_support(solver, plan, np.array([[0.0, 0.0, 0.004]]))
    assert metrics["covered_frames"] == 1
    assert metrics["failed_frames"] == 0


def test_support_replay_defers_bounded_fast_landing_frame():
    from types import SimpleNamespace
    from general_motion_retargeting.validation.kinematic import _replay_support

    class Configuration:
        def update(self, qpos):
            self.qpos = np.asarray(qpos, dtype=float)

    class Point:
        def value(self, configuration):
            return np.array([0.0, 0.0, configuration.qpos[2]])

    class Terrain:
        @staticmethod
        def support_surface(point):
            return SimpleNamespace(supportable=True, signed_distance=float(point[2]))

    solver = SimpleNamespace(
        configuration=Configuration(),
        contact=SimpleNamespace(points={"left_toe": Point()}),
        terrain=Terrain(),
        config={"validation": {"max_support_gap": 0.005, "max_support_penetration": 0.002}},
    )
    plan = {"frames": [{
        "support_expected": True,
        "support_state": "SUPPORTED",
        "support_transition": True,
        "contacts": {"left_toe": {"state": "SLIDING", "activation": 0.1}},
    }, {
        "support_expected": True,
        "support_state": "SUPPORTED",
        "contacts": {"left_toe": {
            "state": "STATIC",
            "activation": 1.0,
            "surface_point_solver": [0.0, 0.0, 0.0],
            "surface_normal_solver": [0.0, 0.0, 1.0],
        }},
    }]}
    metrics = _replay_support(
        solver, plan, np.array([[0.0, 0.0, 0.25], [0.0, 0.0, 0.004]])
    )
    assert metrics["expected_frames"] == 2
    assert metrics["covered_frames"] == 1
    assert metrics["failed_frames"] == 0
    assert metrics["deferred_frames"] == [0]


def test_support_replay_treats_empty_schedule_as_missing():
    from types import SimpleNamespace
    from general_motion_retargeting.validation.kinematic import _replay_support

    class Configuration:
        def update(self, qpos):
            self.qpos = np.asarray(qpos, dtype=float)

    class Point:
        def value(self, configuration):
            return np.array([0.0, 0.0, configuration.qpos[2]])

    class Terrain:
        @staticmethod
        def support_surface(point):
            return SimpleNamespace(supportable=True, signed_distance=float(point[2]))

    solver = SimpleNamespace(
        configuration=Configuration(),
        contact=SimpleNamespace(points={"left_heel": Point()}),
        terrain=Terrain(),
        config={"validation": {"max_support_gap": 0.005, "max_support_penetration": 0.002}},
    )
    metrics = _replay_support(solver, {"frames": []}, np.array([[0.0, 0.0, 0.10]]))
    assert metrics["expected_frames"] == 1
    assert metrics["failed_frames"] == 1


def test_support_replay_does_not_accept_sequence_with_only_activation_ramp():
    from types import SimpleNamespace
    from general_motion_retargeting.validation.kinematic import _replay_support

    class Configuration:
        def update(self, qpos):
            self.qpos = np.asarray(qpos, dtype=float)

    class Point:
        def value(self, configuration):
            return np.array([0.0, 0.0, configuration.qpos[2]])

    class Terrain:
        @staticmethod
        def support_surface(point):
            return SimpleNamespace(supportable=True, signed_distance=float(point[2]))

    solver = SimpleNamespace(
        configuration=Configuration(),
        contact=SimpleNamespace(points={"left_heel": Point()}),
        terrain=Terrain(),
        config={"validation": {"max_support_gap": 0.005, "max_support_penetration": 0.002, "support_activation_min": 0.5}},
    )
    plan = {
        "frames": [{
            "support_expected": True,
            "support_state": "SUPPORTED",
            "contacts": {"left_heel": {"state": "STATIC", "activation": 0.1}},
        }]
    }
    metrics = _replay_support(solver, plan, np.array([[0.0, 0.0, 0.25]]))
    assert metrics["expected_frames"] == 1
    assert metrics["covered_frames"] == 0
    assert metrics["failed_frames"] == 1
    assert metrics["deferred_frames"] == [0]


def test_source_motion_scene_alignment_requires_consistent_labelled_support():
    from general_motion_retargeting.input.motion_scene_alignment import align_motion_to_scene_support
    from general_motion_retargeting.terrain_geometry import SceneTransform, TerrainField

    names = ["pelvis", "left_heel", "right_heel", "left_toe", "right_toe"]
    positions = np.zeros((4, len(names), 3), dtype=float)
    positions[:, 0, 2] = 1.0
    positions[:, 1:, 2] = -0.10
    motion = CanonicalMotion(
        positions, names, 50.0,
        metadata={
            "foot_contact_probs": np.ones((4, 4)),
            "foot_contact_channel_order": (
                "left_heel", "right_heel", "left_toe", "right_toe",
            ),
        },
    )
    report = align_motion_to_scene_support(
        motion, TerrainField(floor_z=0.0),
        SceneTransform(np.eye(3), 1.0, np.zeros(3)),
        {"min_samples": 4, "max_vertical_offset": 0.25},
    )
    assert report["status"] == "APPLIED"
    np.testing.assert_allclose(report["translation_source"], [0.0, 0.0, 0.10])
    np.testing.assert_allclose(motion.canonical_named_positions()[0]["left_heel"][2], 0.0)


def test_source_motion_scene_alignment_refuses_inconsistent_evidence():
    from general_motion_retargeting.input.motion_scene_alignment import align_motion_to_scene_support
    from general_motion_retargeting.terrain_geometry import SceneTransform, TerrainField

    names = ["pelvis", "left_heel", "right_heel", "left_toe", "right_toe"]
    positions = np.zeros((4, len(names), 3), dtype=float)
    positions[:, 0, 2] = 1.0
    positions[:, 1:, 2] = np.array([-0.10, -0.02, 0.10, 0.20])
    initial = positions.copy()
    motion = CanonicalMotion(
        positions, names, 50.0,
        metadata={
            "foot_contact_probs": np.ones((4, 4)),
            "foot_contact_channel_order": (
                "left_heel", "right_heel", "left_toe", "right_toe",
            ),
        },
    )
    report = align_motion_to_scene_support(
        motion, TerrainField(floor_z=0.0),
        SceneTransform(np.eye(3), 1.0, np.zeros(3)),
        {"min_samples": 4, "max_vertical_offset": 0.25, "max_residual_spread": 0.01},
    )
    assert report["status"] == "INCONSISTENT_EVIDENCE"
    np.testing.assert_allclose(motion.positions, initial)


def test_static_contact_locks_matching_world_and_asset_local_anchor():
    from types import SimpleNamespace
    from general_motion_retargeting.contact import SourceContactDetector

    class Terrain:
        floor_z = None

        def __init__(self):
            self.query = 0

        def support_surface(self, point):
            self.query += 1
            # The source point is stationary, but a tessellated surface query
            # may return a different equivalent closest triangle each frame.
            offset = 0.01 * self.query
            return SimpleNamespace(
                closest_point=np.array([offset, 0.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                signed_distance=0.0,
                surface_id="asset:patch_0",
                surface_type="mesh",
                supportable=True,
                triangle_id=self.query,
                barycentric=np.array([1.0, 0.0, 0.0]),
                asset_local_anchor=np.array([offset, 0.0, 0.0]),
                asset_local_normal=np.array([0.0, 0.0, 1.0]),
            )

    names = ["pelvis", "spine3", "left_hip", "right_hip", "left_heel"]
    positions = np.zeros((2, len(names), 3), dtype=float)
    positions[:, names.index("pelvis"), 2] = 1.0
    positions[:, names.index("spine3"), 2] = 1.3
    positions[:, names.index("left_hip")] = [0.0, 0.1, 1.0]
    positions[:, names.index("right_hip")] = [0.0, -0.1, 1.0]
    motion = CanonicalMotion(positions, names, 50.0)
    plan = SourceContactDetector(Terrain(), 50.0).detect(
        [{
            "pelvis": frame["pelvis"],
            "spine3": frame["spine3"],
            "left_hip": frame["left_hip"],
            "right_hip": frame["right_hip"],
            "left_heel": frame["left_heel"],
        } for frame in motion.named_positions()]
    )
    first = plan.per_frame_states[0]["contacts"]["left_heel"]
    second = plan.per_frame_states[1]["contacts"]["left_heel"]
    np.testing.assert_allclose(
        second["tangent_anchor_solver"], first["tangent_anchor_solver"]
    )
    np.testing.assert_allclose(
        second["asset_local_anchor"], first["asset_local_anchor"]
    )


def test_unnamed_foot_scores_do_not_create_a_guessed_heel_toe_schema():
    """Four scores without names are event evidence, never a hidden mapping."""
    from general_motion_retargeting.contact.detector import build_contact_plan
    from general_motion_retargeting.input.motion_scene_alignment import align_motion_to_scene_support
    from general_motion_retargeting.terrain_geometry import SceneTransform, TerrainField

    names = ["pelvis", "left_heel", "right_heel", "left_toe", "right_toe"]
    positions = np.zeros((2, len(names), 3), dtype=float)
    positions[:, 0, 2] = 1.0
    positions[:, 1:, 2] = 0.05
    motion = CanonicalMotion(
        positions, names, 50.0,
        metadata={"foot_contact_probs": np.ones((2, 4))},
    )
    report = align_motion_to_scene_support(
        motion, TerrainField(floor_z=0.0),
        SceneTransform(np.eye(3), 1.0, np.zeros(3)),
        {"min_samples": 4, "geometry_search_distance": 0.08},
    )
    assert report["status"] == "APPLIED"
    assert report["label_schema"] == "geometry_stationary"
    plan = build_contact_plan(motion, TerrainField(floor_z=0.0))
    assert plan.per_frame_states[0]["contacts"]["left_heel"]["source_contact_label"] is None


def test_declared_label_schema_can_cover_a_bounded_surface_marker_offset():
    """Verified labels assist timing without changing the terrain geometry."""
    from general_motion_retargeting.contact.detector import build_contact_plan
    names = ["pelvis", "left_heel", "right_heel", "left_toe", "right_toe"]
    positions = np.zeros((1, len(names), 3), dtype=float)
    positions[0, 0, 2] = 1.0
    # A small, bounded marker offset is still allowed to use the declared
    # label; the deep-penetration case is covered separately below.
    positions[0, 1:, 2] = -0.01
    motion = CanonicalMotion(
        positions, names, 50.0,
        metadata={
            "foot_contact_probs": np.ones((1, 4)),
            "foot_contact_channel_order": (
                "left_heel", "left_toe", "right_heel", "right_toe",
            ),
        },
    )
    plan = build_contact_plan(
        motion, TerrainField(floor_z=0.0),
        config={"label_contact_max_distance": 0.10},
    )
    frame = plan.per_frame_states[0]
    assert frame["support_state"] == "SUPPORTED"
    assert frame["contacts"]["left_heel"]["label_contact"]
    assert frame["contacts"]["left_heel"]["source_contact_label"] == 1.0


def test_low_score_verified_label_keeps_contact_state():
    """A valid source label is not discarded when velocity lowers its score."""
    from general_motion_retargeting.contact.detector import build_contact_plan

    names = ["pelvis", "left_heel", "right_heel", "left_toe", "right_toe"]
    positions = np.zeros((2, len(names), 3), dtype=float)
    positions[:, 0] = [0.0, 0.0, 1.0]
    positions[0, 1:] = [0.0, 0.0, 0.0]
    # The second frame moves the labeled markers at 0.19 m/s.  This passes
    # the normal-speed gate but makes the continuous label score below the
    # activation threshold, which is the regression case for this policy.
    positions[1, 1:] = [0.0, 0.0, 0.0038]
    motion = CanonicalMotion(
        positions,
        names,
        50.0,
        metadata={
            "foot_contact_probs": np.ones((2, 4)),
            "foot_contact_channel_order": (
                "left_heel", "left_toe", "right_heel", "right_toe",
            ),
        },
    )
    plan = build_contact_plan(
        motion,
        TerrainField(floor_z=0.0),
        config={
            "contact_activation_score": 0.15,
            "normal_speed_limit": 0.20,
            "contact_blend_frames": 1,
        },
    )
    item = plan.per_frame_states[1]["contacts"]["left_heel"]
    assert item["label_contact"] is True
    assert item["score"] < 0.15
    assert item["score"] > 0.0
    assert item["state"] != "NONE"


def test_deep_solid_penetration_is_not_desired_contact():
    terrain = TerrainField([BoxPrimitive("seat", [0, 0, 0.5], [1, 1, 0.5], np.eye(3))], floor_z=None)
    frame = {"pelvis": np.array([0.0, 0.0, 0.25]), "left_heel": np.array([0.0, 0.0, 0.25])}
    plan = SourceContactDetector(terrain, 50.0, {"max_allowed_penetration": 0.02}).detect([frame])
    assert plan.per_frame_states[0]["contacts"]["left_heel"]["state"] == "NONE"
    assert plan.per_frame_states[0]["support_state"] == "UNKNOWN"
    assert not plan.per_frame_states[0]["support_expected"]


def test_declared_label_cannot_override_deep_floor_penetration():
    from general_motion_retargeting.contact.detector import build_contact_plan

    names = ["pelvis", "left_foot", "right_foot", "left_toe", "right_toe"]
    positions = np.zeros((1, len(names), 3), dtype=float)
    positions[0, 0] = [0.0, 0.0, 1.0]
    positions[0, 1:] = [0.0, 0.0, -0.08]
    motion = CanonicalMotion(
        positions,
        names,
        50.0,
        metadata={
            "foot_contact_probs": np.ones((1, 4)),
            "foot_contact_channel_order": (
                "left_heel", "left_toe", "right_heel", "right_toe",
            ),
        },
    )
    plan = build_contact_plan(
        motion,
        TerrainField(floor_z=0.0),
        config={"label_contact_max_distance": 0.18, "max_allowed_penetration": 0.02},
    )
    frame = plan.per_frame_states[0]
    assert all(
        frame["contacts"][channel]["state"] == "NONE"
        for channel in ("left_heel", "left_toe", "right_heel", "right_toe")
    )
    assert frame["support_state"] == "UNKNOWN"
    assert not frame["support_expected"]


def test_declared_labels_calibrate_stable_landmark_surface_offset():
    """A learned observation bias does not move either motion or terrain."""
    from general_motion_retargeting.contact.detector import build_contact_plan

    names = ["pelvis", "left_heel", "left_toe", "right_heel", "right_toe"]
    positions = np.zeros((12, len(names), 3), dtype=float)
    positions[:, 0, 2] = 1.0
    positions[:, 1:, 2] = -0.08
    original = positions.copy()
    motion = CanonicalMotion(
        positions,
        names,
        50.0,
        metadata={
            "foot_contact_probs": np.ones((12, 4)),
            "foot_contact_channel_order": (
                "left_heel", "left_toe", "right_heel", "right_toe",
            ),
        },
    )
    terrain = TerrainField(floor_z=0.0)
    plan = build_contact_plan(
        motion,
        terrain,
        config={
            "max_allowed_penetration": 0.02,
            "label_contact_max_distance": 0.18,
            "label_calibrated_max_residual": 0.02,
            "contact_blend_frames": 1,
            "label_surface_offset_calibration": {
                "enabled": True,
                "min_samples": 8,
                "cluster_width": 0.02,
                "min_inlier_ratio": 0.5,
                "max_abs_offset": 0.20,
                "normal_speed": 0.08,
                "tangent_speed": 0.08,
            },
        },
    )
    item = plan.per_frame_states[-1]["contacts"]["left_heel"]
    assert item["label_contact"]
    assert item["state"] == "STATIC"
    assert item["raw_contact_distance"] == pytest.approx(-0.08)
    assert item["landmark_surface_offset"] == pytest.approx(-0.08)
    assert item["contact_distance"] == pytest.approx(0.0)
    assert plan.per_frame_states[-1]["support_state"] == "SUPPORTED"
    assert plan.metadata["label_surface_offsets"]["left_heel"]["inlier_count"] == 12
    np.testing.assert_allclose(motion.positions, original)
    assert terrain.floor_z == 0.0


def test_robot_static_tangent_anchor_locks_realized_robot_point():
    from types import SimpleNamespace
    from general_motion_retargeting.wholebody_omni_gmr_v5 import WholeBodyRetargetSolver

    class Point:
        def value(self, configuration):
            return np.asarray(configuration.point, dtype=float)

    fake = SimpleNamespace(
        configuration=SimpleNamespace(point=np.array([0.3, -0.2, 0.01])),
        contact=SimpleNamespace(points={"left_heel": Point()}),
        scene_model=None,
        _robot_static_anchors={},
    )
    source_anchor = np.array([1.0, 2.0, 0.0])
    frame = {
        "contacts": {
            "left_heel": {
                "state": "STATIC",
                "object_id": "floor",
                "surface_id": "floor",
                "anchor_surface_id": "floor",
                "tangent_anchor_solver": source_anchor,
            }
        }
    }
    first = WholeBodyRetargetSolver._bind_robot_static_anchors(fake, frame, 0.0)
    np.testing.assert_allclose(
        first["contacts"]["left_heel"]["tangent_anchor_solver"],
        [0.3, -0.2, 0.01],
    )
    np.testing.assert_allclose(
        first["contacts"]["left_heel"]["source_tangent_anchor_solver"],
        source_anchor,
    )
    fake.configuration.point = np.array([0.5, 0.4, 0.01])
    second = WholeBodyRetargetSolver._bind_robot_static_anchors(fake, frame, 0.02)
    np.testing.assert_allclose(
        second["contacts"]["left_heel"]["tangent_anchor_solver"],
        [0.3, -0.2, 0.01],
    )
    sliding = {"contacts": {"left_heel": {"state": "SLIDING"}}}
    WholeBodyRetargetSolver._bind_robot_static_anchors(fake, sliding, 0.04)
    third = WholeBodyRetargetSolver._bind_robot_static_anchors(fake, frame, 0.06)
    np.testing.assert_allclose(
        third["contacts"]["left_heel"]["tangent_anchor_solver"],
        [0.5, 0.4, 0.01],
    )


def test_near_surface_fast_transfer_is_flight_not_unknown():
    from general_motion_retargeting.contact.detector import build_contact_plan

    names = ["pelvis", "left_foot", "right_foot", "left_toe", "right_toe"]
    positions = np.zeros((2, len(names), 3), dtype=float)
    positions[:, 0] = [0.0, 0.0, 1.0]
    positions[:, 1] = [0.0, 0.1, 0.3]
    positions[:, 2] = [0.0, -0.1, 0.3]
    positions[0, 3] = [0.0, 0.1, 0.03]
    positions[1, 3] = [0.0, 0.1, 0.04]
    positions[:, 4] = [0.0, -0.1, 0.20]
    motion = CanonicalMotion(positions, names, 50.0)
    plan = build_contact_plan(motion, TerrainField(floor_z=0.0))
    # The second-frame toe is close to the floor but moving at 0.5 m/s in the
    # normal direction, so it is a transfer/flight frame, not valid support.
    assert plan.per_frame_states[1]["support_state"] == "FLIGHT"


def test_missing_toe_is_explicitly_unavailable_not_pelvis_fallback():
    from general_motion_retargeting.contact.detector import SourceContactDetector
    terrain = TerrainField(floor_z=0.0)
    frame = {
        "pelvis": np.array([0.0, 0.0, 1.0]),
        "left_heel": np.array([0.0, 0.0, 0.01]),
    }
    plan = SourceContactDetector(terrain, 50.0).detect([frame])
    toe = plan.per_frame_states[0]["contacts"]["left_toe"]
    assert toe["state"] == "NONE"
    assert toe["provenance"] == "missing_landmark"


def test_build_contact_plan_does_not_fabricate_missing_heel_or_toe():
    from general_motion_retargeting.contact.detector import build_contact_plan
    terrain = TerrainField(floor_z=0.0)
    names = ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
             "left_ankle", "right_ankle", "left_shoulder", "right_shoulder",
             "left_elbow", "right_elbow", "left_wrist", "right_wrist", "spine3"]
    positions = np.zeros((1, len(names), 3), dtype=float)
    positions[0, names.index("pelvis")] = [0, 0, 1]
    positions[0, names.index("spine3")] = [0, 0, 1.4]
    positions[0, names.index("left_hip")] = [0.1, 0, 0.9]
    positions[0, names.index("right_hip")] = [-0.1, 0, 0.9]
    motion = CanonicalMotion(positions, names, 50.0)
    plan = build_contact_plan(motion, terrain)
    for channel in ("left_heel", "right_heel", "left_toe", "right_toe"):
        assert plan.per_frame_states[0]["contacts"][channel]["provenance"] == "missing_landmark"


def test_explicit_motion_format_is_forwarded_to_adapter(tmp_path):
    from general_motion_retargeting.motion_adapters import load_canonical_motion
    path = tmp_path / "motion.npz"
    np.savez(path, positions=np.zeros((1, 1, 3)), joint_names=np.asarray(["pelvis"]), fps=np.asarray(50.0))
    loaded = load_canonical_motion(path, motion_format="canonical_npz")
    assert loaded.source_format == "canonical_npz"


def test_large_mesh_bvh_returns_exact_barycentric_contract():
    """The large-mesh branch must not fall back to centroid-only data."""
    from general_motion_retargeting.scene.loader import MeshSceneField

    # More than the BVH threshold, with the queried triangle deliberately
    # placed far from the first centroid shortlist entries.
    triangles = []
    for index in range(8200):
        x = float(index + 10)
        triangles.append([[x, -1.0, 0.0], [x + 0.4, -1.0, 0.0], [x, -0.6, 0.0]])
    triangles.append([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    triangles = np.asarray(triangles, dtype=float)
    vertices = triangles.reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    field = MeshSceneField(vertices, faces, "large", floor_z=None)
    hit = field.nearest_surface(np.array([0.2, 0.2, 0.1]))
    assert hit.triangle_id == 8200
    np.testing.assert_allclose(hit.closest_point, [0.2, 0.2, 0.0])
    np.testing.assert_allclose(hit.barycentric, [0.6, 0.2, 0.2])


def test_large_mesh_support_query_hits_triangle_not_centroid():
    # The support point lies far from the triangle centroid in XY; the exact
    # ray query must still recover the upward face after the BVH threshold.
    triangles = []
    for index in range(8200):
        x = float(index + 10)
        triangles.append([[x, -1.0, 0.0], [x + 0.4, -1.0, 0.0], [x, -0.6, 0.0]])
    triangles.append([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    triangles = np.asarray(triangles, dtype=float)
    vertices = triangles.reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    field = MeshSceneField(vertices, faces, "large", floor_z=None)
    hit = field.support_surface(np.array([0.2, 0.2, 0.1]))
    assert hit.supportable
    assert hit.triangle_id == 8200
    np.testing.assert_allclose(hit.closest_point, [0.2, 0.2, 0.0])


def test_self_collision_ancestor_walk_stops_at_world_body():
    import mujoco as mj
    from general_motion_retargeting.validation.kinematic import _ancestor_related

    model = mj.MjModel.from_xml_path("assets/ne01/ne01_desktop_assets_wholebody_omni_gmr_v2.xml")
    assert _ancestor_related(model, 1, 1)
    assert not _ancestor_related(model, 1, 0)


def test_dynamic_object_rotation_track_clamps_to_human_timeline():
    import importlib.util
    from pathlib import Path
    module_spec = importlib.util.spec_from_file_location(
        "v5_retarget_cli", Path(__file__).parents[1] / "scripts" / "retarget.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    _pose_trajectory_from_obj = module._pose_trajectory_from_obj
    from general_motion_retargeting.terrain_geometry import SceneTransform

    obj = {
        "obj_t": np.zeros((2, 3)),
        "obj_R": np.tile(np.eye(3), (2, 1, 1)),
    }
    trajectory = _pose_trajectory_from_obj(
        obj, 50.0, SceneTransform(np.eye(3), 1.0, np.zeros(3)),
        np.array([0.0, 0.02, 0.04]),
    )
    assert trajectory is not None
    assert len(trajectory.rotations_wxyz) == 3
    np.testing.assert_allclose(trajectory.rotations_wxyz[-1], trajectory.rotations_wxyz[-2])


def test_v5_external_override_config_resolves_repository_robot_asset(tmp_path):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).parents[1]
    module_spec = importlib.util.spec_from_file_location(
        "v5_retarget_external_config", root / "scripts" / "retarget.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    resolved = module._configured_robot_xml(
        tmp_path / "override.json",
        {"robot_xml": "assets/ne01/ne01_desktop_assets_wholebody_omni_gmr_v2.xml"},
    )
    assert resolved == (root / "assets/ne01/ne01_desktop_assets_wholebody_omni_gmr_v2.xml").resolve()


def test_grail_baked_asset_scale_is_resolved_once(tmp_path):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    from general_motion_retargeting.core.schemas import SceneReference

    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "v5_retarget_scale_policy", root / "scripts" / "retarget.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    baked_path = tmp_path / "mesh_data" / "model.obj"
    baked_path.parent.mkdir()
    baked_path.write_text("# test placeholder\n", encoding="utf-8")
    reference = SceneReference(path=baked_path, metadata={})
    canonical = SimpleNamespace(scene={"obj_data": {}, "asset_scale_baked": None})
    assert module._asset_scale_baked(reference, canonical) is True

    explicit = SceneReference(path=baked_path, metadata={"asset_scale_baked": False})
    assert module._asset_scale_baked(explicit, canonical) is False
