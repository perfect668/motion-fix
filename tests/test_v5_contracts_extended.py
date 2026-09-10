import numpy as np
import pytest

from general_motion_retargeting.core.schemas import PoseTrajectory, SceneAsset
from general_motion_retargeting.motion_adapters import CanonicalMotion, get_motion_adapter
from general_motion_retargeting.contact import SourceContactDetector
from general_motion_retargeting.terrain_geometry import BoxPrimitive
from general_motion_retargeting.input import resolver_for_motion
from general_motion_retargeting.scene.loader import CompositeSceneField, DynamicSceneField, MeshSceneField
from general_motion_retargeting.terrain_geometry import TerrainField


def test_adapter_registry_exposes_all_v5_formats():
    for name in ("smplx_npz", "grail_smplx_recon", "holosoma_global_positions", "bvh", "fbx"):
        assert get_motion_adapter(name).name == name


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


def test_deep_solid_penetration_is_not_desired_contact():
    terrain = TerrainField([BoxPrimitive("seat", [0, 0, 0.5], [1, 1, 0.5], np.eye(3))], floor_z=None)
    frame = {"pelvis": np.array([0.0, 0.0, 0.25]), "left_heel": np.array([0.0, 0.0, 0.25])}
    plan = SourceContactDetector(terrain, 50.0, {"max_allowed_penetration": 0.02}).detect([frame])
    assert plan.per_frame_states[0]["contacts"]["left_heel"]["state"] == "NONE"


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
