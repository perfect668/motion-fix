import json
from pathlib import Path

import numpy as np

from general_motion_retargeting.scene_asset_loader import SceneMesh, _mesh_digest, load_scene_asset
from general_motion_retargeting.scene_mujoco import build_scene_model


CUBE_VERTICES = np.array([
    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
])
CUBE_FACES = np.array([
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
    [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
])


def _write_obj(path: Path) -> None:
    lines = [*(f"v {x} {y} {z}" for x, y, z in CUBE_VERTICES),
             *("f " + " ".join(str(int(i) + 1) for i in face) for face in CUBE_FACES)]
    path.write_text("\n".join(lines), encoding="ascii")


def test_obj_loader_preserves_object_pose(tmp_path):
    path = tmp_path / "chair.obj"
    _write_obj(path)
    mesh = load_scene_asset(path, {"object_id": "seat", "position": [1, 2, 3], "scale": [2, 3, 4]})
    assert mesh.object_id == "seat"
    assert mesh.vertices.shape == (8, 3)
    assert mesh.faces.shape == (12, 3)
    np.testing.assert_allclose(mesh.object_pose[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(np.diag(mesh.object_pose)[:3], [2, 3, 4])


def test_usd_loader_merges_mesh_prims_and_parent_transforms(tmp_path):
    from pxr import Gf, Usd, UsdGeom

    path = tmp_path / "multi.usda"
    stage = Usd.Stage.CreateNew(str(path))
    parent = UsdGeom.Xform.Define(stage, "/chair")
    parent.AddTranslateOp().Set(Gf.Vec3d(2, 0, 0))
    for name, offset in (("seat", 0.0), ("back", 1.0)):
        mesh = UsdGeom.Mesh.Define(stage, f"/chair/{name}")
        mesh.CreatePointsAttr([Gf.Vec3f(offset, 0, 0), Gf.Vec3f(offset + 0.5, 0, 0), Gf.Vec3f(offset, 0.5, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()

    loaded = load_scene_asset(path)
    assert loaded.vertices.shape == (6, 3)
    assert loaded.faces.shape == (2, 3)
    np.testing.assert_allclose(loaded.vertices[:, 0].min(), 2.0)
    np.testing.assert_allclose(loaded.vertices[:, 0].max(), 3.5)


def test_decomposition_cache_key_includes_parameters(tmp_path):
    path = tmp_path / "chair.obj"
    _write_obj(path)
    mesh = SceneMesh(CUBE_VERTICES, CUBE_FACES, "chair", path)
    _, first = _mesh_digest(mesh, {"threshold": 0.05})
    _, second = _mesh_digest(mesh, {"threshold": 0.08})
    assert first != second


def test_combined_model_has_visual_and_collision_geoms(tmp_path, monkeypatch):
    import general_motion_retargeting.scene_mujoco as scene_mujoco

    source = tmp_path / "chair.obj"
    piece = tmp_path / "piece_000.obj"
    visual = tmp_path / "source_visual.obj"
    _write_obj(source)
    _write_obj(piece)
    _write_obj(visual)
    manifest = {
        "object_id": "chair",
        "visual_mesh": visual.name,
        "pieces": [piece.name],
        "piece_count": 1,
    }
    manifest_path = tmp_path / "collision_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(scene_mujoco, "decompose_cached", lambda *args, **kwargs: (manifest, tmp_path))

    robot_xml = tmp_path / "robot.xml"
    robot_xml.write_text(
        """<mujoco><worldbody><geom name="floor" type="plane" size="0 0 .1"/>
        <body name="robot" pos="0 0 1"><joint type="free"/><geom name="robot_geom" type="sphere" size=".1"/></body>
        </worldbody></mujoco>""",
        encoding="ascii",
    )
    mesh = SceneMesh(CUBE_VERTICES, CUBE_FACES, "chair", source, object_pose=np.diag([0.5, 0.5, 0.5, 1]))
    info = build_scene_model(robot_xml, mesh, tmp_path / "combined.xml", return_info=True)
    assert info.object_ids == ("chair",)
    assert info.visual_geom_count == 1
    assert info.collision_geom_count == 1
    assert len(info.scene_geom_ids) == 2
    payload = json.loads(info.manifest_path.read_text(encoding="utf-8"))
    assert payload["objects"][0]["convex_pieces"] == 1
    assert payload["scene_collision_geom_ids"]
