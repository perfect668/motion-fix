import json
from pathlib import Path

import numpy as np
import pytest

from general_motion_retargeting.core.schemas import FloorPolicy, SceneModel, SceneRelation
from general_motion_retargeting.input.bundle_resolver import BundleResolver


def test_scene_model_rejects_non_z_up():
    with pytest.raises(ValueError):
        SceneModel(world_up=np.array([1., 0., 0.]))


def test_bundle_without_scene_is_explicit(tmp_path):
    motion = tmp_path / "walk.bvh"
    motion.write_text("HIERARCHY\n")
    bundle = BundleResolver().resolve(motion)
    assert bundle.scene_relation == SceneRelation.NO_SOURCE_SCENE
    assert bundle.source_scene is None and bundle.target_scene is None
    assert bundle.shared_coordinate_frame is False


def test_declared_missing_scene_does_not_fallback_to_floor(tmp_path):
    motion = tmp_path / "walk.bvh"
    motion.write_text("HIERARCHY\n")
    manifest = tmp_path / "job.json"
    manifest.write_text(json.dumps({"motion": {"path": "walk.bvh"}, "source_scene": {"path": "missing.obj"}}))
    with pytest.raises(FileNotFoundError):
        BundleResolver().resolve(motion, manifest=manifest)


def test_floor_policy_enum_is_versioned():
    assert FloorPolicy.AUTO.value == "auto"
