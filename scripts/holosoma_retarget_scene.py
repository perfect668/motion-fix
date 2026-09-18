"""Retarget HoloSoMo position mocap with WholeBody Omni V4 scene collision.

The standalone HoloSoMo adapter owns input normalisation and dispatches
directly to the V4 solver.  The only scene-side addition is the MuJoCo
robot--scene collision limit.
The source terrain, interaction pool, contact schedule, and scene collision
mesh all receive the same SceneTransform.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import pickle
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V4_CONFIG = ROOT / "general_motion_retargeting/ik_configs/ne01_v4.json"


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_config(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if not raw.get("extends"):
        return raw
    parent = _load_config(path.parent / raw["extends"])
    return _deep_merge(parent, {key: value for key, value in raw.items() if key != "extends"})


def _scene_pose(config: dict, human_height: float | None) -> np.ndarray:
    scene = config["scene"]
    height = float(human_height or scene["default_human_height"])
    scale = float(scene["robot_height"]) / height
    rotation = np.asarray(scene["rotation"], dtype=float).reshape(3, 3)
    translation = np.asarray(scene["translation"], dtype=float).reshape(3)
    pose = np.eye(4)
    pose[:3, :3] = scale * rotation
    pose[:3, 3] = translation
    return pose


def main() -> None:
    import argparse

    import holosoma_retarget
    from general_motion_retargeting.scene_asset_loader import load_scene_asset
    from general_motion_retargeting.scene_mujoco import build_scene_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain", required=True, type=Path)
    parser.add_argument("--save_path", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=V4_CONFIG)
    parser.add_argument("--human_height", type=float, default=None)
    parser.add_argument(
        "--scene_cache", type=Path, default=ROOT / ".cache" / "scene_collision"
    )
    known, _ = parser.parse_known_args()
    known.terrain = known.terrain.expanduser().resolve()
    known.save_path = known.save_path.expanduser().resolve()
    known.config = known.config.expanduser().resolve()
    known.scene_cache = known.scene_cache.expanduser().resolve()

    config = _load_config(known.config)
    robot_xml = Path(config["robot_xml"])
    if not robot_xml.is_absolute():
        robot_xml = (ROOT / robot_xml).resolve()

    scene_mesh = load_scene_asset(
        known.terrain,
        {"object_id": known.terrain.stem},
        sample_count=int(
            config.get("scene_geometry", {})
            .get("surface_sampling", {})
            .get("maximum", 1024)
        ),
    )
    scene_mesh.object_pose = _scene_pose(config, known.human_height)

    output_dir = known.save_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_xml = output_dir / f".{known.save_path.stem}_combined.xml"
    combined = build_scene_model(
        robot_xml,
        scene_mesh,
        combined_xml,
        cache_root=known.scene_cache,
        return_info=True,
    )

    effective = copy.deepcopy(config)
    effective["robot_xml"] = str(combined_xml)
    effective["scene_collision"] = {
        **effective.get("scene_collision", {}),
        "backend": "hybrid",
        "scene_body_prefix": "scene_",
    }
    effective_path = output_dir / f".{known.save_path.stem}_v4_config.json"
    effective_path.write_text(json.dumps(effective, indent=2))

    old_argv = sys.argv[:]
    try:
        # Remove wrapper-only flags and replace any explicit config with the
        # generated config that points at the combined robot+scene model.
        forwarded: list[str] = []
        skip_next = False
        for token in old_argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if token in {"--scene_cache", "--config"}:
                skip_next = True
                continue
            forwarded.append(token)
        forwarded.extend(["--config", str(effective_path)])
        holosoma_retarget.main(forwarded)
    finally:
        sys.argv = old_argv
        effective_path.unlink(missing_ok=True)

    output_pkl = known.save_path.with_suffix(".pkl")
    with output_pkl.open("rb") as stream:
        payload = pickle.load(stream)
    payload["algorithm"] = "wholebody_omni_v4"
    payload["scene_collision"] = {
        "source_mesh": str(known.terrain.resolve()),
        "combined_xml": str(combined.xml_path),
        "visual_geom_count": int(combined.visual_geom_count),
        "collision_geom_count": int(combined.collision_geom_count),
        "scene_geom_ids": list(combined.scene_geom_ids),
    }
    with output_pkl.open("wb") as stream:
        pickle.dump(payload, stream)

    print(f"Saved V4 PKL: {output_pkl}")
    print(f"Combined MuJoCo model: {combined.xml_path}")
    print(
        "Scene geoms: "
        f"visual={combined.visual_geom_count}, collision={combined.collision_geom_count}"
    )


if __name__ == "__main__":
    main()
