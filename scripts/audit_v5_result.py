"""Audit a WholeBody V5 result against its exported validation contract.

This is intentionally read-only: it never modifies qpos or applies a viewer
offset.  It is useful for comparing solver runs after a configuration change.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def replay(path: Path) -> dict:
    """Independently replay qpos against the bound robot and terrain.

    This path intentionally ignores the payload's stored validation/status.
    It is a lightweight geometry replay (scene collision is included when a
    combined model is available) suitable for detecting tampered qpos or
    terrain metadata.
    """
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    import mujoco as mj
    import mink
    from general_motion_retargeting.terrain_geometry import TerrainField
    from general_motion_retargeting.v5_terrain_limit import TerrainNonPenetrationLimit

    xml_value = payload.get("robot_xml") or payload.get("model_path")
    if not xml_value:
        raise ValueError("result does not bind a robot_xml/model_path for replay")
    xml_path = Path(xml_value).expanduser()
    if not xml_path.is_absolute():
        xml_path = (path.parent / xml_path).resolve()
    model = mj.MjModel.from_xml_path(str(xml_path))
    qpos = np.asarray(payload.get("qpos", []), dtype=float)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"qpos shape {qpos.shape} does not match replay model nq={model.nq}")
    terrain_spec = payload.get("terrain")
    if not isinstance(terrain_spec, dict):
        terrain_spec = {"floor_z": 0.0, "primitives": []}
    terrain = TerrainField.from_spec(terrain_spec)
    limit_cfg = dict(payload.get("effective_config", {}).get("terrain_nonpenetration", {}))
    if not limit_cfg:
        limit_cfg = {"margin": float(payload.get("validation", {}).get("terrain_margin", 0.004)), "adaptive_activation": False}
    limit_cfg["adaptive_activation"] = False
    limit = TerrainNonPenetrationLimit(model, terrain, limit_cfg)
    configuration = mink.Configuration(model, qpos[0] if len(qpos) else model.qpos0)
    all_distances = []
    all_slacks = []
    for value in qpos:
        configuration.update(value)
        measurements = limit.measure_all(configuration)
        all_distances.extend(float(item["signed_distance"]) for item in measurements)
        all_slacks.extend(float(item["slack"]) for item in measurements)
    min_distance = float(min(all_distances, default=np.inf))
    min_slack = float(min(all_slacks, default=np.inf))
    max_penetration = float(max(0.0, -min_distance))
    max_margin_violation = float(max(0.0, -min_slack))
    return {
        "path": str(path.resolve()),
        "mode": "independent_replay",
        "frames": int(len(qpos)),
        "qpos_shape": list(qpos.shape),
        "minimum_signed_distance": min_distance,
        "minimum_terrain_slack": min_slack,
        "maximum_geometric_penetration": max_penetration,
        "maximum_margin_violation": max_margin_violation,
        "passed": bool(len(qpos) > 0 and max_margin_violation <= float(payload.get("effective_config", {}).get("validation", {}).get("max_terrain_violation", 0.002))),
    }


def audit(path: Path) -> dict:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    validation = dict(payload.get("validation", {}))
    checks = dict(validation.get("checks", {}))
    qpos = np.asarray(payload.get("qpos", []), dtype=float)
    joints = np.asarray(payload.get("joint_pos", payload.get("dof_pos", [])))
    diagnostics = payload.get("diagnostics", [])
    summary = validation.get("sequence_summary", {})
    return {
        "path": str(path.resolve()),
        "status": payload.get("status"),
        "algorithm": payload.get("algorithm"),
        "frames": int(len(qpos)),
        "qpos_shape": list(qpos.shape),
        "joint_pos_shape": list(joints.shape),
        "joint_count": len(payload.get("joint_names", [])),
        "dof_count": len(payload.get("dof_names", [])),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "validation": validation,
        "diagnostic_frames": len(diagnostics),
        "scene_transform": payload.get("scene_transform"),
        "scene_manifest": payload.get("scene_manifest"),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--replay", action="store_true", help="recompute geometry metrics from qpos/model/terrain")
    args = parser.parse_args()
    value = replay(args.result) if args.replay else audit(args.result)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True)
    if args.json_path is not None:
        args.json_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
