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
    args = parser.parse_args()
    value = audit(args.result)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True)
    if args.json_path is not None:
        args.json_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
