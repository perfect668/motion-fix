"""MuJoCo robot profile and stable model-index helpers for V5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco as mj
import numpy as np


def joint_names(model: mj.MjModel) -> tuple[str, ...]:
    return tuple(
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) or str(i)
        for i in range(model.njnt)
        if model.jnt_type[i] in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE)
    )


def joint_address(model: mj.MjModel, name: str) -> tuple[int, int]:
    jid = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, str(name)))
    if jid < 0:
        raise KeyError(f"Unknown robot joint: {name}")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


@dataclass(frozen=True)
class RobotProfile:
    name: str
    semantic_points: dict[str, dict[str, Any]]
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]

    @classmethod
    def from_model(cls, name: str, model: mj.MjModel, semantic_points: dict[str, Any]):
        bodies = tuple(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or str(i) for i in range(model.nbody))
        return cls(str(name), dict(semantic_points), joint_names(model), bodies)

    def validate(self, model: mj.MjModel) -> None:
        body_set = set(self.body_names)
        for semantic, spec in self.semantic_points.items():
            point = spec.get("robot", spec)
            if point.get("sites"):
                for site in point["sites"]:
                    if mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, str(site)) < 0:
                        raise KeyError(f"Robot profile point {semantic!r} references missing site {site!r}")
            elif str(point.get("body", point.get("robot_body", ""))) not in body_set:
                raise KeyError(f"Robot profile point {semantic!r} references missing body")
