"""Human whole-body ground-contact features for the Omni GMR variant.

The first implementation deliberately uses SMPL-X joints plus oriented local
offsets instead of a dense mesh query.  It is fast enough for dataset
conversion and, unlike a robot-derived contact estimate, is independent of the
current IK result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


def _point(frame: dict, name: str) -> np.ndarray:
    return np.asarray(frame[name][0], dtype=float)


def _oriented_offset(frame: dict, name: str, offset: list[float]) -> np.ndarray:
    pos = _point(frame, name)
    quat = np.asarray(frame[name][1], dtype=float)
    return pos + R.from_quat(quat, scalar_first=True).apply(np.asarray(offset, dtype=float))


def human_surface_points(frame: dict, surface_regions: dict[str, dict]) -> dict[str, np.ndarray]:
    """Return approximate human contact-surface points in world coordinates."""
    points: dict[str, np.ndarray] = {}
    for region, spec in surface_regions.items():
        kind = spec.get("human_kind", "joint")
        joints = spec.get("human_joints", [])
        if not joints or any(name not in frame for name in joints):
            continue
        if kind == "joint":
            point = _point(frame, joints[0])
        elif kind == "midpoint":
            point = sum((_point(frame, name) for name in joints), np.zeros(3)) / len(joints)
        elif kind == "oriented_offset":
            point = _oriented_offset(frame, joints[0], spec.get("human_offset", [0, 0, 0]))
        else:
            raise ValueError(f"Unknown human surface-point kind: {kind}")
        points[region] = point
    return points


def build_whole_body_contact_schedule(
    frames: list[dict],
    surface_regions: dict[str, dict],
    *,
    fps: float,
    floor_z: float,
    contact_height: float = 0.03,
    release_height: float = 0.05,
    vertical_speed_limit: float = 0.35,
    static_speed_limit: float = 0.08,
    smoothing: float = 0.35,
) -> list[dict[str, Any]]:
    """Precompute human contact scores and STATIC/SLIDING/NONE labels.

    ``floor_z`` must be estimated before this call from the human reference
    motion.  No robot pose is used here, preventing contact/IK feedback loops.
    """
    dt = 1.0 / max(float(fps), 1e-6)
    previous_points: dict[str, np.ndarray] = {}
    scores = {name: 0.0 for name in surface_regions}
    active = {name: False for name in surface_regions}
    schedule: list[dict[str, Any]] = []
    for frame in frames:
        points = human_surface_points(frame, surface_regions)
        contacts: dict[str, dict[str, Any]] = {}
        for name, point in points.items():
            previous = previous_points.get(name)
            velocity = np.zeros(3) if previous is None else (point - previous) / dt
            height = float(point[2] - floor_z)
            if active[name]:
                active[name] = height < release_height
            else:
                active[name] = height < contact_height
            height_score = np.clip((contact_height - height) / max(contact_height, 1e-6), 0.0, 1.0)
            vertical_score = np.clip(
                1.0 - abs(float(velocity[2])) / max(vertical_speed_limit, 1e-6), 0.0, 1.0
            )
            raw = float(height_score * vertical_score) if active[name] else 0.0
            scores[name] = (1.0 - smoothing) * scores[name] + smoothing * raw
            horizontal_speed = float(np.linalg.norm(velocity[:2]))
            state = "NONE"
            if scores[name] > 0.15:
                state = "STATIC" if horizontal_speed < static_speed_limit else "SLIDING"
            contacts[name] = {
                "score": float(scores[name]),
                "state": state,
                "point": point.copy(),
                "height": height,
                "horizontal_speed": horizontal_speed,
            }
            previous_points[name] = point.copy()
        schedule.append({"contacts": contacts})
    return schedule


def _foot_temporal_points(frame: dict, side: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Return independent human heel/forefoot points and their provenance."""
    heel_name = f"{side}_heel"
    toe_names = [f"{side}_big_toe", f"{side}_small_toe"]
    if heel_name in frame and any(name in frame for name in toe_names):
        heel = _point(frame, heel_name)
        toes = [_point(frame, name) for name in toe_names if name in frame]
        return heel, np.mean(toes, axis=0), "smplx_surface"

    foot_name = f"{side}_foot"
    if foot_name not in frame:
        raise KeyError(f"Missing {foot_name} for temporal foot contact")
    # Position-only prepared files do not carry SMPL-X surface landmarks.
    # Keep heel/toe channels distinct using oriented foot-local proxies.
    heel = _oriented_offset(frame, foot_name, [-0.075, 0.0, -0.025])
    toe = _oriented_offset(frame, foot_name, [0.115, 0.0, -0.025])
    return heel, toe, "foot_local_proxy"


def build_foot_temporal_contact_schedule(
    frames: list[dict],
    *,
    fps: float,
    floor_z: float,
    enter_height: float = 0.035,
    exit_height: float = 0.055,
    horizontal_speed_limit: float = 0.18,
    vertical_speed_limit: float = 0.20,
    smoothing_frames: int = 7,
) -> list[dict[str, Any]]:
    """Build separate heel/toe source-contact channels at the output rate."""
    dt = 1.0 / max(float(fps), 1e-6)
    alpha = 1.0 / max(int(smoothing_frames), 1)
    previous: dict[str, np.ndarray] = {}
    active = {f"{side}_{part}": False for side in ("left", "right") for part in ("heel", "toe")}
    scores = {name: 0.0 for name in active}
    schedule: list[dict[str, Any]] = []
    for frame in frames:
        result: dict[str, Any] = {}
        for side in ("left", "right"):
            heel, toe, source = _foot_temporal_points(frame, side)
            for part, point in (("heel", heel), ("toe", toe)):
                name = f"{side}_{part}"
                old = previous.get(name)
                velocity = np.zeros(3) if old is None else (point - old) / dt
                height = float(point[2] - floor_z)
                if active[name]:
                    active[name] = height < exit_height
                else:
                    active[name] = height < enter_height
                height_score = np.clip((exit_height - height) / max(exit_height - enter_height, 1e-6), 0.0, 1.0)
                xy_score = np.clip(1.0 - np.linalg.norm(velocity[:2]) / max(horizontal_speed_limit, 1e-6), 0.0, 1.0)
                z_score = np.clip(1.0 - abs(float(velocity[2])) / max(vertical_speed_limit, 1e-6), 0.0, 1.0)
                target = float(height_score * xy_score * z_score) if active[name] else 0.0
                scores[name] += alpha * (target - scores[name])
                result[name] = {
                    "score": float(np.clip(scores[name], 0.0, 1.0)),
                    "height": height,
                    "xy_speed": float(np.linalg.norm(velocity[:2])),
                    "vertical_speed": float(velocity[2]),
                    "point": point.copy(),
                    "source": source,
                }
                previous[name] = point.copy()
        schedule.append({"foot_temporal": result})
    return schedule
