"""Sequence-level terrain contact planning from source human/terrain relations."""
from __future__ import annotations
from typing import Any
import numpy as np
from .terrain_native_geometry import TerrainPatchMap, SupportPatch


def _unit(value: np.ndarray, fallback=(0.0, 0.0, 1.0)) -> np.ndarray:
    value = np.asarray(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        return np.asarray(fallback, dtype=float)
    return value / norm


def _source_sole(frame: dict[str, np.ndarray], side: str) -> dict[str, np.ndarray]:
    heel = np.asarray(frame[f"{side}_heel"], dtype=float)
    toe = np.asarray(frame[f"{side}_toe"], dtype=float)
    big = np.asarray(frame.get(f"{side}_big_toe", toe), dtype=float)
    small = np.asarray(frame.get(f"{side}_small_toe", toe), dtype=float)
    forward = _unit(toe - heel, fallback=(1.0, 0.0, 0.0))
    lateral = small - big
    lateral -= forward * float(lateral @ forward)
    lateral = _unit(lateral, fallback=(0.0, 1.0, 0.0))
    normal = _unit(np.cross(forward, lateral))
    if normal[2] < 0.0:
        normal = -normal
    return {
        "heel": heel,
        "toe": toe,
        "center": 0.5 * (heel + toe),
        "forward": forward,
        "normal": normal,
    }


def build_support_plan(
    source_frames: list[dict[str, np.ndarray]],
    patch_map: TerrainPatchMap,
    fps: float,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Infer whole-sequence support episodes from source human-terrain relations."""
    cfg = config or {}
    contact_distance = float(cfg.get("source_contact_distance", 0.055))
    edge_margin = float(cfg.get("source_edge_margin", 0.025))
    normal_angle = float(cfg.get("source_normal_angle_deg", 40.0))
    max_speed = float(cfg.get("source_contact_speed", 0.30))
    min_stance_frames = max(2, int(cfg.get("minimum_stance_frames", 3)))
    gap_fill_frames = max(0, int(cfg.get("contact_gap_fill_frames", 2)))
    swing_clearance = float(cfg.get("swing_clearance", 0.055))
    require_full_sole = bool(cfg.get("source_require_full_sole_support", True))
    dt = 1.0 / max(float(fps), 1e-9)
    count = len(source_frames)
    result = [{"left": {}, "right": {}} for _ in range(count)]

    for side in ("left", "right"):
        soles = [_source_sole(frame, side) for frame in source_frames]
        centers = np.asarray([item["center"] for item in soles])
        speeds = np.zeros(count)
        if count > 1:
            # Match HoloSoMo's foot-sticking semantics: tangential/XY speed
            # decides sticking; vertical motion is already gated by support
            # distance and must not create false sliding on a landing.
            delta = np.linalg.norm(np.diff(centers[:, :2], axis=0), axis=1) / dt
            speeds[1:] = delta
            speeds[0] = delta[0]
        raw_patch: list[str | None] = [None] * count
        raw_hit: list[tuple[SupportPatch, np.ndarray, float] | None] = [None] * count
        cos_limit = float(np.cos(np.deg2rad(normal_angle)))
        for index, sole in enumerate(soles):
            center_hit = patch_map.support_at(
                sole["center"], edge_margin=edge_margin,
                above_tolerance=contact_distance, max_gap=contact_distance,
            )
            if center_hit is None:
                continue
            patch, surface, gap = center_hit
            if require_full_sole:
                heel_hit = patch_map.support_at(
                    sole["heel"], edge_margin=edge_margin,
                    above_tolerance=contact_distance, max_gap=contact_distance,
                )
                toe_hit = patch_map.support_at(
                    sole["toe"], edge_margin=edge_margin,
                    above_tolerance=contact_distance, max_gap=contact_distance,
                )
                if heel_hit is None or toe_hit is None:
                    continue
                if heel_hit[0].patch_id != patch.patch_id or toe_hit[0].patch_id != patch.patch_id:
                    continue
            if float(sole["normal"] @ patch.normal) < cos_limit:
                continue
            if speeds[index] > max_speed:
                continue
            raw_patch[index] = patch.patch_id
            raw_hit[index] = (patch, surface, gap)

        for index in range(1, count - 1):
            if raw_patch[index] is not None:
                continue
            for width in range(1, gap_fill_frames + 1):
                left = index - width
                right = index + width
                if left < 0 or right >= count:
                    continue
                if raw_patch[left] is not None and raw_patch[left] == raw_patch[right]:
                    raw_patch[index] = raw_patch[left]
                    patch = patch_map.by_id[raw_patch[index]]
                    surface = patch.plane_point(soles[index]["center"][:2])
                    gap = float(patch.normal @ (soles[index]["center"] - surface))
                    raw_hit[index] = (patch, surface, gap)
                    break

        episodes: list[tuple[int, int, str]] = []
        start = 0
        while start < count:
            patch_id = raw_patch[start]
            if patch_id is None:
                start += 1
                continue
            end = start
            while end + 1 < count and raw_patch[end + 1] == patch_id:
                end += 1
            if end - start + 1 >= min_stance_frames:
                episodes.append((start, end, patch_id))
            else:
                for i in range(start, end + 1):
                    raw_patch[i] = None
                    raw_hit[i] = None
            start = end + 1

        for start, end, patch_id in episodes:
            patch = patch_map.by_id[patch_id]
            centers_episode = np.asarray([soles[i]["center"] for i in range(start, end + 1)])
            anchor_xy = np.median(centers_episode[:, :2], axis=0)
            anchor = patch.plane_point(anchor_xy)
            for index in range(start, end + 1):
                sole = soles[index]
                result[index][side] = {
                    "mode": "stance",
                    "patch_id": patch_id,
                    "surface_point": patch.plane_point(sole["center"][:2]),
                    "surface_normal": patch.normal.copy(),
                    "anchor": anchor.copy(),
                    "source_center": sole["center"].copy(),
                    "source_forward": sole["forward"].copy(),
                    "source_normal": sole["normal"].copy(),
                    "source_gap": float(raw_hit[index][2]) if raw_hit[index] is not None else 0.0,
                    "episode": [start, end],
                }

        for episode_index in range(len(episodes) - 1):
            _, prev_end, prev_id = episodes[episode_index]
            next_start, _, next_id = episodes[episode_index + 1]
            if next_start <= prev_end + 1:
                continue
            prev_patch = patch_map.by_id[prev_id]
            next_patch = patch_map.by_id[next_id]
            span = next_start - prev_end
            for index in range(prev_end + 1, next_start):
                phase = (index - prev_end) / float(span)
                sole = soles[index]
                source_center = sole["center"]
                top = max(
                    float(prev_patch.plane_point(source_center[:2])[2]),
                    float(next_patch.plane_point(source_center[:2])[2]),
                )
                arc = swing_clearance * float(np.sin(np.pi * phase))
                result[index][side] = {
                    "mode": "swing",
                    "previous_patch_id": prev_id,
                    "landing_patch_id": next_id,
                    "swing_phase": float(phase),
                    "clearance_floor_z": float(max(source_center[2], top + arc)),
                    "source_center": source_center.copy(),
                    "source_forward": sole["forward"].copy(),
                    "source_normal": sole["normal"].copy(),
                }

        for index in range(count):
            if result[index][side]:
                continue
            sole = soles[index]
            result[index][side] = {
                "mode": "free",
                "source_center": sole["center"].copy(),
                "source_forward": sole["forward"].copy(),
                "source_normal": sole["normal"].copy(),
            }
    return result


def apply_terrain_native_plan(
    schedule: list[dict[str, Any]],
    source_frames: list[dict[str, np.ndarray]],
    patch_map: TerrainPatchMap,
    fps: float,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    plan = build_support_plan(source_frames, patch_map, fps, config)
    for frame_record, frame_plan in zip(schedule, plan):
        frame_record["terrain_native"] = frame_plan
        contacts = frame_record.setdefault("contacts", {})
        flat = frame_record.setdefault("flat_foot", {})
        for side in ("left", "right"):
            foot = frame_plan[side]
            stance = foot.get("mode") == "stance"
            flat[side] = 1.0 if stance else 0.0
            for suffix in ("heel", "toe"):
                name = f"{side}_{suffix}"
                item = contacts.setdefault(name, {})
                item["terrain_native"] = foot
                if stance:
                    item.update({
                        "state": "STATIC",
                        "source_state": "STATIC",
                        "score": 1.0,
                        "object_id": patch_map.object_id,
                        "surface_id": str(foot["patch_id"]),
                        "surface_type": "support_patch",
                        "surface_point_solver": np.asarray(foot["surface_point"], dtype=float).copy(),
                        "surface_normal_solver": np.asarray(foot["surface_normal"], dtype=float).copy(),
                        "signed_distance": float(foot.get("source_gap", 0.0)),
                    })
                else:
                    item.update({"state": "NONE", "source_state": "NONE", "score": 0.0})
    return schedule
