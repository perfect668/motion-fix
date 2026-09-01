"""Terrain-aware contact inference using only source human motion and terrain."""

from __future__ import annotations

from typing import Any

import numpy as np

from .terrain_geometry import SceneTransform, TerrainField, TerrainSurfaceHit


def _proxy_points(frame: dict[str, np.ndarray], config: dict) -> dict[str, tuple[np.ndarray, str, bool]]:
    points: dict[str, tuple[np.ndarray, str, bool]] = {}
    foot_back_fraction = float(config.get("heel_proxy_back_fraction", 0.28))
    sole_offset = float(config.get("heel_proxy_sole_offset", 0.0))
    lateral = np.asarray(frame["LeftFoot"], dtype=float) - np.asarray(frame["RightFoot"], dtype=float)
    lateral /= max(float(np.linalg.norm(lateral)), 1e-12)
    for side, title in (("left", "Left"), ("right", "Right")):
        foot = np.asarray(frame[f"{title}Foot"], dtype=float)
        toe = np.asarray(frame[f"{title}ToeBase"], dtype=float)
        foot_vector = toe - foot
        # FootMod duplicates Foot in the verified sequence.  Construct a heel
        # proxy along the measured foot axis and retain provenance explicitly.
        sole_normal = np.cross(foot_vector, lateral)
        sole_normal /= max(float(np.linalg.norm(sole_normal)), 1e-12)
        if sole_normal[2] < 0.0:
            sole_normal = -sole_normal
        heel = foot - foot_back_fraction * foot_vector - sole_offset * sole_normal
        points[f"{side}_heel"] = (heel, "Foot_to_ToeBase_surface_proxy", True)
        points[f"{side}_toe"] = (toe, "ToeBase", True)
        points[f"{side}_palm"] = (np.asarray(frame[f"{title}Hand"], dtype=float), "Hand", False)
        knee = np.asarray(frame[f"{title}Leg"], dtype=float)
        ankle = foot
        points[f"{side}_knee"] = (knee, "Leg", False)
        points[f"{side}_shin"] = (0.55 * knee + 0.45 * ankle, "Leg_Foot_proxy", False)
    return points


def _surface_for_channel(terrain: TerrainField, point: np.ndarray, support: bool) -> TerrainSurfaceHit:
    return terrain.support_surface(point) if support else terrain.nearest_surface(point)


def build_terrain_contact_schedule(
    source_frames: list[dict[str, np.ndarray]],
    source_terrain: TerrainField,
    scene_transform: SceneTransform,
    joint_mapping: dict,
    fps: float,
    config: dict,
) -> list[dict[str, Any]]:
    del joint_mapping  # Validation and semantic construction happen before this stage.
    terrain = source_terrain.transform(scene_transform)
    dt = 1.0 / max(float(fps), 1e-9)
    enter = float(config.get("contact_enter_distance", 0.03))
    exit_distance = float(config.get("contact_exit_distance", 0.055))
    static_speed = float(config.get("static_tangent_speed", 0.08))
    normal_speed_limit = float(config.get("normal_speed_limit", 0.20))
    switch_hysteresis = float(config.get("surface_switch_hysteresis", 0.015))
    blend_frames = max(1, int(config.get("contact_blend_frames", 7)))
    alpha = 1.0 / blend_frames

    previous: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    active: dict[str, bool] = {}
    locked_surface: dict[str, str | None] = {}
    locked_hit: dict[str, TerrainSurfaceHit] = {}
    schedule = []
    for source_frame in source_frames:
        contacts = {}
        for channel, (point_source, provenance, support) in _proxy_points(source_frame, config).items():
            point_solver = scene_transform.transform_points(point_source)
            hit = _surface_for_channel(terrain, point_solver, support)
            old = previous.get(channel)
            velocity = np.zeros(3) if old is None else (point_solver - old) / dt
            normal_speed = float(velocity @ hit.normal)
            tangent_velocity = velocity - normal_speed * hit.normal
            tangential_speed = float(np.linalg.norm(tangent_velocity))
            was_active = active.get(channel, False)
            is_active = hit.signed_distance <= (exit_distance if was_active else enter)
            active[channel] = is_active

            previous_surface = locked_surface.get(channel)
            if was_active and previous_surface is not None and hit.surface_id != previous_surface:
                prior = locked_hit[channel]
                prior_distance = float(prior.normal @ (point_solver - prior.closest_point))
                if prior_distance <= hit.signed_distance + switch_hysteresis:
                    hit = TerrainSurfaceHit(
                        signed_distance=prior_distance,
                        closest_point=point_solver - prior_distance * prior.normal,
                        normal=prior.normal,
                        surface_id=prior.surface_id,
                        surface_type=prior.surface_type,
                        supportable=prior.supportable,
                    )
            if is_active:
                locked_surface[channel] = hit.surface_id
                locked_hit[channel] = hit
            else:
                locked_surface[channel] = None

            distance_score = float(np.clip((exit_distance - hit.signed_distance) / max(exit_distance, 1e-9), 0.0, 1.0))
            normal_score = float(np.clip(1.0 - abs(normal_speed) / max(normal_speed_limit, 1e-9), 0.0, 1.0))
            target = distance_score * normal_score if is_active else 0.0
            score = scores.get(channel, 0.0) + alpha * (target - scores.get(channel, 0.0))
            scores[channel] = float(np.clip(score, 0.0, 1.0))
            state = "NONE"
            if scores[channel] > float(config.get("state_score_threshold", 0.15)):
                state = "STATIC" if tangential_speed < static_speed else "SLIDING"
            contacts[channel] = {
                "score": scores[channel],
                "state": state,
                "human_point_source": point_source.copy(),
                "human_point_solver": point_solver.copy(),
                "human_point_provenance": provenance,
                "surface_point_source": scene_transform.inverse().transform_points(hit.closest_point),
                "surface_point_solver": hit.closest_point.copy(),
                "surface_normal_source": scene_transform.inverse().transform_normals(hit.normal),
                "surface_normal_solver": hit.normal.copy(),
                "surface_id": hit.surface_id,
                "surface_type": hit.surface_type,
                "signed_distance": float(hit.signed_distance),
                "normal_speed": normal_speed,
                "tangential_speed": tangential_speed,
            }
            previous[channel] = point_solver.copy()

        flat_foot = {}
        max_angle = np.deg2rad(float(config.get("flat_foot_max_normal_angle_deg", 12.0)))
        min_score = float(config.get("flat_foot_min_score", 0.45))
        for side in ("left", "right"):
            heel, toe = contacts[f"{side}_heel"], contacts[f"{side}_toe"]
            dot = float(np.clip(np.dot(heel["surface_normal_solver"], toe["surface_normal_solver"]), -1.0, 1.0))
            valid = (
                heel["surface_id"] == toe["surface_id"]
                and np.arccos(dot) <= max_angle
                and heel["score"] >= min_score
                and toe["score"] >= min_score
            )
            flat_foot[side] = float(min(heel["score"], toe["score"])) if valid else 0.0
        schedule.append({"contacts": contacts, "flat_foot": flat_foot})
    return schedule
