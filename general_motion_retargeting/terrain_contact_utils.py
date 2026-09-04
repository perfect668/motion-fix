"""Terrain-aware contact inference using only source human motion and terrain."""

from __future__ import annotations

from typing import Any

import numpy as np

from .terrain_geometry import SceneTransform, TerrainField, TerrainSurfaceHit


def _proxy_points(frame: dict[str, np.ndarray], config: dict) -> dict[str, tuple[np.ndarray, str, bool]]:
    points: dict[str, tuple[np.ndarray, str, bool]] = {}
    def pick(*names: str) -> np.ndarray | None:
        for name in names:
            if name in frame:
                return np.asarray(frame[name], dtype=float)
        return None
    foot_back_fraction = float(config.get("heel_proxy_back_fraction", 0.28))
    sole_offset = float(config.get("heel_proxy_sole_offset", 0.0))
    left_foot = pick("left_ankle", "LeftFoot", "left_foot")
    right_foot = pick("right_ankle", "RightFoot", "right_foot")
    if left_foot is None or right_foot is None:
        raise KeyError("Terrain contact inference requires left/right foot points")
    lateral = left_foot - right_foot
    lateral /= max(float(np.linalg.norm(lateral)), 1e-12)
    for side, title in (("left", "Left"), ("right", "Right")):
        foot = pick(f"{side}_ankle", f"{title}Foot", f"{side}_foot")
        toe_big = pick(f"{side}_big_toe")
        toe_small = pick(f"{side}_small_toe")
        toe = (0.5 * (toe_big + toe_small) if toe_big is not None and toe_small is not None
               else pick(f"{title}ToeBase", f"{side}_toe", f"{side}_toe_base"))
        if foot is None or toe is None:
            raise KeyError(f"Missing {side} foot/toe points")
        foot_vector = toe - foot
        # FootMod duplicates Foot in the verified sequence.  Construct a heel
        # proxy along the measured foot axis and retain provenance explicitly.
        sole_normal = np.cross(foot_vector, lateral)
        sole_normal /= max(float(np.linalg.norm(sole_normal)), 1e-12)
        if sole_normal[2] < 0.0:
            sole_normal = -sole_normal
        measured_heel = pick(f"{side}_heel")
        if measured_heel is not None:
            heel = measured_heel
            heel_provenance = "measured_heel"
        else:
            heel = foot - foot_back_fraction * foot_vector - sole_offset * sole_normal
            heel_provenance = "Foot_to_ToeBase_surface_proxy"
        toe_provenance = "measured_big_small_toe_midpoint" if toe_big is not None and toe_small is not None else "ToeBase"
        points[f"{side}_heel"] = (heel, heel_provenance, True)
        points[f"{side}_toe"] = (toe, toe_provenance, True)
        hand = pick(f"{title}Hand", f"{side}_hand", f"{side}_wrist")
        if hand is not None:
            points[f"{side}_palm"] = (hand, "hand_surface_proxy", False)
        knee = pick(f"{title}Leg", f"{side}_knee")
        if knee is None:
            continue
        ankle = foot
        points[f"{side}_knee"] = (knee, "Leg", False)
        points[f"{side}_shin"] = (0.55 * knee + 0.45 * ankle, "Leg_Foot_proxy", False)
    # Surface proxies for seated/back contact.  These are deliberately built
    # from source points only and carry provenance for diagnostics.
    pelvis = pick("pelvis", "Pelvis", "hips")
    left_hip = pick("left_hip", "LeftUpLeg")
    right_hip = pick("right_hip", "RightUpLeg")
    spine = pick("spine3", "spine2", "spine", "Spine")
    if pelvis is not None:
        if left_hip is not None and right_hip is not None:
            axis = left_hip - right_hip
            axis /= max(float(np.linalg.norm(axis)), 1e-12)
            # Posterior proxy is opposite the forward spine direction when
            # available; lateral offsets keep left/right channels distinct.
            forward = spine - pelvis if spine is not None else np.array([0.0, 0.0, 1.0])
            forward /= max(float(np.linalg.norm(forward)), 1e-12)
            posterior = np.cross(axis, forward)
            posterior /= max(float(np.linalg.norm(posterior)), 1e-12)
            if posterior[2] < 0:
                posterior = -posterior
            points["left_butt"] = (pelvis + 0.06 * axis - 0.025 * posterior, "hip_derived_butt_proxy", False)
            points["right_butt"] = (pelvis - 0.06 * axis - 0.025 * posterior, "hip_derived_butt_proxy", False)
        else:
            points["left_butt"] = (pelvis + np.array([0.0, 0.045, -0.02]), "pelvis_surface_proxy", False)
            points["right_butt"] = (pelvis + np.array([0.0, -0.045, -0.02]), "pelvis_surface_proxy", False)
    if spine is not None:
        points["lower_back"] = (spine + np.array([0.0, 0.0, -0.08]), "spine_surface_proxy", False)
        points["upper_back"] = (spine + np.array([0.0, 0.0, 0.06]), "spine_surface_proxy", False)
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
    previous_normals: dict[str, np.ndarray] = {}
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
                "source_state": state,
                "normal_error": float(abs(hit.signed_distance)),
                "tangent_error": tangential_speed,
            }
            previous[channel] = point_solver.copy()

        # Keep the channel schema stable across frames/configurations.  A
        # missing optional proxy is an inactive channel, never an implicit
        # contact or a silently re-used joint center.
        for channel in config.get("channels", ("left_heel", "right_heel", "left_toe", "right_toe",
                                                "left_palm", "right_palm", "left_knee", "right_knee",
                                                "left_shin", "right_shin", "left_butt", "right_butt",
                                                "lower_back", "upper_back")):
            contacts.setdefault(channel, {
                "score": 0.0, "state": "NONE", "source_state": "NONE",
                "human_point_source": np.zeros(3), "human_point_solver": np.zeros(3),
                "surface_point_source": np.zeros(3), "surface_point_solver": np.zeros(3),
                "surface_normal_source": np.array([0.0, 0.0, 1.0]),
                "surface_normal_solver": np.array([0.0, 0.0, 1.0]),
                "surface_id": "", "surface_type": "", "signed_distance": float("inf"),
                "normal_speed": 0.0, "tangential_speed": 0.0,
                "normal_error": 0.0, "tangent_error": 0.0,
                "human_point_provenance": "missing_optional_proxy",
            })
        flat_foot = {}
        max_angle = np.deg2rad(float(config.get("flat_foot_max_normal_angle_deg", 12.0)))
        min_score = float(config.get("flat_foot_min_score", 0.45))
        for side in ("left", "right"):
            heel, toe = contacts[f"{side}_heel"], contacts[f"{side}_toe"]
            # Preserve the measured foot frame for airborne orientation.  It
            # is derived only from source landmarks and never from robot q.
            foot_forward = np.asarray(toe["human_point_solver"]) - np.asarray(heel["human_point_solver"])
            foot_forward /= max(float(np.linalg.norm(foot_forward)), 1e-12)
            big = source_frame.get(f"{side}_big_toe")
            small = source_frame.get(f"{side}_small_toe")
            if big is not None and small is not None:
                foot_lateral = np.asarray(small, dtype=float) - np.asarray(big, dtype=float)
            else:
                left_hip = source_frame.get("left_hip")
                right_hip = source_frame.get("right_hip")
                foot_lateral = ((np.asarray(left_hip) - np.asarray(right_hip))
                                if left_hip is not None and right_hip is not None
                                else np.array([0.0, 1.0, 0.0]))
            body_left = foot_lateral.copy()
            left_hip = source_frame.get("left_hip")
            right_hip = source_frame.get("right_hip")
            if left_hip is not None and right_hip is not None:
                body_left = np.asarray(left_hip) - np.asarray(right_hip)
                body_left /= max(float(np.linalg.norm(body_left)), 1e-12)
                if float(foot_lateral @ body_left) < 0.0:
                    foot_lateral = -foot_lateral
            foot_lateral -= foot_forward * float(foot_lateral @ foot_forward)
            if np.linalg.norm(foot_lateral) < 1e-8:
                foot_lateral = body_left - foot_forward * float(body_left @ foot_forward)
            foot_lateral /= max(float(np.linalg.norm(foot_lateral)), 1e-12)
            foot_normal = np.cross(foot_forward, foot_lateral)
            foot_normal /= max(float(np.linalg.norm(foot_normal)), 1e-12)
            body_up = np.asarray(source_frame.get("spine3", source_frame.get("pelvis", [0.0, 0.0, 1.0])), dtype=float) - np.asarray(source_frame.get("pelvis", [0.0, 0.0, 0.0]), dtype=float)
            body_up /= max(float(np.linalg.norm(body_up)), 1e-12)
            if float(foot_normal @ body_up) < 0.0:
                foot_normal = -foot_normal
            previous_normal = previous_normals.get(side)
            if previous_normal is not None and float(previous_normal @ foot_normal) < 0.0:
                foot_normal = -foot_normal
            previous_normals[side] = foot_normal.copy()
            for item in (heel, toe):
                item["human_foot_forward_solver"] = foot_forward.copy()
                item["human_foot_normal_solver"] = foot_normal.copy()
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
