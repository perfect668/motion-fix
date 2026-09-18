"""Terrain-aware contact inference using only source human motion and terrain."""

from __future__ import annotations

from typing import Any

import numpy as np

from .terrain_geometry import SceneTransform, TerrainField, TerrainSurfaceHit


def _closest_point_triangle(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Closest point on a triangle, used by the generic mesh provider."""
    a, b, c = np.asarray(triangle, dtype=float)
    ab, ac, ap = b - a, c - a, np.asarray(point, dtype=float) - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy()
    bp = np.asarray(point, dtype=float) - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return b.copy()
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + (d1 / max(d1 - d3, 1e-12)) * ab
    cp = np.asarray(point, dtype=float) - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return c.copy()
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + (d2 / max(d2 - d6, 1e-12)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denominator = max((d4 - d3) + (d5 - d6), 1e-12)
        return b + ((d4 - d3) / denominator) * (c - b)
    denominator = max(va + vb + vc, 1e-12)
    return a + ab * (vb / denominator) + ac * (vc / denominator)


def _mesh_contact_hit(point: np.ndarray, triangles: np.ndarray, centers: np.ndarray,
                      normals: np.ndarray, support: bool, support_normal_min_z: float):
    """Deterministic nearest/supporting triangle query for any body channel."""
    point = np.asarray(point, dtype=float)
    candidates = np.flatnonzero(normals[:, 2] > support_normal_min_z) if support else np.arange(len(triangles))
    if not len(candidates):
        return None
    if support:
        # Prefer projected triangles containing the point, then the highest
        # valid surface. This rejects a nearby vertical riser at stair edges.
        containing = []
        px, py = point[:2]
        for index in candidates:
            a, b, c = triangles[int(index), :, :2]
            v0, v1, v2 = c - a, b - a, np.array([px, py]) - a
            denominator = float(v0[0] * v1[1] - v1[0] * v0[1])
            if abs(denominator) < 1e-12:
                continue
            u = float((v2[0] * v1[1] - v1[0] * v2[1]) / denominator)
            v = float((v0[0] * v2[1] - v2[0] * v0[1]) / denominator)
            if u >= -1e-8 and v >= -1e-8 and u + v <= 1.0 + 1e-8:
                containing.append(int(index))
        candidates = np.asarray(containing if containing else candidates, dtype=int)
        if containing:
            candidates = np.asarray(sorted(candidates.tolist(), key=lambda i: (-float(centers[i, 2]), i)))
    else:
        # A bounded nearest-centroid shortlist keeps mesh queries predictable.
        order = np.argsort(np.sum((centers[candidates] - point[None, :]) ** 2, axis=1), kind="stable")
        candidates = candidates[order[: min(64, len(order))]]
    best = None
    for index in candidates:
        closest = _closest_point_triangle(point, triangles[int(index)])
        distance = float(np.linalg.norm(point - closest))
        if best is None or (distance, int(index)) < (best[0], best[1]):
            best = (distance, int(index), closest)
    _, index, closest = best
    normal = normals[index].copy()
    if support and normal[2] < 0.0:
        normal = -normal
    elif not support and float(normal @ (point - closest)) < 0.0:
        normal = -normal
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    return int(index), closest, normal, float(normal @ (point - closest))


def _triangle_barycentric(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    a, b, c = np.asarray(triangle, dtype=float)
    v0, v1, v2 = b - a, c - a, np.asarray(point, dtype=float) - a
    d00, d01, d11 = float(v0 @ v0), float(v0 @ v1), float(v1 @ v1)
    d20, d21 = float(v2 @ v0), float(v2 @ v1)
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    return np.array([1.0 - v - w, v, w], dtype=float)


def augment_mesh_contact_schedule(
    schedule: list[dict[str, Any]],
    source_frames: list[dict[str, np.ndarray]],
    vertices: np.ndarray,
    faces: np.ndarray,
    object_id: str,
    object_pose: np.ndarray,
    config: dict,
) -> list[dict[str, Any]]:
    """Augment generic source contacts with a static scene mesh query.

    This provider is deliberately body/asset agnostic: it handles heel/toe
    support queries and nearest-surface queries for every configured body
    channel, while preserving the source schedule when the mesh is too far
    away. The same transformed mesh is used for visual, interaction and
    collision metadata by the caller.
    """
    vertices = np.asarray(vertices, dtype=float).reshape((-1, 3))
    faces = np.asarray(faces, dtype=int).reshape((-1, 3))
    pose = np.asarray(object_pose, dtype=float).reshape(4, 4)
    transformed = (np.c_[vertices, np.ones(len(vertices))] @ pose.T)[:, :3]
    triangles = transformed[faces]
    centers = triangles.mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    threshold = float(config.get("object_contact_distance", 0.05))
    static_speed = float(config.get("static_tangent_speed", 0.08))
    support_min = float(config.get("support_normal_min_z", 0.6))
    switch_hysteresis = float(config.get("surface_switch_hysteresis", 0.015))
    channels = tuple(config.get("channels", ()))
    locked_triangles: dict[str, int] = {}
    for source_frame, frame_record in zip(source_frames, schedule):
        for channel in channels:
            item = frame_record.get("contacts", {}).get(channel)
            if not item:
                continue
            point = np.asarray(item.get("human_point_solver", [np.nan] * 3), dtype=float)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                continue
            result = _mesh_contact_hit(point, triangles, centers, normals,
                                       channel.endswith(("heel", "toe")), support_min)
            if result is None:
                continue
            index, surface_point, normal, signed_distance = result
            previous_index = locked_triangles.get(channel)
            if previous_index is not None and previous_index != index:
                previous_point = _closest_point_triangle(point, triangles[previous_index])
                previous_normal = normals[previous_index].copy()
                if channel.endswith(("heel", "toe")) and previous_normal[2] < 0.0:
                    previous_normal = -previous_normal
                elif not channel.endswith(("heel", "toe")) and float(previous_normal @ (point - previous_point)) < 0.0:
                    previous_normal = -previous_normal
                previous_normal /= max(float(np.linalg.norm(previous_normal)), 1e-12)
                previous_distance = float(previous_normal @ (point - previous_point))
                # Keep an established surface while the alternative is only
                # marginally closer. This prevents stair/chair edge flicker.
                if abs(previous_distance) <= abs(signed_distance) + switch_hysteresis:
                    index, surface_point, normal, signed_distance = (
                        previous_index, previous_point, previous_normal, previous_distance
                    )
            if abs(signed_distance) > threshold:
                locked_triangles.pop(channel, None)
                continue
            locked_triangles[channel] = int(index)
            score = float(np.clip(1.0 - abs(signed_distance) / max(threshold, 1e-9), 0.0, 1.0))
            tangent_speed = float(item.get("tangential_speed", 0.0))
            state = "STATIC" if tangent_speed < static_speed else "SLIDING"
            item.update({
                "score": max(float(item.get("score", 0.0)), score),
                "state": state, "source_state": state, "object_id": str(object_id),
                "surface_id": f"{object_id}:face_{index:05d}",
                "surface_triangle_index": int(index),
                "surface_barycentric": _triangle_barycentric(
                    surface_point, triangles[index]
                ).tolist(),
                "surface_type": "mesh", "surface_point_solver": surface_point.copy(),
                "surface_normal_solver": normal.copy(), "signed_distance": signed_distance,
                "normal_error": abs(signed_distance), "tangent_error": tangent_speed,
            })
    return schedule


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
        butt_drop = float(config.get("butt_surface_vertical_offset", 0.10))
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
            surface_offset = np.array([0.0, 0.0, -butt_drop])
            points["left_butt"] = (pelvis + 0.06 * axis - 0.025 * posterior + surface_offset, "hip_derived_butt_proxy", False)
            points["right_butt"] = (pelvis - 0.06 * axis - 0.025 * posterior + surface_offset, "hip_derived_butt_proxy", False)
        else:
            points["left_butt"] = (pelvis + np.array([0.0, 0.045, -butt_drop]), "pelvis_surface_proxy", False)
            points["right_butt"] = (pelvis + np.array([0.0, -0.045, -butt_drop]), "pelvis_surface_proxy", False)
    if spine is not None:
        # Derive the posterior direction from the anatomical landmark frame;
        # a world-axis offset is wrong when the subject turns or leans.
        if pelvis is not None and left_hip is not None and right_hip is not None:
            lateral = left_hip - right_hip
            up = spine - pelvis
            lateral /= max(float(np.linalg.norm(lateral)), 1e-12)
            up /= max(float(np.linalg.norm(up)), 1e-12)
            backward = -np.cross(lateral, up)
            backward /= max(float(np.linalg.norm(backward)), 1e-12)
        else:
            backward = np.array([0.0, -1.0, 0.0])
        back_offset = float(config.get("back_surface_offset", 0.10))
        lower_drop = float(config.get("lower_back_vertical_offset", 0.08))
        upper_rise = float(config.get("upper_back_vertical_offset", 0.06))
        points["lower_back"] = (
            spine + back_offset * backward - lower_drop * np.asarray(
                (spine - pelvis) / max(float(np.linalg.norm(spine - pelvis)), 1e-12)
                if pelvis is not None else np.array([0.0, 0.0, 1.0])
            ),
            "spine_surface_proxy", False,
        )
        points["upper_back"] = (
            spine + back_offset * backward + upper_rise * np.asarray(
                (spine - pelvis) / max(float(np.linalg.norm(spine - pelvis)), 1e-12)
                if pelvis is not None else np.array([0.0, 0.0, 1.0])
            ),
            "spine_surface_proxy", False,
        )
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
    point_transform: SceneTransform | None = None,
) -> list[dict[str, Any]]:
    del joint_mapping  # Validation and semantic construction happen before this stage.
    terrain = source_terrain.transform(scene_transform)
    point_transform = scene_transform if point_transform is None else point_transform
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
    previous_frames: dict[str, dict[str, np.ndarray]] = {}
    schedule = []
    for source_frame in source_frames:
        contacts = {}
        for channel, (point_source, provenance, support) in _proxy_points(source_frame, config).items():
            point_solver = point_transform.transform_points(point_source)
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
            foot_forward = np.asarray(toe["human_point_solver"], dtype=float) - np.asarray(heel["human_point_solver"], dtype=float)
            forward_norm = float(np.linalg.norm(foot_forward))
            previous_frame = previous_frames.get(side)
            if forward_norm < 1e-8:
                if previous_frame is None:
                    raise ValueError(f"Cannot construct {side} foot frame: heel and toe coincide")
                foot_forward = previous_frame["forward"].copy()
            else:
                foot_forward /= forward_norm
            big = source_frame.get(f"{side}_big_toe")
            small = source_frame.get(f"{side}_small_toe")
            if big is not None and small is not None:
                foot_lateral = scene_transform.transform_normals(
                    np.asarray(small, dtype=float) - np.asarray(big, dtype=float)
                )
            else:
                left_hip = source_frame.get("left_hip")
                right_hip = source_frame.get("right_hip")
                foot_lateral = scene_transform.transform_normals((np.asarray(left_hip) - np.asarray(right_hip))
                                if left_hip is not None and right_hip is not None
                                else np.array([0.0, 1.0, 0.0]))
            body_left = foot_lateral.copy()
            left_hip = source_frame.get("left_hip")
            right_hip = source_frame.get("right_hip")
            if left_hip is not None and right_hip is not None:
                body_left = scene_transform.transform_normals(np.asarray(left_hip) - np.asarray(right_hip))
                body_left /= max(float(np.linalg.norm(body_left)), 1e-12)
                if float(foot_lateral @ body_left) < 0.0:
                    foot_lateral = -foot_lateral
            foot_lateral -= foot_forward * float(foot_lateral @ foot_forward)
            if np.linalg.norm(foot_lateral) < 1e-8:
                foot_lateral = body_left - foot_forward * float(body_left @ foot_forward)
            if np.linalg.norm(foot_lateral) < 1e-8:
                if previous_frame is None:
                    raise ValueError(f"Cannot construct {side} foot frame: lateral axis is degenerate")
                foot_lateral = previous_frame["lateral"].copy()
            else:
                foot_lateral /= float(np.linalg.norm(foot_lateral))
            body_up = scene_transform.transform_normals(
                np.asarray(source_frame.get("spine3", source_frame.get("pelvis", [0.0, 0.0, 1.0])), dtype=float)
                - np.asarray(source_frame.get("pelvis", [0.0, 0.0, 0.0]), dtype=float)
            )
            if np.linalg.norm(body_up) < 1e-8:
                body_up = previous_frame["up"].copy() if previous_frame is not None else np.array([0.0, 0.0, 1.0])
            else:
                body_up /= float(np.linalg.norm(body_up))

            # A previous lateral axis can itself become parallel to the new
            # forward axis after a sharp turn.  Check the cross product too;
            # otherwise the normalized zero vector silently poisons contact
            # normals and every downstream task.
            foot_normal = np.cross(foot_forward, foot_lateral)
            normal_norm = float(np.linalg.norm(foot_normal))
            if normal_norm < 1e-8:
                fallback_normal = None
                for candidate in (
                    previous_frame["normal"] if previous_frame is not None else None,
                    body_up,
                ):
                    if candidate is None:
                        continue
                    projected = np.asarray(candidate, dtype=float) - foot_forward * float(
                        np.asarray(candidate, dtype=float) @ foot_forward
                    )
                    projected_norm = float(np.linalg.norm(projected))
                    if projected_norm >= 1e-8:
                        fallback_normal = projected / projected_norm
                        break
                if fallback_normal is None:
                    raise ValueError(f"Cannot construct {side} foot frame: normal is degenerate")
                foot_normal = fallback_normal
                # Rebuild the lateral axis from the recovered normal so all
                # three axes are mutually orthogonal in solver coordinates.
                foot_lateral = np.cross(foot_normal, foot_forward)
                lateral_norm = float(np.linalg.norm(foot_lateral))
                if lateral_norm < 1e-8:
                    raise ValueError(f"Cannot construct {side} foot frame: recovered lateral axis is degenerate")
                foot_lateral /= lateral_norm
            else:
                foot_normal /= normal_norm
            if float(foot_normal @ body_up) < 0.0:
                foot_normal = -foot_normal
            if previous_frame is not None and float(previous_frame["normal"] @ foot_normal) < 0.0:
                foot_normal = -foot_normal
            previous_frames[side] = {"forward": foot_forward.copy(), "lateral": foot_lateral.copy(), "normal": foot_normal.copy(), "up": body_up.copy()}
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
