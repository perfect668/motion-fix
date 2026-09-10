from __future__ import annotations

import numpy as np
import mujoco as mj

from ..core.schemas import RetargetResult


class FinalValidator:
    """Replay the exported trajectory and validate the final state contract."""

    def __init__(self, solver=None, contact_plan=None):
        self.solver = solver
        self.contact_plan = contact_plan

    def validate(self, result: RetargetResult) -> dict:
        return _validate(result, self.solver, self.contact_plan)


def validate_result(result: RetargetResult, solver=None, contact_plan=None) -> dict:
    return FinalValidator(solver, contact_plan).validate(result)


def _validate(result: RetargetResult, solver=None, contact_plan=None) -> dict:
    q = np.asarray(result.qpos, dtype=float)
    finite = bool(np.isfinite(q).all())
    failures = sum(bool(item.get("qp_failures")) for item in result.diagnostics)
    shape_ok = bool(q.ndim == 2 and (solver is None or q.shape[1] == solver.model.nq))
    diagnostics_complete = bool(len(result.diagnostics) == len(q))
    distances = [float(item.get("minimum_terrain_distance", np.inf)) for item in result.diagnostics]
    checks = {
        "finite_qpos": finite,
        "qp_failures": failures == 0,
        "qpos_shape": shape_ok,
        "diagnostics_complete": diagnostics_complete,
    }
    if solver is not None and len(q):
        fk = solver.forward_kinematics(q)
        derivatives = _joint_derivatives(fk["joint_vel"], float(solver.dt))
        checks["finite_joint_vel"] = bool(np.isfinite(fk["joint_vel"]).all())
        checks["finite_joint_acceleration"] = bool(np.isfinite(derivatives["acceleration"]).all())
        checks["finite_joint_jerk"] = bool(np.isfinite(derivatives["jerk"]).all())
        checks["finite_body_states"] = bool(
            np.isfinite(fk["body_pos_w"]).all() and np.isfinite(fk["body_quat_w"]).all()
        )
        # Final validation replays every exported qpos through MuJoCo rather
        # than trusting the solver's last active set.
        final_scene_distances = []
        final_scene_penetration = 0.0
        terrain_slacks = []
        terrain_distances = []
        for frame_index, qpos in enumerate(q):
            # Dynamic scene poses are part of the final-state contract.  The
            # replay must evaluate each robot frame against the matching
            # asset pose, not against whatever mocap pose the solve loop left
            # in the MuJoCo data buffer.
            if hasattr(solver, "_update_scene_time"):
                solver._update_scene_time(frame_index * solver.dt)
            else:
                solver.configuration.update(qpos)
            solver.configuration.update(qpos)
            solver.scene_collision.prepare_active_set(solver.configuration)
            pair_distances = solver.scene_collision.all_distances(solver.configuration)
            if pair_distances.size:
                final_scene_distances.extend(pair_distances.tolist())
                final_scene_penetration = max(final_scene_penetration, float(max(0.0, -np.min(pair_distances))))
            terrain_measurements = solver.terrain_limit.measure_all(solver.configuration)
            terrain_slacks.extend(item["slack"] for item in terrain_measurements)
            terrain_distances.extend(item["signed_distance"] for item in terrain_measurements)
        checks["scene_penetration"] = final_scene_penetration <= float(
            solver.config.get("validation", {}).get("max_scene_penetration", 0.002)
        )
        checks["finite_final_scene_distance"] = bool(not final_scene_distances or np.isfinite(final_scene_distances).all())
        max_terrain_violation = max(0.0, -min(terrain_slacks, default=0.0))
        checks["terrain_nonpenetration"] = max_terrain_violation <= float(
            solver.config.get("validation", {}).get("max_terrain_violation", 0.002)
        )
        checks["joint_position_limits"] = _joint_limits_ok(solver, q)
        checks["joint_velocity_limits"] = _joint_velocities_ok(solver, fk["joint_vel"])
        solver_limits = solver.config.get("solver", {})
        max_acceleration = float(solver_limits.get("joint_acceleration_limit", np.inf))
        max_jerk = float(solver_limits.get("joint_jerk_limit", np.inf))
        checks["joint_acceleration_limits"] = bool(
            np.max(np.abs(derivatives["acceleration"]), initial=0.0) <= max_acceleration + 1e-6
        )
        checks["joint_jerk_limits"] = bool(
            np.max(np.abs(derivatives["jerk"]), initial=0.0) <= max_jerk + 1e-6
        )
        contact_metrics = _replay_contacts(solver, contact_plan, q)
        checks["contact_residuals"] = contact_metrics["max_normal_residual"] <= float(
            solver.config.get("validation", {}).get("max_contact_normal_residual", 0.02)
        ) and contact_metrics["max_tangent_residual"] <= float(
            solver.config.get("validation", {}).get("max_contact_tangent_residual", 0.08)
        )
        self_collision = _replay_self_collision(solver, q)
        checks["self_collision"] = self_collision["maximum_penetration"] <= float(
            solver.config.get("validation", {}).get("max_self_penetration", 0.002)
        )
        minimum_scene_distance = float(min(final_scene_distances, default=np.inf))
        minimum_terrain_distance_final = float(min(terrain_distances, default=np.inf))
    else:
        checks["finite_joint_vel"] = True
        checks["finite_body_states"] = True
        checks["scene_penetration"] = True
        checks["finite_final_scene_distance"] = True
        checks["terrain_nonpenetration"] = True
        checks["joint_position_limits"] = True
        checks["joint_velocity_limits"] = True
        checks["contact_residuals"] = True
        checks["self_collision"] = True
        minimum_scene_distance = float("inf")
        minimum_terrain_distance_final = float("inf")
        contact_metrics = {"max_normal_residual": 0.0, "max_tangent_residual": 0.0, "contact_frames": 0}
        self_collision = {"maximum_penetration": 0.0, "minimum_distance": float("inf")}
        final_scene_penetration = 0.0
        max_terrain_violation = 0.0
        derivatives = {"acceleration": np.zeros((len(q), 0)), "jerk": np.zeros((len(q), 0))}
    valid = all(checks.values())
    scene_penetrations = np.asarray([max(0.0, float(item.get("maximum_scene_penetration", 0.0))) for item in result.diagnostics])
    active_counts = np.asarray([float(item.get("scene_collision_active_pairs", 0.0)) for item in result.diagnostics])
    collision_query_times = np.asarray([
        float(item.get("collision_query_time", item.get("scene_collision_query_runtime_seconds", np.nan)))
        for item in result.diagnostics
    ], dtype=float)
    qp_times = np.asarray([
        float(item.get("qp_solve_time", item.get("qp_solve_runtime_seconds", np.nan)))
        for item in result.diagnostics
    ], dtype=float)
    contact_summary = {}
    for item in result.diagnostics:
        for channel, contact in item.get("contacts", {}).items():
            state = str(contact.get("robot_state", "NONE"))
            record = contact_summary.setdefault(channel, {"frames": 0, "contact_frames": 0, "static_frames": 0})
            record["frames"] += 1
            record["contact_frames"] += state != "NONE"
            record["static_frames"] += state == "STATIC"
    for record in contact_summary.values():
        frames = max(1, record["frames"])
        record["contact_ratio"] = record["contact_frames"] / frames
        record["static_ratio"] = record["static_frames"] / frames
    # Reuse the format-neutral scene diagnostic reducer so seated-phase
    # butt/back distances, contact ratios and runtime aggregates are present
    # in the authoritative V5 validation payload as well as the standalone
    # debug script.
    from ..scene_diagnostics import summarize_scene_diagnostics
    schedule_frames = getattr(contact_plan, "per_frame_states", contact_plan) if contact_plan is not None else []
    reduced_summary = summarize_scene_diagnostics(result.diagnostics, schedule_frames)
    sequence_summary = {
        "max_penetration": float(np.max(scene_penetrations, initial=0.0)),
        "p99_penetration": float(np.quantile(scene_penetrations, 0.99)) if scene_penetrations.size else 0.0,
        "mean_active_scene_collision_pairs": float(np.mean(active_counts)) if active_counts.size else 0.0,
        "max_active_scene_collision_pairs": int(np.max(active_counts, initial=0.0)),
        "mean_collision_query_runtime_seconds": float(np.nanmean(collision_query_times)) if np.isfinite(collision_query_times).any() else 0.0,
        "max_collision_query_runtime_seconds": float(np.nanmax(collision_query_times)) if np.isfinite(collision_query_times).any() else 0.0,
        "mean_qp_solve_runtime_seconds": float(np.nanmean(qp_times)) if np.isfinite(qp_times).any() else 0.0,
        "max_qp_solve_runtime_seconds": float(np.nanmax(qp_times)) if np.isfinite(qp_times).any() else 0.0,
        "contact_channels": contact_summary,
        "qp_failure_count": int(failures),
    }
    # The reducer's seated/object distance and surface-switch fields are
    # additive; existing V5 keys above remain authoritative when duplicated.
    sequence_summary.update({key: value for key, value in reduced_summary.items() if key not in sequence_summary or key.startswith(("median_", "max_", "seated_", "left_butt", "right_butt", "lower_back", "upper_back"))})
    return {
        "status": "VALID" if valid else "INVALID",
        "checks": checks,
        "finite_qpos": finite,
        "qp_failure_count": int(failures),
        "minimum_terrain_distance": float(min(min(distances, default=np.inf), minimum_terrain_distance_final)),
        "minimum_scene_distance": minimum_scene_distance,
        "maximum_scene_penetration": float(final_scene_penetration),
            "maximum_terrain_violation": float(max_terrain_violation if solver is not None and len(q) else 0.0),
        "contact_metrics": contact_metrics,
        "self_collision": self_collision,
        "motion_quality": {
            "max_joint_velocity": float(np.max(np.abs(fk["joint_vel"]), initial=0.0)) if solver is not None and len(q) else 0.0,
            "max_joint_acceleration": float(np.max(np.abs(derivatives["acceleration"]), initial=0.0)),
            "max_joint_jerk": float(np.max(np.abs(derivatives["jerk"]), initial=0.0)),
        },
        "sequence_summary": sequence_summary,
        "frame_count": int(len(q)),
    }


def _joint_limits_ok(solver, qpos_sequence: np.ndarray) -> bool:
    for joint_id in range(solver.model.njnt):
        if not solver.model.jnt_limited[joint_id]:
            continue
        if solver.model.jnt_type[joint_id] not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            continue
        address = int(solver.model.jnt_qposadr[joint_id])
        lower, upper = solver.model.jnt_range[joint_id]
        if np.any(qpos_sequence[:, address] < lower - 1e-7) or np.any(qpos_sequence[:, address] > upper + 1e-7):
            return False
    return True


def _joint_derivatives(velocity: np.ndarray, dt: float) -> dict[str, np.ndarray]:
    """Differentiate final exported joint velocity with finite, fixed timing."""
    velocity = np.asarray(velocity, dtype=float)
    if velocity.ndim != 2:
        raise ValueError(f"joint velocity must have shape (T,D), got {velocity.shape}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"derivative dt must be positive, got {dt}")
    acceleration = np.zeros_like(velocity)
    if len(velocity) > 1:
        acceleration[1:] = np.diff(velocity, axis=0) / dt
        acceleration[0] = acceleration[1]
    jerk = np.zeros_like(velocity)
    if len(velocity) > 1:
        jerk[1:] = np.diff(acceleration, axis=0) / dt
        jerk[0] = jerk[1]
    return {"acceleration": acceleration, "jerk": jerk}


def _joint_velocities_ok(solver, velocities: np.ndarray) -> bool:
    limit = float(solver.config.get("solver", {}).get("joint_velocity_limit", np.inf))
    return bool(np.isfinite(velocities).all() and np.max(np.abs(velocities), initial=0.0) <= limit + 1e-6)


def _replay_contacts(solver, contact_plan, qpos_sequence: np.ndarray) -> dict:
    maximum_normal = 0.0
    maximum_tangent = 0.0
    contact_frames = 0
    reliable_contact_frames = 0
    minimum_activation = float(solver.config.get("validation", {}).get("contact_min_activation", 0.5)) if solver is not None else 0.5
    raw_maximum_normal = 0.0
    raw_maximum_tangent = 0.0
    if contact_plan is None:
        return {"max_normal_residual": 0.0, "max_tangent_residual": 0.0, "contact_frames": 0}
    frames = getattr(contact_plan, "per_frame_states", contact_plan)
    for frame_index, frame in enumerate(frames):
        if frame_index >= len(qpos_sequence):
            break
        solver.configuration.update(qpos_sequence[frame_index])
        for channel, item in frame.get("contacts", {}).items():
            if item.get("state", "NONE") == "NONE" or channel not in solver.contact.points:
                continue
            # Contact weights are blended over several frames.  Transitional
            # evidence with negligible activation must not fail validation as
            # if it were a fully enforced anchor.
            activation = float(item.get("activation", item.get("score", 0.0)))
            if activation <= 1e-3:
                continue
            point = solver.contact.points[channel].value(solver.configuration)
            state = str(item.get("state", "NONE"))
            normal_key = "anchor_normal_solver" if state == "STATIC" and "anchor_normal_solver" in item else "surface_normal_solver"
            normal = np.asarray(item.get(normal_key, [0, 0, 1]), dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            surface_key = "tangent_anchor_solver" if state == "STATIC" and "tangent_anchor_solver" in item else "surface_point_solver"
            surface = np.asarray(item.get(surface_key, point), dtype=float)
            normal_residual = abs(float(normal @ (point - surface) - solver.contact.clearance))
            tangent = solver.contact._tangent_basis(normal)
            anchor = np.asarray(item.get("tangent_anchor_solver", surface), dtype=float)
            tangent_residual = (
                float(np.linalg.norm(tangent @ (point - anchor)))
                if state == "STATIC" else 0.0
            )
            raw_maximum_normal = max(raw_maximum_normal, normal_residual)
            raw_maximum_tangent = max(raw_maximum_tangent, tangent_residual)
            if activation < minimum_activation:
                continue
            maximum_normal = max(maximum_normal, normal_residual)
            maximum_tangent = max(maximum_tangent, tangent_residual)
            reliable_contact_frames += 1
            contact_frames += 1
    return {
        "max_normal_residual": maximum_normal,
        "max_tangent_residual": maximum_tangent,
        "raw_max_normal_residual": raw_maximum_normal,
        "raw_max_tangent_residual": raw_maximum_tangent,
        "contact_frames": contact_frames,
        "reliable_contact_frames": reliable_contact_frames,
        "contact_min_activation": minimum_activation,
    }


def _replay_self_collision(solver, qpos_sequence: np.ndarray) -> dict:
    geometries = list(getattr(solver.scene_collision, "robot_geoms", ()))
    minimum = np.inf
    for qpos in qpos_sequence:
        solver.configuration.update(qpos)
        for first_index, first in enumerate(geometries):
            first_body = int(solver.model.geom_bodyid[first])
            for second in geometries[first_index + 1:]:
                second_body = int(solver.model.geom_bodyid[second])
                if first_body == second_body or _ancestor_related(solver.model, first_body, second_body):
                    continue
                center_distance = float(np.linalg.norm(
                    solver.configuration.data.geom_xpos[first] - solver.configuration.data.geom_xpos[second]
                ))
                broadphase_limit = min(
                    float(solver.model.geom_rbound[first] + solver.model.geom_rbound[second] + 0.01),
                    float(solver.config.get("validation", {}).get("self_collision_broadphase", 0.12)),
                )
                if center_distance > broadphase_limit:
                    continue
                distance = float(mj.mj_geomDistance(solver.model, solver.configuration.data, first, second, 1e6, np.zeros(6)))
                minimum = min(minimum, distance)
    return {"minimum_distance": float(minimum), "maximum_penetration": float(max(0.0, -minimum))}


def _ancestor_related(model, first_body: int, second_body: int) -> bool:
    """Adjacent/ancestor link collision meshes intentionally overlap."""
    ancestors = set()
    current = first_body
    # MuJoCo's world body has parent id 0 (itself).  Stop at world instead of
    # walking 0 -> 0 forever; malformed/custom models are also protected by
    # the visited set.
    visited = set()
    while current > 0 and current not in visited:
        ancestors.add(current)
        visited.add(current)
        current = int(model.body_parentid[current])
    current = second_body
    visited.clear()
    while current > 0 and current not in visited:
        if current in ancestors:
            return True
        visited.add(current)
        current = int(model.body_parentid[current])
    return False
