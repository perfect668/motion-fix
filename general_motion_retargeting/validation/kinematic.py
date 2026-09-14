from __future__ import annotations

import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation

from ..core.schemas import RetargetResult
from ..robot_profile import is_dof_joint_type


class FinalValidator:
    """Replay the exported trajectory and validate the final state contract."""

    def __init__(self, solver=None, contact_plan=None):
        self.solver = solver
        self.contact_plan = contact_plan

    def validate(self, result: RetargetResult) -> dict:
        return _validate(result, self.solver, self.contact_plan)


def validate_result(result: RetargetResult, solver=None, contact_plan=None) -> dict:
    return FinalValidator(solver, contact_plan).validate(result)


def _schedule_frames(contact_plan):
    """Return normalized per-frame contact records or ``None``.

    V5 keeps the in-memory ``TargetContactSchedule`` during solving, while
    exported artifacts store the same schedule as ``{"frames": [...]}``.
    Validation is a replay boundary and must accept both forms without
    treating an empty/malformed schedule as evidence that the motion is in
    flight.
    """
    if contact_plan is None:
        return None
    if isinstance(contact_plan, dict):
        frames = contact_plan.get("frames", contact_plan.get("per_frame_states"))
    else:
        frames = getattr(contact_plan, "per_frame_states", contact_plan)
    if frames is None:
        return None
    frames = list(frames)
    if not frames:
        return None
    if not all(isinstance(frame, dict) for frame in frames):
        return None
    return frames


def _validate(result: RetargetResult, solver=None, contact_plan=None) -> dict:
    # The exported result owns the authoritative contact schedule.  Callers
    # such as standalone validators and dataset QA tools often only pass the
    # result and solver; silently treating a missing explicit argument as
    # "no contacts" would make a floating trajectory appear valid.
    if contact_plan is None:
        contact_plan = getattr(result, "contact_plan", None)
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
        body_quat = np.asarray(fk["body_quat_w"], dtype=float)
        quat_norms = np.linalg.norm(body_quat, axis=-1)
        checks["legal_body_quaternions"] = bool(
            np.isfinite(quat_norms).all()
            and np.all(np.abs(quat_norms - 1.0) <= 1e-4)
        )
        expected_joint_names = tuple(getattr(solver.robot_profile, "joint_names", ()))
        exported_joint_names = tuple(fk.get("robot_joint_names", ()))
        expected_joint_count = len(expected_joint_names)
        model_joint_count = sum(
            is_dof_joint_type(solver.model.jnt_type[joint_id])
            for joint_id in range(solver.model.njnt)
        )
        checks["joint_contract"] = bool(
            model_joint_count > 0
            and expected_joint_count == model_joint_count
            and expected_joint_count > 0
            and exported_joint_names == expected_joint_names
            and fk["joint_pos"].shape == (len(q), expected_joint_count)
            and fk["joint_vel"].shape == (len(q), expected_joint_count)
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
        joint_limit_violations = _joint_limit_violations(solver, q)
        checks["joint_velocity_limits"] = _joint_velocities_ok(solver, fk["joint_vel"])
        checks["base_velocity_limits"] = _base_velocities_ok(solver, q)
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
        support_metrics = _replay_support(solver, contact_plan, q)
        checks["support_contact_coverage"] = bool(
            support_metrics["failed_frames"] == 0
            and support_metrics.get("verified_schedule", False)
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
        checks["legal_body_quaternions"] = True
        checks["scene_penetration"] = True
        checks["finite_final_scene_distance"] = True
        checks["terrain_nonpenetration"] = True
        checks["joint_position_limits"] = True
        checks["joint_contract"] = True
        joint_limit_violations = []
        checks["joint_velocity_limits"] = True
        checks["base_velocity_limits"] = True
        checks["contact_residuals"] = True
        checks["self_collision"] = True
        minimum_scene_distance = float("inf")
        minimum_terrain_distance_final = float("inf")
        contact_metrics = {"max_normal_residual": 0.0, "max_tangent_residual": 0.0, "contact_frames": 0}
        support_metrics = {"expected_frames": 0, "covered_frames": 0, "failed_frames": 0, "max_support_gap": 0.0}
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
    schedule_frames = _schedule_frames(contact_plan) or []
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
        "support_metrics": support_metrics,
        "joint_limit_violations": joint_limit_violations,
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
        if not is_dof_joint_type(solver.model.jnt_type[joint_id]):
            continue
        address = int(solver.model.jnt_qposadr[joint_id])
        lower, upper = solver.model.jnt_range[joint_id]
        if np.any(qpos_sequence[:, address] < lower - 1e-7) or np.any(qpos_sequence[:, address] > upper + 1e-7):
            return False
    return True


def _joint_limit_violations(solver, qpos_sequence: np.ndarray) -> list[dict]:
    """Return concrete frame/joint violations instead of only a boolean."""
    qpos_sequence = np.asarray(qpos_sequence, dtype=float)
    violations: list[dict] = []
    for joint_id in range(solver.model.njnt):
        if not solver.model.jnt_limited[joint_id] or not is_dof_joint_type(solver.model.jnt_type[joint_id]):
            continue
        address = int(solver.model.jnt_qposadr[joint_id])
        lower, upper = (float(x) for x in solver.model.jnt_range[joint_id])
        values = qpos_sequence[:, address]
        indices = np.flatnonzero((values < lower - 1e-7) | (values > upper + 1e-7))
        name = mj.mj_id2name(solver.model, mj.mjtObj.mjOBJ_JOINT, joint_id) or str(joint_id)
        for index in indices[:32]:
            violations.append({
                "frame": int(index), "joint": name, "value": float(values[index]),
                "lower": lower, "upper": upper,
            })
    return violations


def _replay_support(solver, contact_plan, qpos_sequence: np.ndarray) -> dict:
    """Check source-indicated support against final robot foot proxies.

    Non-penetration alone only provides a lower bound and therefore cannot
    reject a trajectory that floats above the floor.  The detector marks
    ``support_expected`` from source geometry; this replay then checks that at
    least one configured heel/toe proxy is actually on a supportable surface.
    Flight frames carry no such mark and are not forced onto the ground.
    """
    frames = _schedule_frames(contact_plan)
    schedule_missing = frames is None
    if frames is None:
        frames = []
    gap_limit = float(solver.config.get("validation", {}).get("max_support_gap", 0.005))
    required = ("left_heel", "left_toe", "right_heel", "right_toe")
    expected = covered = failed = unknown = 0
    failed_indices: list[int] = []
    deferred_indices: list[int] = []
    state_counts = {"SUPPORTED": 0, "FLIGHT": 0, "UNKNOWN": 0}
    maximum_gap = 0.0
    activation_floor = float(
        solver.config.get("validation", {}).get(
            "support_activation_min", solver.config.get("validation", {}).get("contact_min_activation", 0.5)
        )
    )
    # A result without a source schedule has no evidence that any frame is a
    # jump/flight phase.  Validate it conservatively against every configured
    # foot proxy instead of treating an empty list as success.  This closes
    # the historical hole where a completely floating motion had zero contact
    # residual and was nevertheless marked VALID.
    if schedule_missing:
        frames = [None] * len(qpos_sequence)
    for index, frame in enumerate(frames):
        if frame is None:
            support_state = "SUPPORTED"
            state_counts["SUPPORTED"] += 1
        else:
            support_state = str(frame.get(
                "support_state", "SUPPORTED" if frame.get("support_expected", False) else "UNKNOWN"
            ))
            state_counts[support_state if support_state in state_counts else "UNKNOWN"] += 1
        # FLIGHT is the only state that is intentionally exempt.  UNKNOWN is
        # not a successful no-contact result: it means the source support
        # evidence is incomplete/inconsistent and must fail closed rather
        # than allowing an unconstrained robot to float.
        if support_state == "FLIGHT":
            continue
        if index >= len(qpos_sequence):
            failed += 1
            failed_indices.append(index)
            continue
        expected += 1
        if support_state == "UNKNOWN":
            unknown += 1
            failed += 1
            failed_indices.append(index)
            continue
        solver.configuration.update(qpos_sequence[index])
        # Evaluate support at the frame level.  Every currently active heel or
        # toe is a claimed support channel, so all of those claims must be
        # close to their scheduled surface.  The previous implementation
        # accumulated candidates across channels and allowed one valid toe to
        # hide a floating heel; retain both min/max values for diagnostics.
        frame_gaps = []
        expected_channels = []
        transition_channels = []
        label_transition_channels = []
        if schedule_missing:
            expected_channels = [
                channel for channel in required if channel in solver.contact.points
            ]
        for channel in (required if not schedule_missing else ()):
            item = frame.get("contacts", {}).get(channel, {})
            if channel not in solver.contact.points:
                continue
            # Coverage validates a source contact which was actually active
            # in this frame.  An episode may be winding down while another
            # foot takes over; accepting an arbitrary nearby robot foot in
            # that case hides missed contact, while requiring inactive
            # heel/toe channels creates false failures at the blend boundary.
            if str(item.get("state", "NONE")) == "NONE":
                continue
            if float(item.get("activation", item.get("score", 0.0))) <= 1e-3:
                continue
            # A large source-surface jump means this channel has just crossed
            # a terrain/object edge (for example a foot transferring from the
            # floor to a stair tread).  At 50 Hz the target height may be
            # physically unreachable in this single frame.  Treat it as a
            # bounded transition channel; if another foot is already carrying
            # support, validate that stable support and check this channel on
            # subsequent frames instead of reporting a false whole-frame gap.
            if bool(item.get("surface_transition", False)):
                transition_channels.append(channel)
                continue
            if "activation" in item and float(item.get("activation", 0.0)) < activation_floor:
                transition_channels.append(channel)
                continue
            expected_channels.append(channel)
        for channel in (required if not schedule_missing else ()):
            item = frame.get("contacts", {}).get(channel, {})
            if (
                channel not in expected_channels
                and channel not in transition_channels
                and item.get("label_contact", False)
                and float(item.get("activation", 0.0)) < activation_floor
            ):
                label_transition_channels.append(channel)
        # Contact activation is intentionally blended over several frames.
        # Those transition frames are not yet a meaningful final-support
        # assertion; defer their coverage check until the task reaches its
        # configured weight.  A legacy/minimal schedule without activation
        # fields remains fail-closed (and is covered by the test contract).
        if not schedule_missing and not expected_channels and (transition_channels or label_transition_channels):
            # A ramp frame still has measurable support evidence.  Replay the
            # physical heel/toe gap instead of skipping it wholesale: this
            # accepts a real foot already on the surface while preserving the
            # fail-closed behavior for a floating activation ramp.  Keep the
            # deferred marker for diagnostics and for the bounded-ramp rule.
            expected_channels = transition_channels or label_transition_channels
            deferred_indices.append(index)
            # A low-confidence landing ramp has an explicit bounded reason
            # for not asserting a 5 mm support gap yet.  Defer only this
            # channel (and only when no stable channel exists); the bounded
            # deferred-run guard below rejects an indefinitely floating
            # sequence. Stable channels on the same frame are checked.
            if bool(frame.get("support_transition", False)) and not label_transition_channels:
                continue
        # The detector's SUPPORTED state is evidence that at least one foot was
        # carrying load. If no canonical heel/toe channel survives, keep this
        # frame visible rather than declaring it covered by an unrelated
        # proxy; diagnostics distinguish it clearly.
        if not expected_channels:
            failed += 1
            failed_indices.append(index)
            continue
        for channel in expected_channels:
            item = {} if schedule_missing else frame.get("contacts", {}).get(channel, {})
            point = solver.contact.points[channel].value(solver.configuration)
            # Contact episodes deliberately lock a surface id/anchor across
            # mesh and stair edges.  Re-querying the nearest support from the
            # robot point here can select the lower tread (or the floor) when
            # the proxy is just outside an edge, reporting a large false gap
            # even though the solver satisfied the scheduled contact.  Replay
            # the same surface frame used by the contact task; only schedules
            # without a surface anchor fall back to a fresh terrain query.
            gap = None
            if not schedule_missing:
                state = str(item.get("state", "NONE"))
                # Tangent anchors represent sticking only; support height is
                # always measured against the actual scene surface.
                surface = item.get("surface_point_solver")
                if surface is None and state == "STATIC":
                    # Legacy/minimal schedules may not carry a source surface
                    # point.  Preserve their locked-anchor replay behavior,
                    # while full V5 schedules always provide the real surface
                    # and therefore cannot validate a floating point against
                    # its own anchor.
                    surface = item.get("tangent_anchor_solver")
                normal = item.get(
                    "anchor_normal_solver" if state == "STATIC" else "surface_normal_solver"
                )
                if surface is not None and normal is not None:
                    surface = np.asarray(surface, dtype=float).reshape(3)
                    normal = np.asarray(normal, dtype=float).reshape(3)
                    normal /= max(float(np.linalg.norm(normal)), 1e-12)
                    if np.isfinite(surface).all() and np.isfinite(normal).all() and normal[2] > 0.6:
                        support_value = getattr(solver.contact, "support_value", None)
                        if support_value is not None:
                            point = support_value(solver.configuration, channel, normal)
                        gap = float(normal @ (point - surface))
            if gap is None:
                hit = solver.terrain.support_surface(point)
                if not hit.supportable:
                    continue
                support_value = getattr(solver.contact, "support_value", None)
                if support_value is not None:
                    point = support_value(solver.configuration, channel, hit.normal)
                gap = float(hit.signed_distance)
                if support_value is not None:
                    gap = float(np.asarray(hit.normal, dtype=float) @ (point - hit.closest_point))
            maximum_gap = max(maximum_gap, max(0.0, gap))
            frame_gaps.append(float(gap))
        if frame_gaps and min(frame_gaps) >= -float(
            solver.config.get("validation", {}).get("max_support_penetration", 0.002)
        ) and max(frame_gaps) <= gap_limit:
            covered += 1
        else:
            failed += 1
            failed_indices.append(index)
    # A contact episode may legitimately begin in its blend ramp, but a long
    # run of deferred frames is not a valid support exemption.  Bound the
    # exemption by the detector's configured blend window so an entire
    # floating prefix cannot be hidden simply because a later frame eventually
    # reaches contact.
    blend_window = int(solver.config.get("terrain_contact", {}).get(
        "contact_blend_frames", 7
    )) if solver is not None else 7
    run = []
    for index in sorted(deferred_indices) + [None]:
        if index is not None and (not run or index == run[-1] + 1):
            run.append(index)
            continue
        if len(run) > blend_window:
            overflow = run[blend_window:]
            # Transition frames that already failed the physical gap check
            # must not be counted twice when the bounded-ramp rule is applied.
            new_failures = [frame_index for frame_index in overflow if frame_index not in failed_indices]
            failed_indices.extend(new_failures)
            failed += len(new_failures)
        run = [] if index is None else [index]
    # A sequence containing only ramp frames has no evidence that the
    # exported robot ever reached support.
    if expected and covered == 0 and deferred_indices:
        new_failures = [frame_index for frame_index in deferred_indices if frame_index not in failed_indices]
        failed += len(new_failures)
        failed_indices.extend(new_failures)
    # A missing schedule is not evidence of flight.  It is only accepted when
    # every frame was explicitly supplied as FLIGHT; otherwise the validator
    # must fail closed and require a measurable support proxy.
    explicit_flight = bool(
        not schedule_missing
        and len(frames) == len(qpos_sequence)
        and len(frames) > 0
        and all(str(frame.get("support_state", "")) == "FLIGHT" for frame in frames)
    )
    verified_schedule = bool(not schedule_missing and len(frames) == len(qpos_sequence))
    return {
        "expected_frames": expected,
        "covered_frames": covered,
        "failed_frames": failed,
        "coverage_ratio": covered / max(expected, 1),
        "max_support_gap": maximum_gap,
        "support_state_frames": state_counts,
        "unknown_frames": unknown,
        "failed_frame_indices": failed_indices,
        "deferred_frames": deferred_indices,
        "schedule_missing": bool(schedule_missing),
        "explicit_flight_only": explicit_flight,
        "verified_schedule": verified_schedule,
    }


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


def _base_velocities_ok(solver, qpos_sequence: np.ndarray) -> bool:
    """Validate free-base finite differences, which Mink's joint limit omits."""
    qpos_sequence = np.asarray(qpos_sequence, dtype=float)
    if len(qpos_sequence) < 2:
        return True
    cfg = solver.config.get("solver", {})
    linear_limit = float(cfg.get("base_linear_velocity_limit", np.inf))
    angular_limit = float(cfg.get("base_angular_velocity_limit", np.inf))
    dt = float(solver.dt)
    linear = np.linalg.norm(np.diff(qpos_sequence[:, :3], axis=0), axis=1) / dt
    if not np.isfinite(linear).all() or np.max(linear, initial=0.0) > linear_limit + 1e-6:
        return False
    if not np.isfinite(angular_limit):
        return True
    angular = []
    for previous, current in zip(qpos_sequence[:-1, 3:7], qpos_sequence[1:, 3:7]):
        relative = Rotation.from_quat(previous, scalar_first=True).inv() * Rotation.from_quat(current, scalar_first=True)
        angular.append(np.linalg.norm(relative.as_rotvec()) / dt)
    return bool(np.isfinite(angular).all() and max(angular, default=0.0) <= angular_limit + 1e-6)


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
    frames = _schedule_frames(contact_plan)
    if frames is None:
        return {"max_normal_residual": 0.0, "max_tangent_residual": 0.0, "contact_frames": 0}
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
            state = str(item.get("state", "NONE"))
            normal_key = "anchor_normal_solver" if state == "STATIC" and "anchor_normal_solver" in item else "surface_normal_solver"
            normal = np.asarray(item.get(normal_key, [0, 0, 1]), dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            support_value = getattr(solver.contact, "support_value", None)
            point = (
                support_value(solver.configuration, channel, normal)
                if support_value is not None and channel in getattr(solver.contact, "support_geom_ids", {})
                else solver.contact.points[channel].value(solver.configuration)
            )
            surface = np.asarray(item.get("surface_point_solver", point), dtype=float)
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
