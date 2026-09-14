"""Stage boundaries for the WholeBody V5 solve/export pipeline.

The command-line entry point is intentionally format-aware, but it should not
know how a solved qpos becomes validation evidence.  This module owns the
single solve -> FK -> replay-validation transition and returns a typed bundle
for the exporter.  It contains no dataset or robot-specific branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .core.schemas import RetargetResult
from .validation import validate_result


@dataclass(frozen=True)
class V5SolveArtifacts:
    result: RetargetResult
    runtime_schedule: Any
    kinematics: dict[str, Any]
    validation: dict[str, Any]


def solve_and_validate(
    solver,
    canonical,
    solver_frames: Sequence[dict[str, Any]],
    realized_schedule,
    robot_reference_frames: Sequence[dict[str, Any]] | None = None,
    *,
    source_scene_ok: bool = True,
    scene_alignment_ok: bool = True,
) -> V5SolveArtifacts:
    """Run V5 once and validate the final exported qpos through MuJoCo FK.

    The solver's realized schedule is deliberately used for replay: static
    tangent anchors can be bound to robot proxies during solve, while the
    source schedule remains available as provenance in the exported payload.
    """
    result = solver.solve(
        canonical,
        list(solver_frames),
        list(realized_schedule.per_frame_states),
        robot_reference_frames,
    )
    runtime_schedule = result.contact_plan or realized_schedule
    kinematics = solver.forward_kinematics(result.qpos)
    validation = validate_result(
        result,
        solver=solver,
        contact_plan=runtime_schedule,
    )
    validation["checks"]["source_scene_alignment"] = bool(source_scene_ok)
    validation["checks"]["scene_alignment"] = bool(scene_alignment_ok)
    if not all(validation["checks"].values()):
        validation["status"] = "INVALID"
    result.status = validation["status"]
    return V5SolveArtifacts(result, runtime_schedule, kinematics, validation)


__all__ = ["V5SolveArtifacts", "solve_and_validate"]
