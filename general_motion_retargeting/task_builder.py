"""Task/limit assembly facade for the independent V5 solver."""

from __future__ import annotations


class TaskBuilder:
    """Build a stable task and hard-limit list from canonical frame data.

    The builder intentionally delegates object construction to the solver's
    profile-owned components; it contains no format or scene special cases.
    """

    def __init__(self, solver):
        self.solver = solver

    def build_tasks(self, source, frame, contact_frame):
        solver = self.solver
        solver.interaction.set_target(source)
        solver.bone_direction.set_source(source)
        solver.limb_plane.set_source(source)
        root = frame.get("pelvis") or frame.get("root")
        solver.root.set_target(source.get("pelvis", root[0]), root[1])
        solver.torso.set_source(source, root[1], solver.configuration.data.qpos)
        solver.contact.set_contacts(contact_frame.get("contacts", {}))
        tasks = [
            solver.interaction,
            solver.bone_direction,
            solver.limb_plane,
            solver.contact,
            solver.root,
            solver.torso.task,
            solver.nominal,
        ]
        if solver.previous is not None:
            import mink
            temporal = mink.PostureTask(
                solver.model, solver._temporal_costs, gain=.35, lm_damping=1.0
            )
            temporal.set_target(solver.previous)
            tasks.append(temporal)
        return tasks

    def build_limits(self, include_velocity: bool):
        solver = self.solver
        limits = [solver.config_limit, solver.terrain_limit, solver.scene_collision, solver.trust]
        if include_velocity:
            limits.append(solver.velocity)
        return limits
