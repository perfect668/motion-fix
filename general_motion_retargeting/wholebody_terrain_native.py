"""Terrain-native WholeBody retargeter built on the V4 engineering shell.

The core difference from V4 is structural: source human/terrain relations are
preplanned over the whole sequence, interaction-mesh deformation remains the
main motion objective, and a four-point sole patch is constrained to the chosen
support surface. Scene collision is a feasibility guard, not a contact planner.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .terrain_native_geometry import TerrainPatchMap
from .terrain_native_tasks import (
    NullFootTask,
    SolePatchConstraintLimit,
    TerrainNativeContactTask,
    TerrainNativeSceneLimit,
    WeightedInteractionLaplacianTask,
    make_ne01_soles,
)
from .wholebody_omni_gmr_v4 import WholeBodyOmniGMRV4


class TerrainNativeRetargeter(WholeBodyOmniGMRV4):
    """HoloSoMo/Omni-style terrain interaction with explicit sole support patches."""

    def __init__(
        self,
        config_path: str | Path,
        terrain,
        environment_pool: np.ndarray,
        patch_map: TerrainPatchMap,
        fps: float = 50.0,
        solver: str = "daqp",
    ) -> None:
        self.patch_map = patch_map
        super().__init__(config_path, terrain, environment_pool, fps=fps, solver=solver)
        native = self.config.get("terrain_native", {})
        if not bool(native.get("enabled", True)):
            raise ValueError("TerrainNativeRetargeter requires terrain_native.enabled=true")

        interaction_cfg = self.config.get("interaction_graph", {})
        native_interaction = native.get("interaction", {})
        old_interaction = self.interaction_task
        self.interaction_task = WeightedInteractionLaplacianTask(
            self.model,
            old_interaction.robot_points,
            environment_pool,
            environment_points=int(native_interaction.get(
                "environment_points",
                getattr(old_interaction, "environment_count", 64),
            )),
            semantic_cost=float(native_interaction.get(
                "semantic_cost", interaction_cfg.get("semantic_cost", 10.0)
            )),
            environment_cost=float(native_interaction.get(
                "environment_cost", interaction_cfg.get("environment_cost", 1.0)
            )),
            gain=float(native_interaction.get("gain", interaction_cfg.get("gain", 0.55))),
            distance_decay=float(native_interaction.get("distance_decay", 3.0)),
            points_per_semantic=int(native_interaction.get("points_per_semantic", 5)),
        )

        soles = make_ne01_soles(self.model)
        sole_cfg = native.get("sole_patch", {})
        self.terrain_native_sole_limit = SolePatchConstraintLimit(
            self.model, soles, sole_cfg
        )
        contact_cfg = self.config.get("contact_tasks", {})
        contact_native_cfg = {
            "legacy_normal_cost": contact_cfg.get("normal_cost", 35.0),
            "legacy_tangent_cost": contact_cfg.get("tangent_cost", 18.0),
            "clearance": contact_cfg.get("clearance", 0.004),
            **native.get("contact_task", {}),
        }
        self.contact_task = TerrainNativeContactTask(
            self.model,
            contact_cfg.get("robot_points", {}),
            soles,
            self.terrain_native_sole_limit,
            contact_native_cfg,
        )
        self.foot_temporal_task = None
        self.foot_orientation_task = NullFootTask(self.model)

        self.scene_collision = TerrainNativeSceneLimit(
            self.scene_collision,
            self.terrain_native_sole_limit,
        )
        self.scene_backend = "terrain_native+" + str(self.scene_backend)

    def retarget(self, *args, **kwargs):
        output = super().retarget(*args, **kwargs)
        diag = self.diagnostics[-1]
        diag["terrain_native"] = self.contact_task.diagnostics(self.configuration)
        diag["terrain_native_patch_count"] = int(len(self.patch_map.patches))
        diag["terrain_native_interaction_environment_points"] = int(
            len(self.interaction_task.environment)
        )
        return output
