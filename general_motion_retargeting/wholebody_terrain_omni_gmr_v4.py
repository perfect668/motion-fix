"""Terrain-aware V4 entry point.

This module is intentionally a small composition layer.  The Omni-first V4
solver, primary tasks, and Mink limits remain in ``wholebody_omni_gmr_v4``;
terrain callers get an explicit class/constructor without changing the flat
V4 baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .terrain_geometry import SceneTransform, TerrainField
from .wholebody_omni_gmr_v4 import WholeBodyOmniGMRV4


class WholeBodyTerrainOmniGMRV4(WholeBodyOmniGMRV4):
    """V4 retargeter with an explicit terrain specification and transform."""

    def __init__(
        self,
        config_path: str | Path,
        terrain_spec: TerrainField | dict[str, Any] | str | Path,
        scene_transform_config: SceneTransform | dict[str, Any],
        terrain_config_path: str | Path | None = None,
        environment_pool=None,
        fps: float = 50.0,
        solver: str = "daqp",
    ) -> None:
        if isinstance(terrain_spec, TerrainField):
            source_terrain = terrain_spec
        elif isinstance(terrain_spec, dict):
            source_terrain = TerrainField.from_spec(terrain_spec)
        else:
            source_terrain = TerrainField.from_file(terrain_spec)
        transform = (
            scene_transform_config
            if isinstance(scene_transform_config, SceneTransform)
            else SceneTransform(**scene_transform_config)
        )
        source_terrain.support_normal_min_z = float(
            json.loads(Path(terrain_config_path).read_text()).get("terrain", {}).get(
                "support_normal_min_z", source_terrain.support_normal_min_z
            ) if terrain_config_path else source_terrain.support_normal_min_z
        )
        transformed_terrain = source_terrain.transform(transform)
        if environment_pool is None:
            import numpy as np
            environment_pool = np.empty((0, 3), dtype=float)
        self.scene_transform = transform
        self.source_terrain = source_terrain
        super().__init__(config_path, transformed_terrain, environment_pool, fps=fps, solver=solver)

