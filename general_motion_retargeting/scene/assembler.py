"""Scene model and query-field assembly for WholeBody V5."""

from __future__ import annotations

from ..core.schemas import FloorPolicy, SceneReference
from .loader import load_scene_model, terrain_from_scene


class SceneAssembler:
    def build(self, reference: SceneReference | None, *, floor_policy: FloorPolicy, floor_height):
        model = load_scene_model(reference, floor_policy=floor_policy, floor_height=floor_height)
        return model, terrain_from_scene(model)
