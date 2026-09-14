"""Resolve once, prepare once, then expose scene query/collision geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..core.schemas import FloorPolicy, SceneReference, SceneModel
from .loader import load_scene_model
from .prepared import PreparedSceneGeometry


@dataclass(frozen=True)
class SceneAssembly:
    model: SceneModel
    geometry: PreparedSceneGeometry
    terrain: object


class SceneAssembler:
    def prepare(self, reference: SceneReference | None, *, sample_count: int = 2048) -> SceneAssembly:
        """Decode a scene exactly once, without committing to a floor policy."""
        model = load_scene_model(
            reference, floor_policy=FloorPolicy.DISABLED, floor_height=None
        )
        geometry = PreparedSceneGeometry.from_scene(model, sample_count=sample_count)
        return SceneAssembly(model, geometry, geometry.terrain_field())

    def with_floor(
        self,
        prepared: SceneAssembly,
        *,
        floor_policy: FloorPolicy,
        floor_height,
    ) -> SceneAssembly:
        """Apply final floor metadata without reloading asset vertices."""
        model = replace(
            prepared.model,
            floor_policy=floor_policy,
            floor_height=None if floor_height is None else float(floor_height),
        )
        geometry = prepared.geometry.with_scene(model)
        return SceneAssembly(model, geometry, geometry.terrain_field())

    def assemble(self, reference: SceneReference | None, *, floor_policy: FloorPolicy, floor_height, sample_count: int = 2048) -> SceneAssembly:
        return self.with_floor(
            self.prepare(reference, sample_count=sample_count),
            floor_policy=floor_policy,
            floor_height=floor_height,
        )

    def build(self, reference: SceneReference | None, *, floor_policy: FloorPolicy, floor_height):
        """Compatibility return for old V5 call sites.

        New code should use :meth:`assemble` to preserve the prepared object
        for MuJoCo construction and alignment diagnostics.
        """
        # Keep the compatibility API on the same prepared-geometry path as
        # the V5 orchestrator.  Re-loading through ``terrain_from_scene``
        # would apply asset scale/pose metadata a second time and make query
        # surfaces disagree with visual/CoACD/MuJoCo geometry.
        assembly = self.assemble(
            reference,
            floor_policy=floor_policy,
            floor_height=floor_height,
        )
        return assembly.model, assembly.terrain
