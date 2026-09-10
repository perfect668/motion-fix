"""Scene ownership resolution for the V5 orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.schemas import FloorPolicy, ResolvedBundle, SceneReference


@dataclass(frozen=True)
class SceneResolution:
    source: SceneReference | None
    target: SceneReference | None
    relation: str
    provenance: dict


class SceneResolver:
    """Convert a resolved motion bundle into explicit scene references."""

    def resolve(self, bundle: ResolvedBundle) -> SceneResolution:
        return SceneResolution(
            source=bundle.source_scene,
            target=bundle.target_scene,
            relation=bundle.scene_relation.value,
            provenance=dict(bundle.metadata),
        )
