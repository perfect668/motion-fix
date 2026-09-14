"""WholeBody V5 terrain-only full-sequence scope admission.

Admission is intentionally separate from file discovery.  A scene asset being
present says nothing about whether the complete motion is foot-supported
terrain locomotion.  This module consumes declarations and maintained dataset
partition evidence before expensive body/scene processing starts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ..core.schemas import InteractionMode, TaskFamily, TerrainKind


SCOPE_VERSION = "wholebody-v5-terrain-only-1"


class AdmissionStatus(str, Enum):
    ADMITTED = "ADMITTED"
    EXCLUDED = "EXCLUDED"
    UNRESOLVED_SCOPE = "UNRESOLVED_SCOPE"
    INPUT_ERROR = "INPUT_ERROR"


@dataclass(frozen=True)
class ScopeDecision:
    status: AdmissionStatus
    task_family: TaskFamily | None
    terrain_kind: TerrainKind | None
    interaction_mode: InteractionMode = InteractionMode.FEET_ONLY
    reason_code: str = ""
    evidence_source: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    scope_version: str = SCOPE_VERSION

    @property
    def admitted(self) -> bool:
        return self.status == AdmissionStatus.ADMITTED

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("status", "task_family", "terrain_kind", "interaction_mode"):
            item = value.get(key)
            if isinstance(item, Enum):
                value[key] = item.value
        return value


class ScopeAdmissionError(RuntimeError):
    def __init__(self, decision: ScopeDecision):
        self.decision = decision
        super().__init__(f"{decision.status.value}: {decision.reason_code}")


# Dataset partition names are maintained annotations, not filename keyword
# guesses.  A new GRAIL split remains unresolved until it is reviewed here or
# explicitly declared by a trusted manifest.
_GRAIL_EXCLUDED_PARTITIONS = {
    "sitting": "complex_object_support",
    "pickup_ground": "hand_object_interaction",
    "climb": "non_foot_support_interaction",
    "lying": "non_foot_body_support",
    "crawling": "non_foot_body_support",
    "kneeling": "non_foot_body_support",
}
_GRAIL_TERRAIN_PARTITIONS = {
    "stair_p1": TerrainKind.STAIRS,
    "stairs": TerrainKind.STAIRS,
    "ramp": TerrainKind.RAMP,
    "slope": TerrainKind.RAMP,
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _declared_scope(data: dict[str, Any] | None) -> tuple[TaskFamily, TerrainKind, InteractionMode] | None:
    data = dict(data or {})
    scope = data.get("scope", data)
    family = scope.get("task_family")
    kind = scope.get("terrain_kind")
    mode = scope.get("interaction_mode", InteractionMode.FEET_ONLY.value)
    if family is None and kind is None:
        return None
    if family is None or kind is None:
        raise ValueError("scope must declare both task_family and terrain_kind")
    return TaskFamily(family), TerrainKind(kind), InteractionMode(mode)


def preflight_scope(
    motion_path: str | Path,
    source_format: str,
    declaration: dict[str, Any] | None = None,
) -> ScopeDecision:
    """Classify the complete input before truncation or expensive processing."""
    path = Path(motion_path).expanduser().resolve()
    try:
        declared = _declared_scope(declaration)
    except (TypeError, ValueError) as error:
        return ScopeDecision(
            AdmissionStatus.INPUT_ERROR, None, None,
            reason_code="invalid_scope_declaration",
            evidence_source="manifest_or_cli",
            evidence={"error": str(error)},
        )
    if declared is not None and declared[2] != InteractionMode.FEET_ONLY:
        return ScopeDecision(
            AdmissionStatus.EXCLUDED, declared[0], declared[1], declared[2],
            "unsupported_interaction_mode", "manifest_or_cli",
        )

    if str(source_format).startswith("grail"):
        parts = set(path.parts)
        excluded = sorted(parts & _GRAIL_EXCLUDED_PARTITIONS.keys())
        if excluded:
            partition = excluded[0]
            return ScopeDecision(
                AdmissionStatus.EXCLUDED, None, None,
                reason_code=_GRAIL_EXCLUDED_PARTITIONS[partition],
                evidence_source="maintained_grail_partition_registry",
                evidence={"partition": partition, "full_sequence": True},
            )
        terrain_matches = sorted(parts & _GRAIL_TERRAIN_PARTITIONS.keys())
        if terrain_matches:
            partition = terrain_matches[0]
            expected = (
                TaskFamily.TERRAIN_LOCOMOTION,
                _GRAIL_TERRAIN_PARTITIONS[partition],
                InteractionMode.FEET_ONLY,
            )
            if declared is not None and declared != expected:
                return ScopeDecision(
                    AdmissionStatus.INPUT_ERROR, declared[0], declared[1], declared[2],
                    "scope_contradicts_dataset_partition",
                    "maintained_grail_partition_registry",
                    {"partition": partition, "expected": [item.value for item in expected]},
                )
            declared = expected
        elif declared is None:
            return ScopeDecision(
                AdmissionStatus.UNRESOLVED_SCOPE, None, None,
                reason_code="grail_sequence_requires_reviewed_scope",
                evidence_source="no_registered_partition_or_manifest_scope",
            )

    if declared is None:
        # Existing non-GRAIL format adapters are the reviewed flat-motion
        # baseline.  Scene-bearing or terrain jobs still require an explicit
        # task declaration at the resolver/orchestrator boundary.
        declared = (
            TaskFamily.FLAT_MOTION,
            TerrainKind.FLAT,
            InteractionMode.FEET_ONLY,
        )
        source = "reviewed_non_grail_flat_baseline"
    else:
        source = "manifest_or_cli"
    family, kind, mode = declared
    if (family == TaskFamily.FLAT_MOTION) != (kind == TerrainKind.FLAT):
        return ScopeDecision(
            AdmissionStatus.INPUT_ERROR, family, kind, mode,
            "inconsistent_task_family_and_terrain_kind", source,
        )
    return ScopeDecision(
        AdmissionStatus.ADMITTED, family, kind, mode,
        "within_terrain_only_scope", source,
        {"input_sha256": file_sha256(path), "full_sequence": True},
    )


def validate_loaded_scope(decision: ScopeDecision, canonical, bundle) -> ScopeDecision:
    """Reject declarations contradicted by the complete loaded scene record."""
    if not decision.admitted:
        return decision
    if bundle.scene_relation.value not in {"motion_only", "shared_scene"}:
        return ScopeDecision(
            AdmissionStatus.EXCLUDED, decision.task_family, decision.terrain_kind,
            reason_code="unsupported_scene_relation",
            evidence_source="resolved_bundle",
            evidence={"scene_relation": bundle.scene_relation.value},
        )
    scene = getattr(canonical, "scene", {}) or {}
    obj = scene.get("obj_data", {}) if isinstance(scene, dict) else {}
    translations = np.asarray(obj.get("obj_t", []), dtype=float)
    rotations = np.asarray(obj.get("obj_R", []), dtype=float)
    moving = bool(
        translations.ndim == 2 and len(translations) > 1
        and np.max(np.linalg.norm(translations - translations[0], axis=1)) > 1e-5
    )
    rotating = bool(
        rotations.ndim == 3 and len(rotations) > 1
        and np.max(np.linalg.norm(rotations - rotations[0], axis=(1, 2))) > 1e-5
    )
    if moving or rotating:
        return ScopeDecision(
            AdmissionStatus.EXCLUDED, decision.task_family, decision.terrain_kind,
            reason_code="dynamic_scene_not_supported",
            evidence_source="complete_scene_trajectory",
            evidence={"translation_changes": moving, "rotation_changes": rotating},
        )
    if decision.task_family == TaskFamily.TERRAIN_LOCOMOTION and bundle.source_scene is None:
        return ScopeDecision(
            AdmissionStatus.INPUT_ERROR, decision.task_family, decision.terrain_kind,
            reason_code="terrain_locomotion_requires_explicit_static_scene",
            evidence_source="resolved_bundle",
        )
    return decision


__all__ = [
    "AdmissionStatus", "ScopeAdmissionError", "ScopeDecision", "SCOPE_VERSION",
    "file_sha256", "preflight_scope", "validate_loaded_scope",
]
