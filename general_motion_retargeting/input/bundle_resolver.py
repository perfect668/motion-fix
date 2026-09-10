"""Deterministic task-level bundle discovery.

Resolution happens before motion adapters.  A declared but missing scene is a
hard error; only an entirely undeclared scene receives the default floor.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from ..core.schemas import MotionReference, ResolvedBundle, SceneReference, SceneRelation
from ..motion_adapters import detect_motion_format


def _pickle_record(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as stream:
            value = pickle.load(stream)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


class BundleResolver:
    """Resolve explicit manifest/CLI references with deterministic fallbacks."""

    def resolve(
        self,
        motion: str | Path,
        *,
        manifest: str | Path | None = None,
        source_scene: str | Path | None = None,
        target_scene: str | Path | None = None,
        motion_format: str = "auto",
    ) -> ResolvedBundle:
        motion_path = Path(motion).expanduser().resolve()
        if not motion_path.is_file():
            raise FileNotFoundError(motion_path)
        data: dict[str, Any] = {}
        if manifest is not None:
            manifest_path = Path(manifest).expanduser().resolve()
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        motion_spec = data.get("motion", {}) if isinstance(data, dict) else {}
        if motion_spec.get("path"):
            motion_path = (Path(manifest).parent / motion_spec["path"]).resolve()
        explicit_format = motion_spec.get("format")
        fmt = str(explicit_format if explicit_format is not None else motion_format or "auto")
        if fmt == "auto":
            fmt = detect_motion_format(motion_path)

        source_spec = data.get("source_scene") if isinstance(data, dict) else None
        target_spec = data.get("target_scene") if isinstance(data, dict) else None
        source_path = Path(source_scene).expanduser().resolve() if source_scene else None
        target_path = Path(target_scene).expanduser().resolve() if target_scene else None
        if source_path is None and isinstance(source_spec, dict) and source_spec.get("path"):
            source_path = (Path(manifest).parent / source_spec["path"]).resolve()
        if target_path is None and isinstance(target_spec, dict) and target_spec.get("path"):
            target_path = (Path(manifest).parent / target_spec["path"]).resolve()

        metadata: dict[str, Any] = {"format": fmt}
        # GRAIL reconstructions carry authoritative object metadata inside
        # the pickle.  Manifest use must not disable this discovery: a job
        # manifest can override the scene explicitly, but when it omits one
        # we still resolve the exact dataset asset before considering a floor.
        if source_path is None and motion_path.suffix.lower() == ".pkl":
            record = _pickle_record(motion_path)
            if record and record.get("human_data"):
                source_path = self._resolve_grail_asset(motion_path, record)
                if source_path is not None:
                    metadata["source_scene_from_metadata"] = True
        if source_path is None:
            sidecar = motion_path.with_suffix(".scene.json")
            if sidecar.is_file():
                side = json.loads(sidecar.read_text(encoding="utf-8"))
                declared = side.get("source_scene", side.get("scene", {}))
                if isinstance(declared, dict) and declared.get("path"):
                    source_path = (sidecar.parent / declared["path"]).resolve()
                    metadata["source_scene_from_sidecar"] = str(sidecar)
        if source_path is not None and not source_path.is_file():
            raise FileNotFoundError(f"Declared source scene does not exist: {source_path}")
        if target_path is not None and not target_path.is_file():
            raise FileNotFoundError(f"Declared target scene does not exist: {target_path}")

        requested_relation = data.get("scene_relation") if isinstance(data, dict) else None
        if requested_relation is None and isinstance(data, dict) and isinstance(data.get("scene"), dict):
            requested_relation = data["scene"].get("relation")
        if requested_relation is not None:
            try:
                relation = SceneRelation(str(requested_relation))
            except ValueError as error:
                raise ValueError(f"Unknown V5 scene_relation: {requested_relation!r}") from error
            if relation in {SceneRelation.SHARED_SCENE, SceneRelation.PAIRED_SOURCE_SCENE} and source_path is None:
                raise ValueError("shared_scene requires a source scene")
            if relation in {SceneRelation.SOURCE_TO_TARGET, SceneRelation.TARGET_REPLACEMENT, SceneRelation.TARGET_ONLY_WITH_EXPLICIT_BINDING} and target_path is None:
                raise ValueError("target replacement relation requires a target scene")
        elif source_path is None and target_path is None:
            relation = SceneRelation.MOTION_ONLY
        elif target_path is None:
            relation = SceneRelation.SHARED_SCENE
            target_path = source_path
        elif source_path is None:
            relation = SceneRelation.TARGET_ONLY_WITH_EXPLICIT_BINDING
        else:
            relation = SceneRelation.SOURCE_TO_TARGET if source_path != target_path else SceneRelation.SHARED_SCENE
        scene_ref = lambda path, spec: None if path is None else SceneReference(
            path=path, format=str((spec or {}).get("format", "auto")),
            asset_id=str((spec or {}).get("asset_id", path.stem)), metadata=dict(spec or {}),
            unit_scale=((spec or {}).get("unit_scale") if isinstance(spec, dict) else None),
        )
        metadata["detected_motion_format"] = fmt
        metadata["source_scene_declared"] = source_path is not None
        metadata["target_scene_declared"] = target_path is not None
        return ResolvedBundle(
            motion=MotionReference(motion_path, format=fmt, metadata=motion_spec),
            source_scene=scene_ref(source_path, source_spec),
            target_scene=scene_ref(target_path, target_spec),
            scene_relation=relation,
            shared_coordinate_frame=relation == SceneRelation.SHARED_SCENE,
            metadata=metadata,
        )

    @staticmethod
    def _resolve_record_asset(motion_path: Path, value: Any) -> Path:
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (motion_path.parent / path).resolve()

    @classmethod
    def _resolve_grail_asset(cls, motion_path: Path, record: dict[str, Any]) -> Path | None:
        """Resolve only the exact GRAIL asset named by metadata/layout."""
        candidates: list[Path] = []
        if record.get("object_path"):
            candidates.append(cls._resolve_record_asset(motion_path, record["object_path"]))
        obj_data = record.get("obj_data") or {}
        for key in ("path", "object_path", "mesh_path", "asset_path"):
            if isinstance(obj_data, dict) and obj_data.get(key):
                candidates.append(cls._resolve_record_asset(motion_path, obj_data[key]))
        # Released GRAIL reconstructions can omit object_path while keeping an
        # exact stem in recon/ and colocating the USD beside the sequence.
        dataset_root = motion_path.parents[3] if len(motion_path.parents) > 3 else motion_path.parent
        stem = motion_path.stem
        candidates.extend(sorted((dataset_root / "data").glob(f"*/object_usd/{stem}.*")))
        # mesh_data/model.obj is a dataset convention only when it is directly
        # adjacent to this reconstruction or its exact split directory.
        candidates.extend([
            motion_path.parent / "mesh_data" / "model.obj",
            motion_path.parent.parent / "mesh_data" / "model.obj",
        ])
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in {".obj", ".usd", ".usda", ".usdc"}:
                return candidate.resolve()
        return None


class GenericBundleResolver(BundleResolver):
    """Resolver for formats without dataset-owned scene metadata."""

    def resolve(self, motion, **kwargs):
        return super().resolve(motion, **kwargs)


class GrailBundleResolver(BundleResolver):
    def resolve(self, motion, **kwargs):
        bundle = super().resolve(motion, **kwargs)
        if bundle.motion.format.startswith("grail") and bundle.source_scene is None:
            raise FileNotFoundError(
                f"GRAIL motion declares human_data but no exact source scene was found: {bundle.motion.path}"
            )
        return bundle


class HoloSoMoBundleResolver(BundleResolver):
    """HoloSoMo motions are position-only by default and use explicit scenes."""

    def resolve(self, motion, **kwargs):
        bundle = super().resolve(motion, **kwargs)
        if bundle.motion.format.startswith("holosoma") and bundle.scene_relation == SceneRelation.SOURCE_TO_TARGET:
            raise ValueError("HoloSoMo source_to_target requires a manifest contact_binding")
        return bundle
