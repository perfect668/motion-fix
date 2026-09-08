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
        fmt = str(motion_spec.get("format", motion_format or "auto"))
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
        if source_path is None and manifest is None:
            record = _pickle_record(motion_path) if motion_path.suffix.lower() == ".pkl" else None
            if record and record.get("object_path"):
                source_path = self._resolve_record_asset(motion_path, record["object_path"])
                if not source_path.is_file() and "human_data" in record:
                    # GRAIL stores an internal results_collection path in the
                    # reconstruction, while the released asset is colocated
                    # under data/<split>/object_usd. Resolve only the exact
                    # object stem; never select an arbitrary directory asset.
                    dataset_root = motion_path.parents[3] if len(motion_path.parents) > 3 else motion_path.parent
                    stem = motion_path.stem
                    candidates = sorted((dataset_root / "data").glob(f"*/object_usd/{stem}.*"))
                    if candidates:
                        source_path = candidates[0].resolve()
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

        if source_path is None and target_path is None:
            relation = SceneRelation.NO_SOURCE_SCENE
        elif target_path is None:
            relation = SceneRelation.PAIRED_SOURCE_SCENE
            target_path = source_path
        elif source_path is None:
            relation = SceneRelation.TARGET_REPLACEMENT
        else:
            relation = SceneRelation.TARGET_REPLACEMENT if source_path != target_path else SceneRelation.PAIRED_SOURCE_SCENE
        scene_ref = lambda path, spec: None if path is None else SceneReference(
            path=path, format=str((spec or {}).get("format", "auto")),
            asset_id=str((spec or {}).get("asset_id", path.stem)), metadata=dict(spec or {}),
        )
        metadata["detected_motion_format"] = fmt
        metadata["source_scene_declared"] = source_path is not None
        metadata["target_scene_declared"] = target_path is not None
        return ResolvedBundle(
            motion=MotionReference(motion_path, format=fmt, metadata=motion_spec),
            source_scene=scene_ref(source_path, source_spec),
            target_scene=scene_ref(target_path, target_spec),
            scene_relation=relation,
            shared_coordinate_frame=relation == SceneRelation.PAIRED_SOURCE_SCENE,
            metadata=metadata,
        )

    @staticmethod
    def _resolve_record_asset(motion_path: Path, value: Any) -> Path:
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (motion_path.parent / path).resolve()


class GenericBundleResolver(BundleResolver):
    pass


class GrailBundleResolver(BundleResolver):
    pass


class HoloSoMoBundleResolver(BundleResolver):
    pass
