"""Atomic V5 result export."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def export_result(output: str | Path, payload: dict[str, Any], arrays: dict[str, Any], diagnostics: Any, *, allow_invalid: bool = False) -> tuple[Path, Path, Path]:
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if payload.get("status") != "VALID" and not allow_invalid:
        raise RuntimeError(
            "V5 validation failed; formal output was not written. "
            "Use --save-invalid-debug only for an explicit debug artifact."
        )
    if payload.get("status") != "VALID":
        output = output.with_name(output.stem + ".INVALID_DEBUG" + output.suffix)
    stream_data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_write_bytes(output, stream_data)
    npz_path = output.with_suffix(".npz")
    temporary_npz = npz_path.with_name(f".{npz_path.name}.tmp.npz")
    np.savez(temporary_npz, **arrays)
    os.replace(temporary_npz, npz_path)
    diagnostics_path = output.with_suffix(".diagnostics.json")
    _atomic_write_bytes(
        diagnostics_path,
        json.dumps(jsonable(diagnostics), indent=2, ensure_ascii=False).encode("utf-8"),
    )
    return output, npz_path, diagnostics_path
