from __future__ import annotations

import numpy as np

from ..core.schemas import RetargetResult


def validate_result(result: RetargetResult) -> dict:
    q = np.asarray(result.qpos, dtype=float)
    finite = bool(np.isfinite(q).all())
    failures = sum(bool(item.get("qp_failures")) for item in result.diagnostics)
    distances = [float(item.get("minimum_terrain_distance", np.inf)) for item in result.diagnostics]
    return {
        "status": "VALID" if finite and failures == 0 else "INVALID",
        "finite_qpos": finite,
        "qp_failure_count": int(failures),
        "minimum_terrain_distance": float(min(distances, default=np.inf)),
        "frame_count": int(len(q)),
    }
