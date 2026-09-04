"""Compare two GRAIL V4 outputs and reject contact/collision regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np


def _metrics(path: Path) -> dict:
    with path.open("rb") as stream:
        motion = pickle.load(stream)
    diagnostics = motion.get("terrain_diagnostics", [])
    contacts = motion.get("contact_schedule", [])
    result = {
        "motion": str(path),
        "frames": int(len(motion.get("qpos", []))),
        "qp_failures": int(sum(bool(item.get("qp_failure") or item.get("qp_failures")) for item in diagnostics)),
        "minimum_scene_distance": float(min((item.get("minimum_scene_distance", np.inf) for item in diagnostics), default=np.inf)),
        "maximum_penetration": float(max((item.get("maximum_penetration", 0.0) for item in diagnostics), default=0.0)),
    }
    for channel in ("left_butt", "right_butt", "lower_back", "upper_back"):
        values = [
            float(frame.get("contacts", {}).get(channel, {}).get("signed_distance", np.nan))
            for frame in contacts
            if frame.get("contacts", {}).get(channel, {}).get("object_id")
        ]
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        result[f"{channel}_object_contact_frames"] = int(len(finite))
        result[f"median_{channel}_object_distance"] = float(np.median(finite)) if len(finite) else np.inf
    return result


def compare(baseline: dict, candidate: dict, tolerance: float = 0.002) -> dict:
    failures = []
    if candidate["qp_failures"] > baseline["qp_failures"]:
        failures.append("qp_failures increased")
    if candidate["maximum_penetration"] > baseline["maximum_penetration"] + tolerance:
        failures.append("maximum penetration increased")
    if candidate["minimum_scene_distance"] + tolerance < baseline["minimum_scene_distance"]:
        failures.append("minimum scene distance decreased")
    for channel in ("left_butt", "right_butt", "lower_back", "upper_back"):
        base_n = baseline[f"{channel}_object_contact_frames"]
        cand_n = candidate[f"{channel}_object_contact_frames"]
        if base_n and cand_n < base_n * 0.95:
            failures.append(f"{channel} contact frames decreased")
        base_d = baseline[f"median_{channel}_object_distance"]
        cand_d = candidate[f"median_{channel}_object_distance"]
        if np.isfinite(base_d) and (not np.isfinite(cand_d) or cand_d > base_d + tolerance):
            failures.append(f"{channel} median distance increased")
    return {"accepted": not failures, "failures": failures, "baseline": baseline, "candidate": candidate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=0.002)
    args = parser.parse_args()
    result = compare(_metrics(args.baseline), _metrics(args.candidate), args.tolerance)
    encoded = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    if not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
