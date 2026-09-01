"""Convert GRAIL reconstruction pickles into GMR-compatible SMPL-X NPZ files.

GRAIL stores the body pose as 55 axis-angle joints (165 values).  GMR only
needs the root and the 21-joint body pose for NE01; hand/object fields are
intentionally ignored by this adapter.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def load_pickle(path: pathlib.Path):
    # GRAIL pickles may reference numpy._core from a newer NumPy release.
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise
        import numpy.core
        import numpy.core.numeric
        sys.modules.setdefault("numpy._core", numpy.core)
        sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
        with path.open("rb") as f:
            return pickle.load(f)


def convert_one(source: pathlib.Path, target: pathlib.Path) -> None:
    record = load_pickle(source)
    if not isinstance(record, dict) or not isinstance(record.get("human_data"), dict):
        raise ValueError(f"{source} does not contain a GRAIL human_data dictionary")
    human = record["human_data"]
    poses = np.asarray(human["poses"], dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] < 66:
        raise ValueError(f"{source}: expected poses (T, >=66), got {poses.shape}")
    trans = np.asarray(human["trans"], dtype=np.float32)
    if trans.shape != (len(poses), 3):
        raise ValueError(f"{source}: trans shape {trans.shape} does not match poses")
    betas = np.asarray(human.get("betas", np.zeros(10)), dtype=np.float32).reshape(-1)
    if betas.size < 10:
        betas = np.pad(betas, (0, 10 - betas.size))
    betas = betas[:10]
    gender = str(human.get("gender", "neutral"))
    fps = float(human.get("mocap_frame_rate", 50.0))
    if fps <= 0:
        fps = 50.0
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        root_orient=poses[:, :3],
        pose_body=poses[:, 3:66],
        trans=trans,
        betas=betas,
        gender=np.asarray(gender),
        mocap_frame_rate=np.asarray(fps),
        source_file=np.asarray(str(source)),
        grail_hands_ignored=np.asarray(True),
    )
    obj = record.get("obj_data") or {}
    if "obj_t" in obj:
        payload["object_pos_w"] = np.asarray(obj["obj_t"], dtype=np.float32)
    if "obj_R" in obj:
        matrices = np.asarray(obj["obj_R"], dtype=np.float64)
        payload["object_quat_w"] = Rotation.from_matrix(matrices).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
    if "obj_scale" in obj:
        payload["object_scale"] = np.asarray(obj["obj_scale"], dtype=np.float32)
    if record.get("object_path"):
        # GRAIL recon stores an internal generated mesh path.  Prefer the
        # matching downloadable USD beside recon/ so HoloSoMo can load the
        # actual chair asset during scene construction.
        usd = source.parent.parent / "object_usd" / f"{source.stem}.usd"
        payload["object_asset_path"] = np.asarray(str(usd if usd.exists() else record["object_path"]))
    np.savez_compressed(target, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", type=pathlib.Path, required=True)
    parser.add_argument("--tgt_folder", type=pathlib.Path, required=True)
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()
    sources = sorted(args.src_folder.rglob("*.pkl"))
    if not sources:
        raise RuntimeError(f"No pickle files found under {args.src_folder}")
    failures = []
    converted = 0
    for source in sources:
        relative = source.relative_to(args.src_folder).with_suffix(".npz")
        target = args.tgt_folder / relative
        if target.exists() and not args.override:
            continue
        try:
            convert_one(source, target)
            converted += 1
            print(f"Converted {source} -> {target}")
        except Exception as exc:
            failures.append(f"{source}: {type(exc).__name__}: {exc}")
    print(f"Converted {converted} files; skipped existing outputs")
    if failures:
        report = args.tgt_folder / "grail_conversion_failures.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise RuntimeError(f"{len(failures)} files failed; see {report}")


if __name__ == "__main__":
    main()
