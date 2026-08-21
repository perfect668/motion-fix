"""Convert the selected GMR SMPL-X files to ProtoMotions PyRoki keypoints.

This is an adapter only; it does not alter any GMR retargeting path.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

NAMES = ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
         "left_ankle", "right_ankle", "left_foot", "right_foot",
         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
         "left_wrist", "right_wrist"]

def quat_wxyz_to_matrix(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()

def convert(src: Path, out: Path, fps: float):
    data, body_model, output, _ = load_smplx_file(src, ROOT / "assets" / "body_models")
    frames, aligned_fps = get_smplx_data_offline_fast(data, body_model, output, tgt_fps=fps)
    T = len(frames)
    pos = np.stack([[frames[t][n][0] for n in NAMES] for t in range(T)], dtype=np.float32)
    ori = np.stack([[quat_wxyz_to_matrix(frames[t][n][1]) for n in NAMES] for t in range(T)], dtype=np.float32)

    # Match ProtoMotions' 18-point convention: 15 semantic points plus two hand
    # auxiliaries and one pelvis/torso auxiliary.
    for t in range(T):
        pos[t, 5] += ori[t, 5] @ np.array([0.05, 0, 0], dtype=np.float32)
        pos[t, 6] += ori[t, 6] @ np.array([0.05, 0, 0], dtype=np.float32)
        pos[t, 7] = pos[t, 5] + ori[t, 5] @ np.array([0.20, 0, 0], dtype=np.float32)
        pos[t, 8] = pos[t, 6] + ori[t, 6] @ np.array([0.20, 0, 0], dtype=np.float32)
    aux_l = pos[:, 13] + np.einsum("tij,j->ti", ori[:, 13], np.array([0, 0, -0.20], dtype=np.float32))
    aux_r = pos[:, 14] + np.einsum("tij,j->ti", ori[:, 14], np.array([0, 0, -0.20], dtype=np.float32))
    aux_p = pos[:, 0] + np.einsum("tij,j->ti", ori[:, 0], np.array([0.20, 0, 0], dtype=np.float32))
    pos = np.concatenate([pos, aux_l[:, None], aux_r[:, None], aux_p[:, None]], axis=1)
    ori = np.concatenate([ori, ori[:, 13:14], ori[:, 14:15], ori[:, 0:1]], axis=1)

    dt = 1.0 / float(aligned_fps)
    left_speed = np.linalg.norm(np.diff(pos[:, 5], axis=0, prepend=pos[:1, 5]), axis=1)
    right_speed = np.linalg.norm(np.diff(pos[:, 6], axis=0, prepend=pos[:1, 6]), axis=1)
    left_contact = ((pos[:, 5, 2] < np.quantile(pos[:, 5, 2], 0.30) + 0.05) & (left_speed / dt < 0.8)).astype(np.int64)
    right_contact = ((pos[:, 6, 2] < np.quantile(pos[:, 6, 2], 0.30) + 0.05) & (right_speed / dt < 0.8)).astype(np.int64)
    contacts = np.stack([left_contact, left_contact, right_contact, right_contact], axis=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, {"positions": pos, "orientations": ori,
                  "left_foot_contacts": contacts[:, :2],
                  "right_foot_contacts": contacts[:, 2:]}, allow_pickle=True)
    return aligned_fps, T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-list", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=50.0)
    args = ap.parse_args()
    for line in args.input_list.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        src = Path(line.strip())
        try:
            fps, frames = convert(src, args.output_dir / (src.stem + ".npy"), args.fps)
            print(f"OK {src.name}: {frames} frames @ {fps:.1f} Hz")
        except Exception as exc:
            print(f"FAIL {src}: {type(exc).__name__}: {exc}")

if __name__ == "__main__":
    main()
