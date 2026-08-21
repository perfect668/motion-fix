"""Convert the independent ProtoMotions NE01 adapter output to GMR pkl."""
from pathlib import Path
import argparse
import pickle
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--resample-fps", type=float, default=None)
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    for src in sorted(args.src.glob("*_retargeted.npz")):
        d = np.load(src)
        root_wxyz = np.asarray(d["base_frame_wxyz"], dtype=np.float32)
        source_fps = float(np.asarray(d["fps"]).item())
        root_pos = np.asarray(d["base_frame_pos"], dtype=np.float32)
        dof_pos = np.asarray(d["joint_angles"], dtype=np.float32)
        if args.resample_fps and args.resample_fps > 0 and not np.isclose(args.resample_fps, source_fps):
            target_n = max(1, int(round((len(root_pos) - 1) * args.resample_fps / source_fps)) + 1)
            old_t = np.arange(len(root_pos), dtype=np.float64) / source_fps
            new_t = np.arange(target_n, dtype=np.float64) / args.resample_fps
            root_pos = np.stack([np.interp(new_t, old_t, root_pos[:, i]) for i in range(3)], axis=1).astype(np.float32)
            dof_pos = np.stack([np.interp(new_t, old_t, dof_pos[:, i]) for i in range(dof_pos.shape[1])], axis=1).astype(np.float32)
            # Slerp expects scalar-last quaternions; Proto output is wxyz.
            rot = Rotation.from_quat(root_wxyz[:, [1, 2, 3, 0]])
            root_wxyz = Slerp(old_t, rot)(new_t).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
            source_fps = float(args.resample_fps)
        result = {
            "fps": source_fps,
            "root_pos": root_pos,
            "root_rot": root_wxyz[:, [1, 2, 3, 0]],  # GMR stores xyzw
            "dof_pos": dof_pos,
            "local_body_pos": None,
            "link_body_list": None,
        }
        with (args.dst / (src.stem.replace("_retargeted", "") + ".pkl")).open("wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {src.name}")

if __name__ == "__main__":
    main()
