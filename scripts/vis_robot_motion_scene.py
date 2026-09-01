"""Replay a GMR PKL using an explicit MuJoCo scene XML."""

from __future__ import annotations

import argparse
import pickle
import time

import mujoco
import mujoco.viewer
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    if args.motion.endswith(".npz"):
        motion = np.load(args.motion, allow_pickle=True)
        if "qpos" not in motion:
            raise ValueError("NPZ motion must contain qpos")
        qpos = np.asarray(motion["qpos"], dtype=np.float64)
        fps = float(np.asarray(motion["fps"]).item())
    else:
        with open(args.motion, "rb") as file:
            motion = pickle.load(file)
        fps = float(motion["fps"])
        root_pos = np.asarray(motion["root_pos"])
        root_xyzw = np.asarray(motion["root_rot"])
        dof = np.asarray(motion["dof_pos"])
        qpos = np.concatenate((root_pos, root_xyzw[:, [3, 0, 1, 2]], dof), axis=1)
    model = mujoco.MjModel.from_xml_path(args.xml)
    if qpos.shape[1] != model.nq:
        raise ValueError(f"Motion nq={qpos.shape[1]} but model nq={model.nq}")
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0
        while viewer.is_running():
            start = time.monotonic()
            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(model, data)
            viewer.sync()
            frame += 1
            if frame == len(qpos):
                if not args.loop:
                    break
                frame = 0
            time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
