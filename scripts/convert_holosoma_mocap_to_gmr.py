"""Convert HoloSoMo's 53-joint climbing mocap arrays to GMR joint NPZ."""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


MOCAP_NAMES = [
    "Hips", "Spine", "Spine1", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
    "RightHandRing1", "RightHandRing2", "RightHandRing3",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "LeftFootMod", "RightFootMod",
]

# This order is exactly smplx.joint_names.JOINT_NAMES[:22], which is what the
# wholebody V2 global-joint loader consumes.
SMPLX_BODY_MAPPING = [
    "Hips", "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg",
    "Spine1", "LeftFoot", "RightFoot", "Spine1", "LeftToeBase",
    "RightToeBase", "Neck", "LeftShoulder", "RightShoulder", "Head",
    "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHandMiddle3",
    "RightHandMiddle3",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--source-fps", type=float, default=120.0)
    parser.add_argument("--human-height", type=float, default=1.78)
    args = parser.parse_args()

    joints = np.asarray(np.load(args.input), dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1:] != (53, 3):
        raise ValueError(f"Expected (T, 53, 3), got {joints.shape}")
    index = {name: i for i, name in enumerate(MOCAP_NAMES)}
    converted = np.stack([joints[:, index[name]] for name in SMPLX_BODY_MAPPING], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        global_joint_positions=converted,
        fps=np.asarray(args.source_fps, dtype=np.float32),
        height=np.asarray(args.human_height, dtype=np.float32),
        source_file=np.asarray(str(args.input.resolve())),
        source_joint_names=np.asarray(MOCAP_NAMES),
    )
    print(f"Converted {joints.shape} -> {converted.shape}: {args.output}")


if __name__ == "__main__":
    main()
