"""Inspect a motion file and optionally load it into CanonicalMotion."""

from __future__ import annotations

import argparse
from pathlib import Path

from general_motion_retargeting.motion_adapters import (
    detect_motion_format,
    load_canonical_motion,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("motion", type=Path)
    parser.add_argument("--joint_map", type=Path, default=None)
    parser.add_argument("--body_models", type=Path, default=Path("assets/body_models"))
    parser.add_argument("--target_fps", type=float, default=None)
    parser.add_argument("--bvh_format", choices=("lafan1", "nokov"), default="lafan1")
    parser.add_argument("--load", action="store_true", help="Also run the canonical adapter")
    args = parser.parse_args()

    kind = detect_motion_format(args.motion)
    print(f"format: {kind}")
    if not args.load:
        return
    motion = load_canonical_motion(
        args.motion,
        joint_map=args.joint_map,
        body_models=args.body_models,
        target_fps=args.target_fps,
        bvh_format=args.bvh_format,
    )
    print(f"frames: {motion.frame_count}")
    print(f"joints: {motion.joint_count}")
    print(f"fps: {motion.fps:g}")
    print(f"orientation_valid: {motion.orientation_valid}")
    print(f"root_name: {motion.root_name}")
    print(f"scene_fields: {sorted(motion.scene)}")
    print(f"joint_names: {motion.joint_names}")


if __name__ == "__main__":
    main()
