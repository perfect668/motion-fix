"""Convert gear-sonic SMPL files to GMR SMPL npz files.

Purpose:
    Read gear-sonic .pkl/.joblib/.npz SMPL motions and normalize them into the
    SMPL format expected by scripts/smpl_to_smplx.py.

Typical usage:
    conda run -n gear_sonic_train python scripts/gear_sonic_smpl_to_gmr_smpl.py \
        --src_folder data/sonic_smpl_data/selected_10_12h/motions \
        --tgt_folder data/smpl_data/selected_10_12h \
        --coord_transform sonic_yup_to_gmr_zup --overwrite

Output format:
    .npz files with poses, betas, trans, mocap_framerate, gender, source_file,
    source_pose_key, source_trans_key, and coord_transform.

Notes:
    The default coord transform converts Sonic Y-up root orientation and
    translation to GMR Z-up: [x, y, z] -> [x, -z, y].
"""

import argparse
from pathlib import Path

import numpy as np
from data_process.sonic_smpl import COORD_TRANSFORMS, load_sonic_motion, to_gmr_smpl


def convert_file(input_path, output_path, args):
    betas = None
    if args.betas:
        betas = [float(item) for item in args.betas.split(",") if item.strip()]
        if not betas:
            raise ValueError("--betas was provided but no numeric values were parsed")
    motion = load_sonic_motion(
        input_path,
        pose_key=args.pose_key,
        trans_key=args.trans_key,
        fps_key=args.fps_key,
        fps=args.fps,
        default_fps=args.default_fps,
        betas=betas,
        num_betas=args.num_betas,
        resample_mismatched_trans=args.resample_mismatched_trans,
    )
    converted = to_gmr_smpl(motion, coord_transform=args.coord_transform, gender=args.gender)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **converted)
    print(
        f"{input_path} -> {output_path} "
        f"frames={converted['poses'].shape[0]} fps={converted['mocap_framerate'].item():g}"
    )


def iter_input_files(src_folder):
    suffixes = {".pkl", ".pickle", ".joblib", ".npz"}
    for path in sorted(src_folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert gear-sonic SMPL files to GMR smpl_to_smplx-compatible SMPL npz files."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_file", type=Path)
    input_group.add_argument("--src_folder", type=Path)

    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output_file", type=Path)
    output_group.add_argument("--tgt_folder", type=Path)

    parser.add_argument("--pose_key", default="auto", help="Pose key to use, or auto.")
    parser.add_argument("--trans_key", default="auto", help="Translation key to use, or auto.")
    parser.add_argument("--fps_key", default="auto", help="FPS key to use, or auto.")
    parser.add_argument("--fps", type=float, default=None, help="Override output mocap framerate.")
    parser.add_argument(
        "--default_fps",
        type=float,
        default=None,
        help="Fallback FPS used only when the source has no FPS field. Missing FPS fails by default.",
    )
    parser.add_argument("--gender", choices=["male", "female", "neutral"], default="neutral")
    parser.add_argument("--betas", default=None, help="Comma-separated beta values. Defaults to zeros if absent.")
    parser.add_argument("--num_betas", type=int, default=10, help="Number of zero betas to synthesize when absent.")
    parser.add_argument(
        "--coord_transform",
        choices=sorted(COORD_TRANSFORMS.keys()),
        default="sonic_yup_to_gmr_zup",
        help="Coordinate transform applied to root orientation and translation.",
    )
    resample_group = parser.add_mutually_exclusive_group()
    resample_group.add_argument(
        "--resample_mismatched_trans",
        dest="resample_mismatched_trans",
        action="store_true",
        help="Linearly resample translation when pose/trans frame counts differ.",
    )
    resample_group.add_argument(
        "--no_resample_mismatched_trans",
        dest="resample_mismatched_trans",
        action="store_false",
        help="Deprecated compatibility flag; strict mismatch handling is now the default.",
    )
    parser.set_defaults(resample_mismatched_trans=False)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.input_file and not args.output_file:
        parser.error("--output_file is required with --input_file")
    if args.src_folder and not args.tgt_folder:
        parser.error("--tgt_folder is required with --src_folder")

    if args.input_file:
        if args.output_file.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output_file}. Use --overwrite to replace it.")
        convert_file(args.input_file, args.output_file, args)
        return

    converted = 0
    skipped = 0
    for input_path in iter_input_files(args.src_folder):
        rel_path = input_path.relative_to(args.src_folder).with_suffix(".npz")
        output_path = args.tgt_folder / rel_path
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        convert_file(input_path, output_path, args)
        converted += 1
        if args.max_files is not None and converted >= args.max_files:
            break

    print(f"Done. converted={converted} skipped={skipped} output={args.tgt_folder}")


if __name__ == "__main__":
    main()
