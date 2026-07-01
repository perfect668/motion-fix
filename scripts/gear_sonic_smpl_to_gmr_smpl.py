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
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


POSE_KEYS = ("pose_aa", "poses", "original_pose_aa")
TRANS_KEYS = ("transl", "trans", "translation", "root_trans")
FPS_KEYS = ("fps", "mocap_framerate", "mocap_frame_rate", "original_fps")
BETAS_KEYS = ("betas", "beta", "shape")
COORD_TRANSFORMS = {
    "none": None,
    # Sonic SMPL data uses SMPL's native Y-up frame for root/transl. GMR expects Z-up.
    # Row-vector form: [x, y, z] -> [x, -z, y].
    "sonic_yup_to_gmr_zup": np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    ),
}


def load_motion_file(path):
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}

    if suffix in {".pkl", ".pickle", ".joblib"}:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError(
                "Reading gear-sonic .pkl files requires joblib. "
                "Install joblib in this environment or run this converter with "
                "an environment that already has joblib, such as gear_sonic_train."
            ) from exc

        try:
            return joblib.load(path)
        except Exception:
            with path.open("rb") as f:
                return pickle.load(f)

    raise ValueError(f"Unsupported input file type: {path}")


def choose_key(data, candidates, preferred=None, required=True):
    if preferred and preferred != "auto":
        if preferred in data:
            return preferred
        if required:
            raise KeyError(f"Requested key '{preferred}' not found. Available keys: {sorted(data.keys())}")
        return None

    for key in candidates:
        if key in data:
            return key

    if required:
        raise KeyError(f"None of {candidates} found. Available keys: {sorted(data.keys())}")
    return None


def as_numpy(value, name):
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object:
        array = np.asarray(array.tolist())
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got dtype {array.dtype}")
    return array


def normalize_poses(poses):
    poses = as_numpy(poses, "poses").astype(np.float32, copy=False)
    if poses.ndim == 3 and poses.shape[-1] == 3:
        poses = poses.reshape(poses.shape[0], -1)
    if poses.ndim != 2:
        raise ValueError(f"poses must have shape (T, D) or (T, J, 3), got {poses.shape}")

    if poses.shape[1] > 72:
        poses = poses[:, :72]
    elif poses.shape[1] < 72:
        if poses.shape[1] < 66:
            raise ValueError(f"poses must have at least 66 columns, got {poses.shape}")
        padded = np.zeros((poses.shape[0], 72), dtype=np.float32)
        padded[:, : poses.shape[1]] = poses
        poses = padded

    return np.ascontiguousarray(poses, dtype=np.float32)


def normalize_trans(trans, num_frames, resample_mismatch=True):
    trans = as_numpy(trans, "trans").astype(np.float32, copy=False)
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"trans must have shape (T, 3), got {trans.shape}")

    if trans.shape[0] == num_frames:
        return np.ascontiguousarray(trans, dtype=np.float32)

    if not resample_mismatch:
        raise ValueError(f"pose/trans frame mismatch: {num_frames} pose frames vs {trans.shape[0]} trans frames")

    source_time = np.linspace(0.0, 1.0, trans.shape[0], dtype=np.float32)
    target_time = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    resampled = np.stack(
        [np.interp(target_time, source_time, trans[:, axis]) for axis in range(3)],
        axis=1,
    ).astype(np.float32)
    return np.ascontiguousarray(resampled, dtype=np.float32)


def apply_coord_transform(poses, trans, transform_name):
    transform = COORD_TRANSFORMS[transform_name]
    if transform is None:
        return poses, trans

    root_transform = R.from_matrix(transform.astype(np.float64))
    root_rot = R.from_rotvec(poses[:, :3].astype(np.float64))

    transformed_poses = poses.copy()
    transformed_poses[:, :3] = (root_transform * root_rot).as_rotvec().astype(np.float32)

    transformed_trans = (trans @ transform.T).astype(np.float32)
    return np.ascontiguousarray(transformed_poses), np.ascontiguousarray(transformed_trans)


def parse_betas(args, data):
    if args.betas:
        values = np.array([float(item) for item in args.betas.split(",") if item.strip()], dtype=np.float32)
        if values.size == 0:
            raise ValueError("--betas was provided but no numeric values were parsed")
        return values

    betas_key = choose_key(data, BETAS_KEYS, required=False)
    if betas_key:
        betas = as_numpy(data[betas_key], betas_key).astype(np.float32, copy=False)
        if betas.ndim == 2:
            betas = betas[0]
        betas = betas.reshape(-1)
        if betas.size:
            return np.ascontiguousarray(betas, dtype=np.float32)

    return np.zeros(args.num_betas, dtype=np.float32)


def get_fps(args, data, pose_key):
    if args.fps is not None:
        return float(args.fps)

    if pose_key == "original_pose_aa" and "original_fps" in data:
        return float(np.asarray(data["original_fps"]).item())

    fps_key = choose_key(data, FPS_KEYS, args.fps_key, required=False)
    if fps_key:
        return float(np.asarray(data[fps_key]).item())

    return float(args.default_fps)


def convert_motion(data, args, source_path):
    pose_key = choose_key(data, POSE_KEYS, args.pose_key)
    trans_key = choose_key(data, TRANS_KEYS, args.trans_key)

    poses = normalize_poses(data[pose_key])
    trans = normalize_trans(data[trans_key], poses.shape[0], resample_mismatch=args.resample_mismatched_trans)
    poses, trans = apply_coord_transform(poses, trans, args.coord_transform)
    betas = parse_betas(args, data)
    fps = get_fps(args, data, pose_key)

    return {
        "poses": poses,
        "betas": betas,
        "trans": trans,
        "mocap_framerate": np.array(fps, dtype=np.float32),
        "gender": np.array(args.gender),
        "source_file": np.array(str(source_path)),
        "source_pose_key": np.array(pose_key),
        "source_trans_key": np.array(trans_key),
        "coord_transform": np.array(args.coord_transform),
    }


def convert_file(input_path, output_path, args):
    data = load_motion_file(input_path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict in {input_path}, got {type(data).__name__}")

    converted = convert_motion(data, args, input_path)
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
    parser.add_argument("--default_fps", type=float, default=30.0)
    parser.add_argument("--gender", choices=["male", "female", "neutral"], default="neutral")
    parser.add_argument("--betas", default=None, help="Comma-separated beta values. Defaults to zeros if absent.")
    parser.add_argument("--num_betas", type=int, default=10, help="Number of zero betas to synthesize when absent.")
    parser.add_argument(
        "--coord_transform",
        choices=sorted(COORD_TRANSFORMS.keys()),
        default="sonic_yup_to_gmr_zup",
        help="Coordinate transform applied to root orientation and translation.",
    )
    parser.add_argument(
        "--no_resample_mismatched_trans",
        dest="resample_mismatched_trans",
        action="store_false",
        help="Fail instead of linearly resampling trans when pose/trans frame counts differ.",
    )
    parser.set_defaults(resample_mismatched_trans=True)
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
