"""Convert GMR SMPL npz motions to the SMPL-X npz format used by GMR retargeting.

Purpose:
    Take SMPL files with poses/trans/betas and produce SMPL-X-style files with
    root_orient and pose_body fields. This is the bridge between
    gear_sonic_smpl_to_gmr_smpl.py and smplx_to_robot_dataset.py.

Typical usage:
    conda run -n gmr python scripts/smpl_to_smplx.py \
        --src_folder data/smpl_data/selected_10_12h \
        --tgt_folder data/smplx_data/selected_10_12h \
        --gender neutral

Output:
    One .npz per input motion, with root_orient, pose_body, trans, betas,
    mocap_frame_rate, and gender.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

def convert_smpl_to_smplx(input_path, output_path, gender='neutral'):
    # Load SMPL data
    smpl_data = np.load(input_path, allow_pickle=True)
    data_dict = dict(smpl_data)  # Convert to dict for modification

    # Handle betas padding for legacy SMPL data while preserving higher-order
    # SMPL-X shape coefficients (for example, MOYO stores 300 coefficients).
    if 'betas' in data_dict:
        betas = np.asarray(data_dict['betas'])
        if betas.ndim == 2 and betas.shape[0] == 1:
            betas = betas.reshape(-1)
        if betas.shape == (10,):
            betas = np.concatenate([betas, np.zeros(6, dtype=betas.dtype)])
            print(f"Padded betas from 10 to 16 for {input_path}")
        elif betas.ndim != 1 or betas.size < 16:
            raise ValueError(
                f"Unexpected betas shape: {betas.shape}. Expected a 10-element "
                "legacy SMPL vector or an SMPL-X vector with at least 16 elements."
            )
        data_dict['betas'] = betas

    # Handle mocap_frame_rate variations
    if 'mocap_framerate' in data_dict:
        data_dict['mocap_frame_rate'] = data_dict.pop('mocap_framerate')
        print(f"Renamed 'mocap_framerate' to 'mocap_frame_rate' for {input_path}")

    if 'poses' not in data_dict:
        raise ValueError("Input file does not contain 'poses' key. Is this an SMPL file?")

    poses = data_dict['poses']
    if poses.shape[1] > 72:
        poses = poses[:, :72]

    # Map to SMPL-X format
    data_dict['root_orient'] = poses[:, :3]
    data_dict['pose_body'] = poses[:, 3:66]  # 21 joints x 3 = 63, ignoring SMPL hand poses

    # Ensure gender is set
    if 'gender' not in data_dict:
        data_dict['gender'] = np.array(gender)

    # Remove original poses key
    del data_dict['poses']

    # Save as SMPL-X npz
    np.savez(output_path, **data_dict)
    print(f"Converted {input_path} to {output_path}")

def process_directory(src_folder, tgt_folder, gender='neutral'):
    src_folder = os.path.abspath(src_folder)
    tgt_folder = os.path.abspath(tgt_folder)
    os.makedirs(tgt_folder, exist_ok=True)

    input_paths = []
    for dirpath, dirnames, filenames in os.walk(src_folder):
        dirnames.sort()
        for filename in sorted(filenames):
            if filename.endswith('.npz'):
                input_paths.append(os.path.join(dirpath, filename))

    for input_path in tqdm(input_paths):
        relative_path = os.path.relpath(input_path, src_folder)
        output_path = os.path.join(tgt_folder, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        convert_smpl_to_smplx(input_path, output_path, gender)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SMPL motion data to SMPL-X format.")
    parser.add_argument("--src_folder", type=str, help="Source directory of SMPL .npz files")
    parser.add_argument("--tgt_folder", type=str, help="Target directory for SMPL-X .npz files")
    parser.add_argument("--input_file", type=str, help="Single input SMPL .npz file")
    parser.add_argument("--output_file", type=str, help="Single output SMPL-X .npz file")
    parser.add_argument("--gender", type=str, default="neutral", choices=["male", "female", "neutral"],
                        help="Gender for SMPL-X model if not present in file.")
    args = parser.parse_args()

    if args.src_folder and args.tgt_folder:
        process_directory(args.src_folder, args.tgt_folder, args.gender)
    elif args.input_file and args.output_file:
        convert_smpl_to_smplx(args.input_file, args.output_file, args.gender)
    else:
        parser.print_help()
