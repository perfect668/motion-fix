#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/convert_smpl_to_ne01.sh <smpl_dir> <output_pkl_dir> [num_cpus] [device]

Example:
  bash scripts/convert_smpl_to_ne01.sh \
    ~/桌面/gmr_ne01/smpl/diverse_actions_8h_v3 \
    ~/桌面/GMR/data/retarget_data/ne01/diverse_actions_8h_v3 \
    2 cuda:0

The script expects SMPL .npz files already normalized to GMR coordinates.
It bridges the file fields, then converts the motions to NE01 robot-motion .pkl
files at 50 FPS. It does not perform a coordinate-system transformation.
Existing .pkl files are skipped, so rerunning the command resumes safely.
EOF
}

if [[ $# -lt 2 || $# -gt 4 ]]; then
    usage
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GMR_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
SMPL_DIR=$(realpath -m "$1")
OUTPUT_DIR=$(realpath -m "$2")
NUM_CPUS=${3:-2}
DEVICE=${4:-cuda:0}
SMPLX_DIR="${GMR_ROOT}/work/smplx_to_ne01/$(basename "${OUTPUT_DIR}")"

if [[ ! -d "${SMPL_DIR}" ]]; then
    echo "SMPL directory does not exist: ${SMPL_DIR}" >&2
    exit 1
fi

if ! [[ "${NUM_CPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "num_cpus must be a positive integer: ${NUM_CPUS}" >&2
    exit 1
fi

mkdir -p "${SMPLX_DIR}" "${OUTPUT_DIR}" "${GMR_ROOT}/work"

echo "GMR root:       ${GMR_ROOT}"
echo "SMPL input:     ${SMPL_DIR}"
echo "SMPL-X bridge:  ${SMPLX_DIR}"
echo "NE01 PKL output: ${OUTPUT_DIR}"
echo "Target FPS:      50"
echo "CPUs/device:     ${NUM_CPUS} / ${DEVICE}"

echo "[1/3] Bridging SMPL fields to GMR's SMPL-X input fields..."
conda run --no-capture-output -n gmr python "${GMR_ROOT}/scripts/smpl_to_smplx.py" \
    --src_folder "${SMPL_DIR}" \
    --tgt_folder "${SMPLX_DIR}" \
    --gender neutral

echo "[2/3] Retargeting SMPL-X motions to NE01..."
conda run --no-capture-output -n gmr python -u "${GMR_ROOT}/scripts/smplx_to_robot_dataset.py" \
    --src_folder "${SMPLX_DIR}" \
    --tgt_folder "${OUTPUT_DIR}" \
    --robot ne01 \
    --tgt_fps 50 \
    --num_cpus "${NUM_CPUS}" \
    --device "${DEVICE}" \
    --min_available_memory_gb 4 \
    --disable_hard_motion_filter \
    --disable_name_exclude_filter

echo "[3/3] Checking output count..."
INPUT_COUNT=$(find "${SMPL_DIR}" -type f -name '*.npz' | wc -l)
OUTPUT_COUNT=$(find "${OUTPUT_DIR}" -type f -name '*.pkl' | wc -l)
echo "Input SMPL files:  ${INPUT_COUNT}"
echo "Output PKL files:  ${OUTPUT_COUNT}"

if [[ "${INPUT_COUNT}" -ne "${OUTPUT_COUNT}" ]]; then
    echo "WARNING: output count differs from input count. Rerun this command to resume failed files." >&2
    exit 2
fi

echo "Done. NE01 robot-motion files are in: ${OUTPUT_DIR}"
