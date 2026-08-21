#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/convert_amass_archive_to_ne01.sh <amass_archive.tar.bz2|amass_archive.zip> [num_cpus] [device]

Example:
  bash scripts/convert_amass_archive_to_ne01.sh \
    ~/桌面/gmr_ne01/source_archives/amass_smplx/ACCAD.tar.bz2 \
    2 cuda:0

The archive is unpacked under ~/桌面/gmr_ne01/smplx/amass/<dataset>/ and
retargeted directly from AMASS SMPL-X to 50 FPS NE01 robot-motion PKL files.
Existing extracted files and PKL files are retained, so rerunning resumes.

The script supports standard AMASS .tar.bz2 archives and the LARa/MOYO SMPL-X
locked-head .zip archives.
EOF
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
    usage
    exit 1
fi

ARCHIVE=$(realpath -m "$1")
NUM_CPUS=${2:-2}
DEVICE=${3:-cuda:0}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GMR_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_ROOT=${GMR_DATA_ROOT:-"${HOME}/桌面/gmr_ne01"}

if [[ ! -f "${ARCHIVE}" ]]; then
    echo "Archive does not exist: ${ARCHIVE}" >&2
    exit 1
fi

if ! [[ "${NUM_CPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "num_cpus must be a positive integer: ${NUM_CPUS}" >&2
    exit 1
fi

case "${ARCHIVE}" in
    *.tar.bz2)
        DATASET=$(basename "${ARCHIVE}" .tar.bz2)
        ARCHIVE_TYPE=tar_bz2
        ;;
    *.zip)
        DATASET=$(basename "${ARCHIVE}" .zip)
        ARCHIVE_TYPE=zip
        ;;
    *)
        echo "Supported archive types are .tar.bz2 and .zip: ${ARCHIVE}" >&2
        exit 1
        ;;
esac
SMPLX_DIR="${DATA_ROOT}/smplx/amass/${DATASET}"
OUTPUT_DIR="${DATA_ROOT}/robot_motion/amass/${DATASET}"
EXTRACTION_MARKER="${SMPLX_DIR}/.gmr_extraction_complete"
SELECTED_DIR="${DATA_ROOT}/smplx_selected/environment_free/${DATASET}"
SELECTION_REPORT="${DATA_ROOT}/reports/environment_free/source/${DATASET}.csv"
QUARANTINE_ROOT=${GMR_QUARANTINE_ROOT:-"${DATA_ROOT}_quarantine"}
QUALITY_QUARANTINE_DIR="${QUARANTINE_ROOT}/quality_rejected_pkl/${DATASET}"

mkdir -p "${SMPLX_DIR}" "${OUTPUT_DIR}"

echo "Archive:          ${ARCHIVE}"
echo "SMPL-X output:    ${SMPLX_DIR}"
echo "NE01 PKL output:  ${OUTPUT_DIR}"
echo "Target FPS:       50"
echo "CPUs/device:      ${NUM_CPUS} / ${DEVICE}"

if [[ -f "${EXTRACTION_MARKER}" ]]; then
    echo "[1/3] Reusing extracted AMASS SMPL-X files..."
else
    echo "[1/3] Extracting AMASS SMPL-X files..."
    if [[ "${ARCHIVE_TYPE}" == tar_bz2 ]]; then
        tar -xjf "${ARCHIVE}" -C "${SMPLX_DIR}"
    else
        unzip -o -q "${ARCHIVE}" -d "${SMPLX_DIR}"
    fi
    touch "${EXTRACTION_MARKER}"
fi

INPUT_COUNT=$(find "${SMPLX_DIR}" -type f -name '*_stageii.npz' | wc -l)
if [[ "${INPUT_COUNT}" -eq 0 ]]; then
    echo "No AMASS *_stageii.npz files found after extraction." >&2
    exit 1
fi

echo "[2/4] Selecting environment-free motions..."
conda run --no-capture-output -n gmr python "${GMR_ROOT}/scripts/data_process/select_environment_free_motions.py" \
    --input_root "${SMPLX_DIR}" \
    --suffix .npz \
    --stageii_only \
    --selected_root "${SELECTED_DIR}" \
    --report_csv "${SELECTION_REPORT}"

if [[ -d "${QUALITY_QUARANTINE_DIR}" ]]; then
    while IFS= read -r -d '' source_file; do
        relative_path=${source_file#"${SELECTED_DIR}/"}
        quarantined_pkl="${QUALITY_QUARANTINE_DIR}/${relative_path%.npz}.pkl"
        if [[ -f "${quarantined_pkl}" ]]; then
            rm "${source_file}"
        fi
    done < <(find "${SELECTED_DIR}" -type l -name '*_stageii.npz' -print0)
fi

SELECTED_COUNT=$(find "${SELECTED_DIR}" \( -type f -o -type l \) -name '*_stageii.npz' | wc -l)
if [[ "${SELECTED_COUNT}" -eq 0 ]]; then
    echo "No environment-free AMASS motions selected." >&2
    exit 1
fi

echo "[3/4] Retargeting selected AMASS SMPL-X motions to NE01..."
conda run --no-capture-output -n gmr python -u "${GMR_ROOT}/scripts/smplx_to_robot_dataset.py" \
    --src_folder "${SELECTED_DIR}" \
    --tgt_folder "${OUTPUT_DIR}" \
    --robot ne01 \
    --tgt_fps 50 \
    --num_cpus "${NUM_CPUS}" \
    --device "${DEVICE}" \
    --min_available_memory_gb 4 \
    --disable_hard_motion_filter \
    --disable_name_exclude_filter

echo "[4/4] Checking output count..."
OUTPUT_COUNT=$(find "${OUTPUT_DIR}" -type f -name '*.pkl' | wc -l)
echo "Input AMASS motions: ${INPUT_COUNT}"
echo "Selected motions:    ${SELECTED_COUNT}"
echo "Output NE01 PKLs:    ${OUTPUT_COUNT}"

if [[ "${SELECTED_COUNT}" -ne "${OUTPUT_COUNT}" ]]; then
    echo "WARNING: output count differs from selected count. Rerun this command to resume failed files." >&2
    exit 2
fi

echo "Done. NE01 robot-motion files are in: ${OUTPUT_DIR}"
