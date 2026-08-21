#!/usr/bin/env bash

set -uo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/convert_all_standard_amass_to_ne01.sh [num_cpus] [device]

Example:
  bash scripts/convert_all_standard_amass_to_ne01.sh 6 cuda:0

Processes every standard AMASS .tar.bz2 archive under
~/桌面/gmr_ne01/source_archives/amass_smplx/.
Completed datasets are resumed safely. A failed dataset is recorded and does
not prevent later archives from being processed.
EOF
}

if [[ $# -gt 2 ]]; then
    usage
    exit 1
fi

NUM_CPUS=${1:-6}
DEVICE=${2:-cuda:0}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATA_ROOT=${GMR_DATA_ROOT:-"${HOME}/桌面/gmr_ne01"}
ARCHIVE_DIR="${DATA_ROOT}/source_archives/amass_smplx"
FAILURE_LOG="${SCRIPT_DIR}/../work/amass_conversion_failures.txt"

if ! [[ "${NUM_CPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "num_cpus must be a positive integer: ${NUM_CPUS}" >&2
    exit 1
fi

mkdir -p "$(dirname "${FAILURE_LOG}")"
: > "${FAILURE_LOG}"
shopt -s nullglob
archives=("${ARCHIVE_DIR}"/*.tar.bz2)

if [[ ${#archives[@]} -eq 0 ]]; then
    echo "No standard AMASS .tar.bz2 archives found in: ${ARCHIVE_DIR}" >&2
    exit 1
fi

failed=0
for archive in "${archives[@]}"; do
    echo
    echo "=== $(basename "${archive}") ==="
    if ! bash "${SCRIPT_DIR}/convert_amass_archive_to_ne01.sh" "${archive}" "${NUM_CPUS}" "${DEVICE}"; then
        echo "${archive}" | tee -a "${FAILURE_LOG}"
        failed=$((failed + 1))
    fi
done

if [[ ${failed} -gt 0 ]]; then
    echo "Completed with ${failed} failed archive(s). See: ${FAILURE_LOG}" >&2
    exit 2
fi

echo "Completed all standard AMASS archives."
