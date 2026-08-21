#!/usr/bin/env bash

set -uo pipefail

DATA_ROOT=${GMR_DATA_ROOT:-"${HOME}/桌面/gmr_ne01"}
SMPLX_ROOT="${DATA_ROOT}/smplx/amass"
OUTPUT_ROOT="${DATA_ROOT}/robot_motion/amass"
FAILURE_LOG="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/work/amass_conversion_failures.txt"

while true; do
    clear
    date '+%F %T'
    echo

    active=$(ps -eo args | sed -n 's#.*source_archives/amass_smplx/\([^ ]*\).*#\1#p' | head -1)
    workers=$(pgrep -fc 'multiprocessing.spawn import spawn_main')
    echo "Current archive: ${active:-none}"
    echo "Active workers:  ${workers}"
    echo
    printf '%-18s %12s %12s\n' 'Dataset' 'SMPL-X' 'NE01 PKL'
    printf '%-18s %12s %12s\n' '-------' '------' '--------'
    for dataset_path in "${SMPLX_ROOT}"/*; do
        [[ -d "${dataset_path}" ]] || continue
        dataset=$(basename "${dataset_path}")
        input=$(find "${dataset_path}" -type f -name '*_stageii.npz' 2>/dev/null | wc -l)
        output=$(find "${OUTPUT_ROOT}/${dataset}" -type f -name '*.pkl' 2>/dev/null | wc -l)
        printf '%-18s %12s %12s\n' "${dataset}" "${input}" "${output}"
    done
    echo
    free -h | sed -n '2,3p'
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
    if [[ -s "${FAILURE_LOG}" ]]; then
        echo
        echo 'Failed archives:'
        cat "${FAILURE_LOG}"
    fi
    sleep 3
done
