#!/usr/bin/env bash

set -uo pipefail

GPU_INDEX=0
POLL_SECONDS=30
REQUIRED_IDLE_CHECKS=4

GMR_ROOT="/home/user/桌面/GMR"
INPUT_DIR="/home/user/桌面/humanoid_motion_sources/curated/smplx_selected/motions"
OUTPUT_DIR="/home/user/桌面/humanoid_retarget_results/gmr/ne01/baseline_v1/motions"
LOG_DIR="/home/user/桌面/humanoid_retarget_results/gmr/ne01/baseline_v1"
RUN_LOG="${LOG_DIR}/gmr_conversion.log"
LOCK_FILE="${GMR_ROOT}/work/wait_then_convert_amass_selected_1444.lock"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi was not found."
    exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
    log "ERROR: input directory does not exist: ${INPUT_DIR}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    log "ERROR: another wait/conversion process already holds ${LOCK_FILE}"
    exit 1
fi

idle_checks=0
log "Waiting for GPU ${GPU_INDEX} to have no compute processes."
log "A stable idle period of $((POLL_SECONDS * (REQUIRED_IDLE_CHECKS - 1))) seconds is required."

while true; do
    if ! compute_pids=$(nvidia-smi -i "${GPU_INDEX}" \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null); then
        idle_checks=0
        log "WARNING: failed to query GPU ${GPU_INDEX}; retrying in ${POLL_SECONDS}s."
    elif [[ -z "${compute_pids//[[:space:]]/}" ]]; then
        idle_checks=$((idle_checks + 1))
        log "GPU ${GPU_INDEX} idle check ${idle_checks}/${REQUIRED_IDLE_CHECKS}."
        if (( idle_checks >= REQUIRED_IDLE_CHECKS )); then
            break
        fi
    else
        idle_checks=0
        log "GPU ${GPU_INDEX} busy; compute PID(s): $(tr '\n' ' ' <<<"${compute_pids}" | xargs)"
    fi

    sleep "${POLL_SECONDS}"
done

log "GPU ${GPU_INDEX} is stably idle. Starting GMR conversion."
log "Conversion output: ${OUTPUT_DIR}"
log "Conversion log: ${RUN_LOG}"

cd "${GMR_ROOT}" || exit 1
bash scripts/convert_smpl_to_ne01.sh \
    "${INPUT_DIR}" \
    "${OUTPUT_DIR}" \
    2 cuda:0 \
    >>"${RUN_LOG}" 2>&1
status=$?

if (( status == 0 )); then
    log "GMR conversion completed successfully."
else
    log "ERROR: GMR conversion exited with status ${status}. See ${RUN_LOG}."
fi

exit "${status}"
