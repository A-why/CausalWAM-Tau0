#!/usr/bin/env bash
set -euo pipefail

# ---- auto-locate project root (this script lives in <root>/scripts/) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export CAUSALWAM_ROOT="${CAUSALWAM_ROOT:-$ROOT}"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/data/QWW/RoboTwin}"

CONDA_ROOT="${CONDA_ROOT:-/opt/conda}"
TAU0_PY="${CONDA_ROOT}/envs/tau0_wm/bin/python"
ROBOTWIN_PY="${CONDA_ROOT}/envs/robotwin/bin/python"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  _n="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  _n="${_n:-0}"
  [[ "${_n}" -lt 1 ]] && _n=1
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((_n - 1)))"
fi

TS="$(date +%Y%m%d_%H%M%S)"
NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
LOG_DIR="${CAUSALWAM_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${NAME}_${TS}.log"

log()    { printf '[%s] %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*" | tee -a "${LOG_FILE}"; }
banner() { log "================================================================"; log "$*"; log "================================================================"; }
snapshot() {
  local dst="${LOG_DIR}/${TS}/config_snapshot" src rel
  mkdir -p "${dst}"
  for src in "$@"; do
    if [[ -e "${src}" ]]; then
      rel="${src#${CAUSALWAM_ROOT}/}"
      mkdir -p "${dst}/$(dirname "${rel}")"
      if [[ -d "${src}" ]]; then cp -rp "${src}" "${dst}/${rel}"; else cp -p "${src}" "${dst}/${rel}"; fi
      log "snapshot: ${rel}"
    else
      log "snapshot: MISSING ${src}"
    fi
  done
}

banner "02_train_sft — multi-task SFT init (theta_init_multi_v0)"

CONFIG="${CAUSALWAM_ROOT}/configs/training/theta_init_multi_v0.yaml"
log "env: tau0_wm  -> ${TAU0_PY}"
log "config: ${CONFIG}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# config snapshot: training config + 49 data configs + runtime deploy config
snapshot "${CONFIG}" \
         "${CAUSALWAM_ROOT}/configs/data/robotwin_multitask_v0" \
         "${CAUSALWAM_ROOT}/configs/runtime/vam_deploy.yaml" \
         "${CAUSALWAM_ROOT}/configs/checkpoints.yaml"

cd "${CAUSALWAM_ROOT}/tau-0-wm"
log "launching Trainer (cwd=$(pwd)) ..."
set +e
"${TAU0_PY}" -u main.py \
  --config_file "${CONFIG}" \
  --runner_class_path runner/posttrain.py \
  --runner_class Trainer \
  --mode train 2>&1 | tee -a "${LOG_FILE}"
rc="${PIPESTATUS[0]}"
set -e
if [[ ${rc} -eq 0 ]]; then log "RESULT: PASS (SFT complete)"; else log "RESULT: FAIL (rc=${rc})"; exit ${rc}; fi
