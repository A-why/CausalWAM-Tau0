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

CKPT="${CAUSALWAM_ROOT}/checkpoints/theta_init_multi_v0/step_500"
HOST="127.0.0.1"
PORT="8765"
DEVICE="cuda:0"
BACKGROUND=0

# Default initial normalization = first ready task's statistics.
INITIAL_STATS="$("${TAU0_PY}" -c "import json,os; s=json.load(open(os.environ['CAUSALWAM_ROOT']+'/outputs/multitask_init/final_ready_tasks.json')); t=s['ready_tasks'][0]; print(os.path.join(os.environ['CAUSALWAM_ROOT'],'datasets/tau0_robotwin_multitask_v0',t,'statistics_relative_v2.json'))")"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)         CKPT="$2"; shift 2;;
    --initial-statistics) INITIAL_STATS="$2"; shift 2;;
    --host)               HOST="$2"; shift 2;;
    --port)               PORT="$2"; shift 2;;
    --device)             DEVICE="$2"; shift 2;;
    --background)         BACKGROUND=1; shift;;
    *) log "unknown arg: $1"; exit 2;;
  esac
done

banner "03_launch_vam_server — multitask τ0 VAM server"
log "checkpoint: ${CKPT}"
log "initial-statistics: ${INITIAL_STATS}"
log "host:port ${HOST}:${PORT}   device ${DEVICE}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

[[ -d "${CKPT}" ]]         || { log "checkpoint dir missing: ${CKPT}"; exit 1; }
[[ -f "${INITIAL_STATS}" ]] || { log "initial-statistics missing: ${INITIAL_STATS}"; exit 1; }

snapshot "${CAUSALWAM_ROOT}/configs/runtime/vam_deploy.yaml" \
         "${CAUSALWAM_ROOT}/outputs/multitask_init/final_ready_tasks.json"

cd "${CAUSALWAM_ROOT}"
if [[ "${BACKGROUND}" -eq 1 ]]; then
  log "launching server in background (log -> ${LOG_FILE})"
  nohup "${TAU0_PY}" -u scripts/launch_multitask_tau_server.py \
    --checkpoint "${CKPT}" --initial-statistics "${INITIAL_STATS}" \
    --host "${HOST}" --port "${PORT}" --device "${DEVICE}" \
    > "${LOG_FILE}" 2>&1 &
  log "server PID: $!"
  log "wait for 'MULTITASK_TAU_SERVER_READY' in ${LOG_FILE}"
  exit 0
fi

log "launching server in foreground (Ctrl-C to stop) ..."
"${TAU0_PY}" -u scripts/launch_multitask_tau_server.py \
  --checkpoint "${CKPT}" --initial-statistics "${INITIAL_STATS}" \
  --host "${HOST}" --port "${PORT}" --device "${DEVICE}" 2>&1 | tee -a "${LOG_FILE}"
