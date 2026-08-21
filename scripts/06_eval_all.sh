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

HOST="127.0.0.1"
PORT="8765"
EXECUTION_STEP="33"
INFERENCE_STEPS="5"
EPISODES="3"
TASKS_ARGS=()
LAUNCH_SERVER=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)            HOST="$2"; shift 2;;
    --port)            PORT="$2"; shift 2;;
    --execution-step)  EXECUTION_STEP="$2"; shift 2;;
    --inference-steps) INFERENCE_STEPS="$2"; shift 2;;
    --episodes)        EPISODES="$2"; shift 2;;
    --tasks)           shift; while [[ $# -gt 0 && "$1" != --* ]]; do TASKS_ARGS+=("$1"); shift; done;;
    --launch-server)   LAUNCH_SERVER=1; shift;;
    *) log "unknown arg: $1"; exit 2;;
  esac
done

banner "06_eval_all — 49-task closed-loop initialization eval"

server_up() {
  "${ROBOTWIN_PY}" -c "import socket,sys; socket.create_connection((sys.argv[1],int(sys.argv[2])),timeout=2).close()" "${HOST}" "${PORT}" >/dev/null 2>&1
}

if ! server_up; then
  if [[ "${LAUNCH_SERVER}" -eq 1 ]]; then
    log "VAM server not reachable; launching 03 in background ..."
    bash "${CAUSALWAM_ROOT}/scripts/03_launch_vam_server.sh" --host "${HOST}" --port "${PORT}" --background
    for _ in $(seq 1 60); do
      server_up && break
      sleep 5
    done
  fi
  if ! server_up; then
    log "VAM server NOT reachable at ${HOST}:${PORT}"
    log "Start it first: bash scripts/03_launch_vam_server.sh (or pass --launch-server)"
    exit 1
  fi
fi
log "VAM server reachable at ${HOST}:${PORT}"

snapshot "${CAUSALWAM_ROOT}/outputs/multitask_init/final_ready_tasks.json"

cd "${CAUSALWAM_ROOT}"
ARGS=(scripts/eval_theta_init_multi_closed_loop.py --host "${HOST}" --port "${PORT}" \
      --execution-step "${EXECUTION_STEP}" --inference-steps "${INFERENCE_STEPS}" \
      --episodes "${EPISODES}")
if [[ "${#TASKS_ARGS[@]}" -gt 0 ]]; then ARGS+=(--tasks "${TASKS_ARGS[@]}"); fi

log "env: robotwin -> ${ROBOTWIN_PY}"
log "launching eval orchestrator (cwd=${CAUSALWAM_ROOT}) ..."
set +e
"${ROBOTWIN_PY}" -u "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
rc="${PIPESTATUS[0]}"
set -e
if [[ ${rc} -eq 0 ]]; then
  log "RESULT: PASS (all 49 tasks complete)"
else
  log "RESULT: PARTIAL/FAIL (rc=${rc}) — inspect outputs/multitask_init/closed_loop_eval.json"
fi
exit ${rc}
