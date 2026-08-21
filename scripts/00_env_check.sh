#!/usr/bin/env bash
set -euo pipefail

# ---- auto-locate project root (this script lives in <root>/scripts/) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export CAUSALWAM_ROOT="${CAUSALWAM_ROOT:-$ROOT}"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/data/QWW/RoboTwin}"

# ---- conda interpreters (override CONDA_ROOT if envs live elsewhere) ----
CONDA_ROOT="${CONDA_ROOT:-/opt/conda}"
TAU0_PY="${CONDA_ROOT}/envs/tau0_wm/bin/python"
ROBOTWIN_PY="${CONDA_ROOT}/envs/robotwin/bin/python"

# ---- auto CUDA_VISIBLE_DEVICES (honour caller, else expose all GPUs) ----
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  _n="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  _n="${_n:-0}"
  [[ "${_n}" -lt 1 ]] && _n=1
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((_n - 1)))"
fi

# ---- timestamped log under <root>/logs/ ----
TS="$(date +%Y%m%d_%H%M%S)"
NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
LOG_DIR="${CAUSALWAM_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${NAME}_${TS}.log"

log()    { printf '[%s] %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*" | tee -a "${LOG_FILE}"; }
banner() { log "================================================================"; log "$*"; log "================================================================"; }

banner "00_env_check — environment & artifact validation"
log "CAUSALWAM_ROOT=${CAUSALWAM_ROOT}"
log "ROBOTWIN_ROOT=${ROBOTWIN_ROOT}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

FAILS=()
check() {
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then
    log "PASS  ${desc}"
  else
    log "FAIL  ${desc}"
    FAILS+=("${desc}")
  fi
}

# ---- interpreters ----
check "conda env tau0_wm interpreter"  test -x "${TAU0_PY}"
check "conda env robotwin interpreter" test -x "${ROBOTWIN_PY}"

# ---- external RoboTwin benchmark (formal eval dependency) ----
check "RoboTwin benchmark present"     test -d "${ROBOTWIN_ROOT}"

# ---- checkpoints (5 KEEP entries, key weight files) ----
check "checkpoint tau0_wm/vam"                 test -f "${CAUSALWAM_ROOT}/checkpoints/tau0_wm/vam/diffusion_pytorch_model.bin.index.json"
check "checkpoint tau0_wm/simulator"           test -f "${CAUSALWAM_ROOT}/checkpoints/tau0_wm/simulator/diffusion_pytorch_model.bin"
check "checkpoint wan2.2-ti2v-5b VAE"          test -f "${CAUSALWAM_ROOT}/checkpoints/wan2.2-ti2v-5b/Wan2.2_VAE.pth"
check "checkpoint wan2.2-ti2v-5b T5 encoder"   test -f "${CAUSALWAM_ROOT}/checkpoints/wan2.2-ti2v-5b/models_t5_umt5-xxl-enc-bf16.pth"
check "checkpoint theta_init_multi_v0/step_500" test -f "${CAUSALWAM_ROOT}/checkpoints/theta_init_multi_v0/step_500/diffusion_pytorch_model.bin.index.json"
check "checkpoint pbb2_turn_switch/step_802"    test -f "${CAUSALWAM_ROOT}/checkpoints/pbb2_turn_switch/step_802/diffusion_pytorch_model.bin.index.json"

# ---- runtime + training configs ----
check "config runtime/vam_deploy.yaml"            test -f "${CAUSALWAM_ROOT}/configs/runtime/vam_deploy.yaml"
check "config runtime/acvs_deploy.yaml"           test -f "${CAUSALWAM_ROOT}/configs/runtime/acvs_deploy.yaml"
check "config training/theta_init_multi_v0.yaml"  test -f "${CAUSALWAM_ROOT}/configs/training/theta_init_multi_v0.yaml"
check "config checkpoints.yaml"                   test -f "${CAUSALWAM_ROOT}/configs/checkpoints.yaml"

# ---- suite manifest + hold reference ----
check "suite final_ready_tasks.json"  test -f "${CAUSALWAM_ROOT}/outputs/multitask_init/final_ready_tasks.json"
check "reference hold_actions.json"   test -f "${CAUSALWAM_ROOT}/outputs/v3b1_acvs_positive_neutral/tau_candidates/hold_actions.json"

# ---- datasets: 49 lerobot roots each with per-task statistics ----
N_DATASETS="$(find "${CAUSALWAM_ROOT}/datasets/tau0_robotwin_multitask_v0" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')"
N_STATS="$(find "${CAUSALWAM_ROOT}/datasets/tau0_robotwin_multitask_v0" -maxdepth 2 -name statistics_relative_v2.json 2>/dev/null | wc -l | tr -d ' ')"
log "INFO  dataset roots found: ${N_DATASETS}"
log "INFO  statistics files found: ${N_STATS}"
check "49 dataset roots present"  test "${N_DATASETS}" -eq 49
check "49 statistics files present" test "${N_STATS}" -eq 49

# ---- GPU ----
if command -v nvidia-smi >/dev/null 2>&1; then
  log "INFO  visible GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd, -)"
else
  log "WARN  nvidia-smi not found"
fi

# ---- summary ----
log ""
if [[ "${#FAILS[@]}" -gt 0 ]]; then
  log "Summary: ${#FAILS[@]} failure(s)"
  for f in "${FAILS[@]}"; do log "  - ${f}"; done
  log "RESULT: FAIL"
  exit 1
fi
log "Summary: 0 failures"
log "RESULT: PASS"
