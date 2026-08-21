#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Prepare the ER-CAG snapshot artifacts (V3-B0 outcome contract + V3-B1 tau
# candidates) required by the production RL launchers 04 (vanilla) and
# 05 (ER-CAG). Five sequential capture steps, no RL training, optimizer.step=0.
#
#   1. v3b0_capture_native_snapshots.py   (robotwin) -> native_snapshots/{S0..S3}.pkl
#   2. v3b0_roundtrip_and_hold.py         (robotwin) -> hold_reference.json
#   3. v3b0_generate_sde_candidates.py    (tau0_wm)  -> sde_candidates/candidates.jsonl
#   4. v3b0_sign_probe.py                 (robotwin) -> sign_probe_results.jsonl
#   5. v3b1_generate_tau_candidates.py    (tau0_wm)  -> tau_candidates/{candidates.jsonl,hold_actions.json}
#
# Usage:
#   bash scripts/prepare_ercag_snapshots.sh [--snapshots S0 S1 S2 S3] [--k 8]
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export CAUSALWAM_ROOT="${CAUSALWAM_ROOT:-$ROOT}"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/data/QWW/RoboTwin}"

CONDA_ROOT="${CONDA_ROOT:-/opt/conda}"
TAU0_PY="${CONDA_ROOT}/envs/tau0_wm/bin/python"
ROBOTWIN_PY="${CONDA_ROOT}/envs/robotwin/bin/python"
DISPLAY="${DISPLAY:-:99}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  _n="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  _n="${_n:-0}"
  [[ "${_n}" -lt 1 ]] && _n=1
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((_n - 1)))"
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${CAUSALWAM_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/prepare_ercag_snapshots_${TS}.log"

log()    { printf '[%s] %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*" | tee -a "${LOG_FILE}"; }
banner() { log "================================================================"; log "$*"; log "================================================================"; }

SNAPSHOTS=("S0" "S1" "S2" "S3")
K="8"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --snapshots) SNAPSHOTS=("$2"); shift 2;;   # caller may pass a single snapshot
    --k)         K="$2"; shift 2;;
    *) log "unknown arg: $1"; exit 2;;
  esac
done

log "snapshots=${SNAPSHOTS[*]} k=${K} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} DISPLAY=${DISPLAY}"

run_step() {
  local step="$1" env_py="$2" cmd=("${@:3}")
  banner "Step ${step}: ${cmd[*]}"
  set +e
  "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
  local rc="${PIPESTATUS[0]}"
  set -e
  if [[ ${rc} -ne 0 ]]; then log "Step ${step}: FAIL (rc=${rc})"; exit ${rc}; fi
  log "Step ${step}: OK"
}

# ---- Step 1: native snapshots (RoboTwin) ----
run_step 1 "${ROBOTWIN_PY}" bash -c \
  "cd '${ROBOTWIN_ROOT}' && DISPLAY=${DISPLAY} PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} '${ROBOTWIN_PY}' '${CAUSALWAM_ROOT}/eval/v3b0_capture_native_snapshots.py'"

# ---- Step 2: round-trip gate + hold reference (RoboTwin) ----
run_step 2 "${ROBOTWIN_PY}" bash -c \
  "cd '${ROBOTWIN_ROOT}' && DISPLAY=${DISPLAY} PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} '${ROBOTWIN_PY}' '${CAUSALWAM_ROOT}/eval/v3b0_roundtrip_and_hold.py' --snapshots ${SNAPSHOTS[*]}"

# ---- Step 3: SDE candidates (tau0_wm) ----
run_step 3 "${TAU0_PY}" bash -c \
  "cd '${CAUSALWAM_ROOT}' && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} '${TAU0_PY}' '${CAUSALWAM_ROOT}/eval/v3b0_generate_sde_candidates.py' --snapshots ${SNAPSHOTS[*]} --k ${K}"

# ---- Step 4: sign probe (RoboTwin) ----
run_step 4 "${ROBOTWIN_PY}" bash -c \
  "cd '${ROBOTWIN_ROOT}' && DISPLAY=${DISPLAY} PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} '${ROBOTWIN_PY}' '${CAUSALWAM_ROOT}/eval/v3b0_sign_probe.py' --snapshots ${SNAPSHOTS[*]}"

# ---- Step 5: tau-relative candidates + hold actions (tau0_wm) ----
run_step 5 "${TAU0_PY}" bash -c \
  "cd '${CAUSALWAM_ROOT}' && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} '${TAU0_PY}' '${CAUSALWAM_ROOT}/eval/v3b1_generate_tau_candidates.py' --snapshots ${SNAPSHOTS[*]} --k ${K}"

banner "prepare_ercag_snapshots — DONE"
log "Artifacts under outputs/v3b0_outcome_contract/ and outputs/v3b1_acvs_positive_neutral/"
