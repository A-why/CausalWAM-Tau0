#!/usr/bin/env bash
set -euo pipefail

# ---- auto-locate project root (this script lives in <root>/scripts/) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export CAUSALWAM_ROOT="${CAUSALWAM_ROOT:-$ROOT}"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/data/QWW/RoboTwin}"

CONDA_ROOT="${CONDA_ROOT:-/opt/conda}"
TAU0_PY="${CONDA_ROOT}/envs/tau0_wm/bin/python"

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

CONFIG="${CAUSALWAM_ROOT}/configs/training/ercag_production.yaml"
DRIVER="${CAUSALWAM_ROOT}/eval/r2c_joint_rl_smoke.py"
SNAPSHOT=""; K=""; MAX_STEPS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)    CONFIG="$2"; shift 2;;
    --snapshot)  SNAPSHOT="$2"; shift 2;;
    --k)         K="$2"; shift 2;;
    --max-steps) MAX_STEPS="$2"; shift 2;;
    *) log "unknown arg: $1"; exit 2;;
  esac
done

banner "05_train_ercag — ER-CAG joint RL (native paired counterfactual) production"

cat <<'NOTE' | tee -a "${LOG_FILE}"
STATUS: production launcher. eval/r2c_joint_rl_smoke.py implements the ER-CAG
method: native future hook Zhat -> shared ValueHead -> Q_i = V(Zhat_i),
G_i = Q_i - Q_0 (Hold reference), sign-preserving advantage into Flow-GRPO with
official reward r_t = float(check_success()). Config-driven via
configs/training/ercag_production.yaml (multi-task compatible; no hardcoded
smoke budget/output path). Both GPUs are used (policy cuda:0, simulator+ValueHead
cuda:1). Iteration budget is a placeholder until the final compute budget is set.
NOTE

log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
snapshot "${CONFIG}" "${CAUSALWAM_ROOT}/configs/runtime/vam_deploy.yaml" \
         "${CAUSALWAM_ROOT}/configs/runtime/acvs_deploy.yaml"

ARGS=(--config "${CONFIG}")
[[ -n "${SNAPSHOT}" ]]  && ARGS+=(--snapshot "${SNAPSHOT}")
[[ -n "${K}" ]]         && ARGS+=(--k "${K}")
[[ -n "${MAX_STEPS}" ]] && ARGS+=(--max-steps "${MAX_STEPS}")

cd "${CAUSALWAM_ROOT}"
log "launching ${DRIVER#${CAUSALWAM_ROOT}/} ${ARGS[*]} ..."
set +e
"${TAU0_PY}" -u "${DRIVER}" "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
rc="${PIPESTATUS[0]}"
set -e
if [[ ${rc} -eq 0 ]]; then log "RESULT: PASS"; else log "RESULT: FAIL (rc=${rc})"; exit ${rc}; fi
