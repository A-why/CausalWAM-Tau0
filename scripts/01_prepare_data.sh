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

banner "01_prepare_data — data & suite integrity verification (read-only)"

snapshot "${CAUSALWAM_ROOT}/outputs/multitask_init/final_ready_tasks.json" \
         "${CAUSALWAM_ROOT}/configs/training/theta_init_multi_v0.yaml"

log "Verifying suite manifest, per-task statistics, and data configs ..."

"${TAU0_PY}" - "${CAUSALWAM_ROOT}" 2>&1 <<'PY' | tee -a "${LOG_FILE}"
import json, os, sys
from pathlib import Path
import yaml

root = Path(sys.argv[1])
suite = json.loads((root / "outputs/multitask_init/final_ready_tasks.json").read_text())
ready = suite.get("ready_tasks", [])
n_ready = suite.get("N_ready")
tasks = suite.get("tasks", [])

fails = []
if n_ready != len(ready) or len(ready) != 49:
    fails.append(f"N_ready={n_ready} len(ready)={len(ready)} (expected 49)")

missing_stats, missing_cfg, bad_stats, bad_cfg_ref = [], [], [], []
for name in ready:
    stats = root / "datasets/tau0_robotwin_multitask_v0" / name / "statistics_relative_v2.json"
    cfg = root / "configs/data/robotwin_multitask_v0" / f"{name}.yaml"
    if not stats.exists():
        missing_stats.append(name)
    else:
        s = json.loads(stats.read_text())
        for key in ("action", "state"):
            for sub in ("mean", "std"):
                v = s.get(key, {}).get(sub)
                if not isinstance(v, list) or len(v) != 20:
                    bad_stats.append(f"{name}.{key}.{sub}")
    if not cfg.exists():
        missing_cfg.append(name)
    else:
        try:
            c = yaml.safe_load(cfg.read_text())
            roots = c.get("data", {}).get("data_roots", [])
            if not any(name in str(r) for r in roots):
                bad_cfg_ref.append(name)
        except Exception as e:
            bad_cfg_ref.append(f"{name}({e})")

if missing_stats:
    fails.append(f"missing statistics ({len(missing_stats)}): {missing_stats[:6]}")
if missing_cfg:
    fails.append(f"missing data config ({len(missing_cfg)}): {missing_cfg[:6]}")
if bad_stats:
    fails.append(f"malformed statistics dims ({len(bad_stats)}): {bad_stats[:6]}")
if bad_cfg_ref:
    fails.append(f"data config does not reference task dataset ({len(bad_cfg_ref)}): {bad_cfg_ref[:6]}")

hold = root / "outputs/v3b1_acvs_positive_neutral/tau_candidates/hold_actions.json"
if not hold.exists():
    fails.append("hold_actions.json missing")
else:
    try:
        json.loads(hold.read_text())
    except Exception as e:
        fails.append(f"hold_actions.json unparsable: {e}")

print(f"[01] N_ready={n_ready} ready_tasks={len(ready)} task_meta={len(tasks)}")
print(f"[01] statistics_ok={49-len(missing_stats)-len(bad_stats)}/49")
print(f"[01] data_config_ok={49-len(missing_cfg)-len(bad_cfg_ref)}/49")
hold_fail = any("hold_actions" in f for f in fails)
print(f"[01] hold_actions={'OK' if not hold_fail else 'FAIL'}")
if fails:
    print("[01] FAILURES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("[01] RESULT: PASS")
PY
rc=$?
if [[ ${rc} -eq 0 ]]; then log "RESULT: PASS"; else log "RESULT: FAIL (rc=${rc})"; exit ${rc}; fi
