# Changelog

## 2026-08-20 — Release preparation (Tau0 mainline)

Engineering release of the Tau0 mainline (τ0 WAM + RoboTwin + ER-CAG). The
historical source `/data/QWW/CausalWAM` is unchanged and remains the read-only
origin.

### Added
- Launch scripts `scripts/00_env_check.sh` … `06_eval_all.sh` — bash-runnable,
  auto-locate root, auto `CUDA_VISIBLE_DEVICES`, timestamped logs, config
  snapshots.
- `configs/server_large/large_train.yaml` + `configs/server_large/large_eval.yaml`
  — throughput-only variants (batch size, workers, grad accumulation, rollout
  batch, eval parallel workers); no experiment-logic change.
- Docs: `README.md`, `QUICKSTART.md`, `ENVIRONMENT.md`, `CHANGELOG.md`.

### Changed (migration)
- Layout re-map: `outputs/checkpoints/*` → `checkpoints/*`; import roots
  `rl.ercag.*` → `ercag.*`, `rl.flow_grpo.*` → `flow_grpo.*`,
  `adapters.tau0_robotwin.*` → `adapters.robotwin.*`.
- All hardcoded `/data/QWW/CausalWAM` paths replaced with `${CAUSALWAM_ROOT}`
  / `os.environ.get("CAUSALWAM_ROOT", ...)`; RoboTwin paths → `${ROBOTWIN_ROOT}`
  (default `/data/QWW/RoboTwin`).
- `TauPolicy` / `TauSimulator` config load wrapped in `expand_env_vars`.
- Migrated checkpoint path references in Python re-pointed to `checkpoints/...`.

### Excluded (left in the read-only source only)
ACVS scorer / V3-C, F_act / F_env diagnostics, state-history experiments, early
smoke/debug scripts, failed experiments, and non-KEEP checkpoints/datasets
(~1.4 TB experiment logs). See `docs/SOURCE_REPO_AUDIT.md` §4.

### Known limitations
- `scripts/04_train_vanilla.sh` / `05_train_ercag.sh` are production drivers but
  consume v3b0/v3b1 snapshot artifacts (`native_snapshots/`,
  `sign_probe_results.jsonl`, `tau_candidates/`) that are generated at runtime by
  `scripts/prepare_ercag_snapshots.sh` — run that capture step before either RL
  launcher (see `docs/PRODUCTION_READINESS.md`).
- `configs/training/pbb2_canonical_turn_switch.yaml` resume fields still point
  to a non-migrated historical step_700 (provenance continuation config).

## 2026-08-20 (later) — Production readiness audit

- Added `docs/EXPERIMENT_RUNBOOK.md` (exact reproduction commands) and
  `docs/PRODUCTION_READINESS.md` (PASS/FAIL checklist + verdict).
- Confirmed: SFT (02), VAM server (03), and 49-task eval (06) are production.
- Archived 13 provenance/smoke/repro configs to `configs/archive/` (not
  deleted); `configs/training/` now holds only the formal
  `theta_init_multi_v0.yaml`.

## 2026-08-20 (final) — RL launchers promoted from SMOKE to PRODUCTION

- **Vanilla baseline** (PART 1): new config-driven driver
  `flow_grpo/tau_vanilla_grpo_production.py` + `configs/training/vanilla_production.yaml`.
  Reward → official `float(check_success())`; checkpoint → `pbb2_turn_switch/step_802`;
  statistics → `statistics_relative_v2.json`; observation → real RoboTwin native
  snapshot; advantage → group-relative; loop → multi-iteration. ACVS / dummy obs /
  smoke data removed. `--validate-only` dry-run added.
- **ER-CAG snapshot pipeline** (PART 2): migrated the two missing capture drivers
  (`eval/v3b0_roundtrip_and_hold.py`, `eval/v3b1_generate_tau_candidates.py`) and
  added `scripts/prepare_ercag_snapshots.sh` (5 sequential capture steps: native
  snapshots → roundtrip/hold → SDE candidates → sign probe → tau-relative candidates).
- **ER-CAG launcher** (PART 3): `configs/training/ercag_production.yaml`
  (multi-task compatible) + config-aware `eval/r2c_joint_rl_smoke.py` / `05_train_ercag.sh`
  (no hardcoded smoke budget/output path).
- Stale `rl/` import/path refs removed across 6 drivers (`rl/` does not exist in
  the Tau0 layout).
- Final verdict: **PRODUCTION_READY: NO** — engineering complete and portable; the
  only remaining steps are runtime capture (`scripts/prepare_ercag_snapshots.sh`)
  and setting the final RL `max_steps` budget (see `docs/PRODUCTION_READINESS.md`).
