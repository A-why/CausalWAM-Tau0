# Production Readiness — CausalWAM-Tau0

Final migration checklist for "another server can directly reproduce the
intended experiments". Date: 2026-08-20.

Legend: ✅ PASS · ⚠️ WARN (works but caveat) · ❌ FAIL (blocks reproduction)

---

## Checklist

| # | Item | Verdict | Evidence / note |
|---|---|---|---|
| 1 | **Code paths** | ✅ PASS | All migrated Python uses `${CAUSALWAM_ROOT}` / `os.environ.get("CAUSALWAM_ROOT", …)`; import roots re-mapped (`rl.ercag.*`→`ercag.*`, `rl.flow_grpo.*`→`flow_grpo.*`, `adapters.tau0_robotwin.*`→`adapters.robotwin.*`). 0 stale `rl/` path refs remain. 63 `.py` compile clean. |
| 2 | **Checkpoints** | ✅ PASS | 5/5 KEEP entries present (`tau0_wm/vam`, `tau0_wm/simulator`, `wan2.2-ti2v-5b`, `theta_init_multi_v0/step_500`, `pbb2_turn_switch/step_802`). |
| 3 | **Datasets** | ✅ PASS | 2/2 KEEP entries present; 49 task roots + 49 `statistics_relative_v2.json` + `turn_switch` statistics. |
| 4 | **Configs** | ✅ PASS | Production configs use `${CAUSALWAM_ROOT}` / relative paths. Added `configs/training/vanilla_production.yaml` + `configs/training/ercag_production.yaml`. 13 provenance/smoke/repro configs archived. |
| 5 | **Scripts** | ✅ PASS | 8/8 `bash -n` clean (00–06 + `prepare_ercag_snapshots.sh`). |
| 6 | **No old absolute paths** | ✅ PASS | 0 hardcoded `/data/QWW/CausalWAM` (non-Tau0) in code/configs. Source repo left unchanged (0 re-path markers). |
| 7 | **Vanilla launcher** | ✅ PASS | `04_train_vanilla.sh` → `flow_grpo/tau_vanilla_grpo_production.py` (config-driven; official reward; step_802; `statistics_relative_v2.json`; real RoboTwin observation; group-relative advantage; multi-iteration). **No ACVS / dummy obs / smoke data.** |
| 8 | **ER-CAG launcher** | ✅ PASS | `05_train_ercag.sh` → `eval/r2c_joint_rl_smoke.py` (config-aware via `ercag_production.yaml`; no hardcoded smoke budget/output path; multi-task compatible). |
| 9 | **Snapshot pipeline** | ✅ PASS | Full 5-step capture chain migrated + scripted (`prepare_ercag_snapshots.sh`): native snapshots → roundtrip/hold → SDE candidates → sign probe → tau-relative candidates. |
| 10 | **Runtime artifacts** | ⚠️ WARN | 3 capture outputs not yet generated in-tree (`native_snapshots/S0.pkl`, `sign_probe_results.jsonl`, `tau_candidates/candidates.jsonl`). Produced by `bash scripts/prepare_ercag_snapshots.sh` — a runtime capture step, not a code gap. |

---

## Launcher status

| Launcher | Status (was) | Driver | Config |
|---|---|---|---|
| `scripts/04_train_vanilla.sh` | **PRODUCTION** (SMOKE) | `flow_grpo/tau_vanilla_grpo_production.py` | `configs/training/vanilla_production.yaml` |
| `scripts/05_train_ercag.sh` | **PRODUCTION** (SMOKE) | `eval/r2c_joint_rl_smoke.py` | `configs/training/ercag_production.yaml` |

Both RL launchers are now config-driven, official-reward, and multi-iteration.
The ER-CAG method (`G_i = Q_i − Q_0`, shared `ValueHead`, native future hook,
4/4 gradient gates) is unchanged — only the smoke scale and hardcoded paths were
replaced.

---

## What changed in this pass

- **PART 1 (vanilla)**: new `flow_grpo/tau_vanilla_grpo_production.py` + `configs/training/vanilla_production.yaml`; reward → official `float(check_success())`; checkpoint → `step_802`; statistics → `statistics_relative_v2.json`; observation → real RoboTwin native snapshot; advantage → group-relative (vanilla GRPO); loop → multi-iteration. ACVS / dummy obs / smoke data removed.
- **PART 2 (ER-CAG snapshots)**: migrated the two missing capture drivers from the read-only source (`eval/v3b0_roundtrip_and_hold.py` → `hold_reference.json`, `eval/v3b1_generate_tau_candidates.py` → `tau_candidates/`); re-pointed stale `rl/flow_grpo` path refs in the v3b0/r2c drivers; added `scripts/prepare_ercag_snapshots.sh` (5 sequential capture steps).
- **PART 3 (ER-CAG launcher)**: `configs/training/ercag_production.yaml` (multi-task compatible) + config-aware `05_train_ercag.sh` (no hardcoded `--max-steps 20`, no smoke output path).

---

## Final verdict

**PRODUCTION_READY: NO** *(engineering complete; runtime capture + compute budget pending)*

The package is **release-clean and portable** for every experiment: no old
absolute paths, all KEEP artifacts present, all configs re-pointed, provenance
configs archived, and both RL launchers promoted from SMOKE to PRODUCTION. The
only remaining steps are runtime operations (not code changes):

### Remaining blockers

1. **Generate the 3 snapshot artifacts** — run `bash scripts/prepare_ercag_snapshots.sh`
   (captures `native_snapshots/`, `hold_reference.json`, `sde_candidates/`,
   `sign_probe_results.jsonl`, `tau_candidates/`). This is a capture step, not training.
2. **Set the final RL compute budget** — `training.max_steps` is an explicit
   placeholder in both production configs; choose it at experiment time.

### Exact next experiment

```bash
bash scripts/prepare_ercag_snapshots.sh                      # 1. capture (5 steps)
bash scripts/04_train_vanilla.sh --max-steps <budget>        # 2. vanilla baseline
bash scripts/05_train_ercag.sh   --max-steps <budget>        # 3. ER-CAG
```

---

## Resume-checkpoint audit summary

| Config | Resume field | Exists | Action |
|---|---|---|---|
| `configs/training/theta_init_multi_v0.yaml` | `latest_log_dir`/`optimizer_path` = `null` | ✅ (clean) | Production — fresh start from `checkpoints/tau0_wm/vam` |
| `configs/training/vanilla_production.yaml` | n/a (driver inits optimizer fresh) | ✅ | Production — policy from `checkpoints/pbb2_turn_switch/step_802` |
| `configs/training/ercag_production.yaml` | n/a (ValueHead fresh init, policy from `step_802`) | ✅ | Production |
| `configs/archive/**` | various stale resume fields | ❌ | Archived provenance only (not launched) |
