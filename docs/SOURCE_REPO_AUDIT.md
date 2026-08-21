# SOURCE_REPO_AUDIT — /data/QWW/CausalWAM

**Audit date**: 2026-08-20
**Auditor**: automated release-prep pass (read-only; no source file modified)
**Source**: `/data/QWW/CausalWAM` (historical development directory — MUST remain untouched)
**Method**: directory walk + import/`data_class_path`/`model_path`/config-reference tracing +
`outputs/status/*.md` latest-state reading + mtime ordering of `.py`/`.md` files. No
filename guessing — every KEEP/EXCLUDE below is justified by a reference or a status verdict.

---

## 1. 当前目录结构 (top level)

```
CausalWAM/
├── .claude/                        # editor metadata (not code)
├── ER_CAG_GRPO_experiment_plan_v2.md      # experiment plan doc
├── tau0_ercag_incremental_experiment_plan.md
├── adapters/tau0_robotwin/         # τ0 ↔ RoboTwin adapter (FORMAL)
├── configs/                        # training / data / runtime / tau_model / checkpoints
├── data/                           # robotwin_tau0 shim + v3c_dataset (V3-C)
├── datasets/                       # 17 dataset roots (1 formal multitask + 16 historical)
├── eval/                           # closed-loop / RL smoke / audit scripts
├── outputs/                        # 1.5 TB: experiments, checkpoints, status docs
│   ├── checkpoints/                # tau0_wm/{vam,simulator} + wan2.2-ti2v-5b
│   ├── status/                     # 80+ status/report md (source of truth)
│   ├── <experiment dirs>/          # per-experiment outputs
│   └── <trained checkpoints>/      # theta_init_multi_v0_*, pbb2_canonical, pbb/...
├── rl/                             # ercag / flow_grpo / legacy + PB-A/B/C scripts
├── scripts/                        # data build / eval / launch / debug scripts
├── tau-0-wm/                       # τ0 world-action-model package (git repo, branch main)
└── third_party/flow_grpo/          # vendored upstream flow_grpo reference
```

## 2. 正式使用代码路径 (the release mainline)

The project converged (Aug-2026) on a **three-method comparison** over RoboTwin:

| # | Experiment | Code path |
|---|---|---|
| 1 | Multi-task τ0 SFT initialization | `tau-0-wm/main.py` → `tau-0-wm/runner/posttrain.py::Trainer`, config `configs/training/theta_init_multi_v0.yaml`, data `configs/data/robotwin_multitask_v0/*.yaml` (49 tasks) via `tau-0-wm/data/example_dataset.py::CustomLeRobotDataset` |
| 2 | RoboTwin closed-loop evaluation | `scripts/eval_theta_init_multi_closed_loop.py` → `scripts/robotwin_theta_init_eval_one.py` → `eval/theta_init_vam_worker.py` (VAM worker) + `eval/theta_init_closed_loop_screen.py` |
| 3 | Vanilla baseline (True Flow-GRPO, critic-free, official-success reward) | `rl/flow_grpo/*` + `eval/r2c_joint_rl_smoke.py` (policy path only) |
| 4 | ER-CAG method (native paired counterfactual + shared ValueHead + Flow-GRPO) | `rl/ercag/{value_head,official_reward,native_hook,losses,metrics}.py` + `rl/flow_grpo/*` + `eval/r2c_joint_rl_smoke.py` |
| 5 | Multi-task / multi-seed final run | scripts `02–06` (see `scripts/`) over the same pipeline |

Formal component → file mapping (from `outputs/status/PROJECT_STATE.md` V4-A table +
`HISTORICAL_ASSET_REUSE_AUDIT.md`):

| Component | File | Reuse verdict |
|---|---|---|
| τ0 VAM backbone (frozen world) | `tau-0-wm/models/wan_2_2_models/transformers/model.py` + `model_sim.py` | REUSE_AS_IS |
| RoboTwin adapter | `adapters/tau0_robotwin/*` | REUSE_AS_IS |
| Official reward wrapper | `rl/ercag/official_reward.py` (`r_t = float(check_success())`) | REUSE_AS_IS |
| Native future hook | `rl/ercag/native_hook.py` (read-only monkey-patch → `video_states_buffer[-1]`) | REUSE_AS_IS |
| ValueHead (shared scalar head) | `rl/ercag/value_head.py` | REUSE_AS_IS |
| `l_val` loss + gain algebra | `rl/ercag/losses.py` (`l_val`, `l_pair_future`) | REUSE_AS_IS |
| Hold reference action | `hold_actions.json` (`tau_relative_hold_action`) | REUSE_AS_IS |
| ER-CAG structural core (Word method) | `rl/ercag/{shared_environment,action_residual,ercag_model}.py` | KEEP as paper-method reference implementation (diagnostic branch); mainline R2C uses native paired counterfactual instead of explicit F_act |
| True Flow-GRPO (critic-free) | `rl/flow_grpo/*` | REUSE_AS_IS |
| Tie-safe metrics | `rl/ercag/metrics.py` | REUSE_AS_IS |

## 3. 已验证可运行模块

- **FG-A/B/C Flow-GRPO** — `rl/flow_grpo/test_tau_flow_grpo.py` 6/6 gates PASS; `tau_flow_grpo_training.py` + `tau_flow_grpo_stability.py` 20/20 steps (see `FGC_STABILITY_SMOKE.md`).
- **V4-A ER-CAG core** — forward+loss only, 6/6 structural gates + real-arch (dim=3072/30-block) smoke PASS (`V4A_SHARED_ENVIRONMENT_CORE.md`).
- **R2C joint RL smoke** — official-success reward + native shared ValueHead + Flow-GRPO, `JOINT_RL_PIPELINE_READY=YES`, 4/4 gradient gates over 20 iters (`OFFICIAL_SUCCESS_JOINT_RL_SMOKE.md`).
- **PB-B2 turn_switch policy** — `outputs/pbb2_canonical/turn_switch/2026_08_13_03_21_49/step_802`, loads clean (mismatched=0).
- **STAT-A canonical statistics** — round-trip hard gate PASS (position 1.04e-7 m).
- **Multi-task SFT reproduction** — 500-step clean run on GPU1, `FAULT_REPRODUCED=NO`, deterministic (`TECHNICIAN_REPRO_500_RESULT.md`).
- **49-task dataset** — loader READY, `theta_init_multi` feasible (`ROBOTTWIN_ALL_TASK_AUDIT.md`).

## 4. 历史实验 / 废弃模块 (EXCLUDE — do not migrate)

Grouped by the project's own frozen verdicts (`NATIVE_ERCAG_GAIN_AUDIT.md` §0, `HISTORICAL_ASSET_REUSE_AUDIT.md` §9, `PROJECT_STATE.md` "Frozen conclusions"):

| Group | Why excluded | Files (representative) |
|---|---|---|
| **ACVS reward-head scorer** (V3-B1/V3-C) | mis-specified for turn_switch (anti-correlated); V3-C adaptation FAILED | `rl/ercag/*acvs*`, `eval/v3b1_*`, `eval/v3c_*`, `tau-0-wm/runner/posttrain_sim.py`, `data/v3c_dataset.py`, `configs/tau_model/v3c_acvs_turn_switch.yaml`, `configs/data/v3c_*` |
| **F_act / F_env latent-decomposition diagnostics** (V4-B10…B24) | state branch dead / residual collapse; state-source arc exhausted | `rl/ercag/v4b*`, `fact_*`, `causal_state_source_audit.py`, `obs_only_*`, `reference_centered_fact.py`, `structured_state_readout.py`, `trainable_causal_state_encoder.py`, `b19r_*`, `paired_effect_sampling_recovery.py`, `state_conditioned_effect_diagnosis.py`, `gradient_gates_v4b6.py`, `preflight_gates.py`, `test_*`, `smoke_real_arch.py`, `structural_tests_*`, `residual_tests.py` |
| **state-history experiments** | B15–B24 word-history encoders exhausted | `rl/ercag/v4b24_word_complete_history.py`, `eval/v4b24_*` |
| **Legacy FPO** | discontinued, sign bug | `rl/legacy/`, `rl/test_flow_grpo_ratio.py` |
| **PB-A/B/C capture & v2/v3 pilots** | superseded | `rl/pba_*`, `rl/pbb_*`, `rl/v2c_*`, `rl/v2d_*`, `rl/v3a_*` |
| **early smoke / debug / gate / monitor scripts** | dev-only | `scripts/smoke_*`, `scripts/gate_*`, `scripts/monitor_*`, `scripts/*_diagnostic*`, `scripts/test_*`, `scripts/audit_*`, `scripts/generate_all_task_audit.py`, `scripts/make_smoke_config.py`, `scripts/robotwin_r3_*`, `scripts/stata_*`, `scripts/v0*`, `scripts/run_v0d1_smoke.py`, `scripts/gpu_recovery_gate_r4.py` |
| **early closed-loop / adapter experiments** (V0/V1/V4b eval) | superseded | `eval/v0_*`, `eval/v1a_*`, `eval/v1b_*`, `eval/v3b1_*`, `eval/v3c_*`, `eval/v4b*`, `eval/theta_init_*.orig` |
| **historical runtime configs** | smoke/V0D debug | `configs/runtime/v0d*`, `configs/runtime/v0d1_smoke/*`, `configs/runtime/v0d5_smoke/*` |
| **.orig / .pyc / __pycache__** | backups/caches | everywhere |

## 5. checkpoint 位置

`outputs/checkpoints/` (35 GB) + trained model dirs:

| Checkpoint | Path | Size | Keep? |
|---|---|---|---|
| τ0 VAM pretrained | `outputs/checkpoints/tau0_wm/vam/` | 11 G | ✅ KEEP (SFT init + eval + ER-CAG world backbone) |
| τ0 simulator/ACVS backbone | `outputs/checkpoints/tau0_wm/simulator/` | 12 G | ✅ KEEP (frozen native-future world model for ER-CAG; reward head bypassed) |
| Wan2.2-TI2V-5B shared backbone | `outputs/checkpoints/wan2.2-ti2v-5b/` | 14 G | ✅ KEEP (VAE + T5-XXL + backbone, required by both above) |
| SFT init (validated) | `outputs/theta_init_multi_v0_technician_repro_500/canonical_bf16_seed42/step_500/` | 11 G | ✅ KEEP (multi-task SFT initialization) |
| PB-B2 turn_switch policy | `outputs/pbb2_canonical/turn_switch/2026_08_13_03_21_49/step_802/` | 11 G | ✅ KEEP (frozen policy for ER-CAG/vanilla closed-loop) |
| Everything else under `outputs/` | ~1.4 TB of experiment logs / debug ckpts | — | ❌ EXCLUDE (historical debug) |

## 6. dataset 位置

`datasets/` (17 roots, ~29 GB total):

| Dataset | Path | Size | Keep? |
|---|---|---|---|
| Multi-task SFT dataset (49 tasks) | `datasets/tau0_robotwin_multitask_v0/` | 5.4 G | ✅ KEEP (formal SFT training data) |
| Raw multi-task capture | `datasets/robotwin_multitask_raw_v0/` | 22 G | ⚠️ reference-only (source of the 5.4 G lerobot dataset; NOT needed to run) |
| turn_switch canonical lerobot (PB-B2) | `datasets/tau0_robotwin_success_v3_lerobot/turn_switch/` + `statistics_relative_v2.json` | <1 M | ✅ KEEP (PB-B2 policy + ER-CAG stats) |
| All other `datasets/*` (tau0_robotwin_v2, tiny, dev_16hz, tau30hz, lr_test, raw_tau30hz, current_success_raw, success_v3, …) | 16 historical roots | ~1 G | ❌ EXCLUDE (debug / 1-ep / cache) |

Manifest / task list:
- `outputs/multitask_init/final_ready_tasks.json` — the 49-task ready suite (task name, instruction, dataset_lerobot_root, statistics path). **KEEP** (formal task list).

## 7. config 位置

| Config | Path | Keep? |
|---|---|---|
| SFT training (formal) | `configs/training/theta_init_multi_v0.yaml` | ✅ KEEP |
| SFT smoke/repro variants | `configs/training/theta_init_multi_v0_{smoke,smoke500,smoke500_fp32,smoke_fp32,exclusive_replay_2000,technician_repro_100,technician_repro_500}.yaml` | ✅ KEEP (repro/smoke, small) |
| 49 task data configs | `configs/data/robotwin_multitask_v0/*.yaml` | ✅ KEEP |
| turn_switch data configs | `configs/data/robotwin_tau0/{turn_switch_pbb_canonical_eef6d,turn_switch_abs_eef6d,turn_switch_pbb_abs_eef6d,turn_switch_v2_abs_eef6d}.yaml` | ✅ KEEP (canonical stats contract) |
| VAM runtime deploy | `configs/runtime/vam_deploy.yaml` | ✅ KEEP |
| Simulator deploy | `configs/runtime/acvs_deploy.yaml` | ✅ KEEP (frozen world-model infra; scorer role OFF) |
| Adapter contract | `configs/runtime/robotwin_tau0_adapter.yaml` | ✅ KEEP |
| Checkpoint paths | `configs/checkpoints.yaml` | ✅ KEEP (re-path in release) |
| PB-B2 turn_switch SFT | `configs/training/pbb2_canonical_turn_switch.yaml`, `configs/training/pbb_turn_switch.{yaml,json}` | ✅ KEEP (PB-B2 policy provenance) |
| V3-C ACVS | `configs/tau_model/v3c_acvs_turn_switch.yaml`, `configs/data/v3c_*` | ❌ EXCLUDE |
| V0D/smoke runtime | `configs/runtime/v0d*` | ❌ EXCLUDE |

## 8. evaluation 入口

- **Multi-task closed-loop (49-task)**: `scripts/eval_theta_init_multi_closed_loop.py` (orchestrator, resumable) → per-task `scripts/robotwin_theta_init_eval_one.py` → `eval/theta_init_vam_worker.py` (persistent VAM subprocess, pickle protocol) served by `scripts/launch_multitask_tau_server.py`.
- **Turn_switch closed-loop (PB-B2 / vanilla / ER-CAG comparison)**: `eval/pbb2_closed_loop.py` + `eval/vam_server.py` (VAM subprocess).
- **Evaluation-only outcome probes**: `eval/v3b0_sign_probe.py`, `eval/pbc2_run_interventions.py` (never a training reward).

## 9. training 入口

- **SFT (multi-task init)**: `tau-0-wm/main.py --config_file <cfg> --runner_class_path runner/posttrain.py --runner_class Trainer` (launched via `tau-0-wm/scripts/train.sh` or the release `scripts/02_train_sft.sh`). Trainer: `tau-0-wm/runner/posttrain.py`.
- **ER-CAG / Vanilla joint RL**: `eval/r2c_joint_rl_smoke.py` (policy + shared ValueHead joint loop; smoke scale) — the release `scripts/04/05` wrap the production equivalent.
- **Flow-GRPO components**: `rl/flow_grpo/tau_flow_grpo_training.py` (FG-B training loop, ACVS-rewarded reference — do NOT use ACVS as formal reward).

## 10. 当前最终实验流程 (as of 2026-08-20)

1. **SFT**: 49-task `theta_init_multi_v0` (task-balanced sampler, `sub_folder=one_balanced_pass`, 32340 steps). Validated to step_500 on GPU1 (clean). Blocked historically by a post-reboot H100 `misaligned address` driver fault (worked around via SDPA math-backend pin + `AdamW(foreach=False)`).
2. **Closed-loop eval**: resumable 49-task, 3 seeds/task, via VAM worker + RoboTwin env (`/data/QWW/RoboTwin`, `robotwin` conda env).
3. **Vanilla**: True Flow-GRPO critic-free, reward `r_t = float(check_success())`, group-relative advantage.
4. **ER-CAG**: native paired counterfactual (shared ξ) → shared `ValueHead` on native future `Zhat` → `G_i = Q_i − Q_0` → sign-preserving advantage into Flow-GRPO.
5. **Final**: multi-task × multi-seed three-method comparison (1-seed 147 runs / 1470 episodes; 3-seed 441 runs / 4410 episodes, per `ROBOTTWIN_ALL_TASK_AUDIT.md`).

---

## KEEP / EXCLUDE decision table (migration input)

**KEEP (migrate)**:
- `tau-0-wm/` (minus `.git/`, `__pycache__/`, `figures/`, `runner/posttrain_sim.py`, `configs/tau_model/posttrain_sim_taco_play_abs.yaml` — V3-C / taco-play posttrain routes)
- `adapters/tau0_robotwin/` (all)
- `rl/ercag/` formal subset: `__init__.py`, `value_head.py`, `official_reward.py`, `native_hook.py`, `losses.py`, `metrics.py`, `shared_environment.py`, `action_residual.py`, `ercag_model.py`
- `rl/flow_grpo/` (all)
- `eval/` formal subset: `theta_init_vam_worker.py`, `theta_init_closed_loop_screen.py`, `vam_server.py`, `pbb_closed_loop.py`, `pbb_productivity.py`, `pbb2_closed_loop.py`, `pbb2_productivity.py`, `r2c_joint_rl_smoke.py`, `r2c_native_value_smoke.py`, `native_ercag_gain_audit.py`, `v3b0_sign_probe.py`, `v3b0_capture_native_snapshots.py`, `v3b0_generate_sde_candidates.py`, `pbc2_run_interventions.py`, `pbc2_capture_snapshots.py`, `pbc2_generate_sde_candidates.py`
- `scripts/` formal subset: `eval_theta_init_multi_closed_loop.py`, `robotwin_theta_init_eval_one.py`, `launch_multitask_tau_server.py`, `launch_tau_vam_server.py`, `prepare_theta_init_multi_training.py`, `prepare_multitask_init.py`, `build_multitask_datasets.py`, `build_robotwin_multitask_dataset_one.py`, `collect_multitask_demos.py`, `robotwin_multitask_collect_one.py`, `compute_multitask_statistics_one.py`, `finalize_multitask_init_partial.py`, `convert_robotwin_to_lerobot.py`, `convert_robotwin_to_lerobot_final.py`, `convert_robotwin_to_tau0.py`, `validate_tau0_checkpoints.py`
- `configs/` formal subset (see §7 KEEP rows)
- `data/robotwin_tau0/local_tau_dataset.py`
- Datasets: `datasets/tau0_robotwin_multitask_v0/`, `datasets/tau0_robotwin_success_v3_lerobot/turn_switch/`
- Checkpoints: 5 entries in §5 KEEP rows
- Manifest: `outputs/multitask_init/final_ready_tasks.json`

**EXCLUDE (isolate, not migrated)**: everything else, esp. ACVS scorer / V3-C / F_act-F_env diagnostics / B15–B24 / PB-A/B/C capture / v0-v3 pilots / smoke-debug-gate-monitor / `.orig`/`.pyc`.

**Environment note**: two conda envs are required — `tau0_wm` (τ0 + ER-CAG/Flow-GRPO joint RL) and `robotwin` (RoboTwin closed-loop). RoboTwin itself lives OUTSIDE this repo at `/data/QWW/RoboTwin`; the release treats it as an external dependency (see `ENVIRONMENT.md`).
