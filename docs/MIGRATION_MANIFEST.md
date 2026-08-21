# Migration Manifest — CausalWAM-Tau0 release

Date: 2026-08-20
Source (read-only, unchanged): `/data/QWW/CausalWAM`
Destination (this release): `/data/QWW/CausalWAM-Tau0`

This manifest records every migrated dependency in `source → destination` form with a
migration **reason** and the **used_by** evidence (import / config-reference / launch entry).
Nothing is migrated by filename guessing — each entry below is either imported by a training/eval
entry point, referenced by a formal config, or is a checkpoint/dataset/manifest named in the
phase-1 audit (`docs/SOURCE_REPO_AUDIT.md`).

## 0. Layout re-map (directory-level)

| Source (under `/data/QWW/CausalWAM`) | Destination (under `/data/QWW/CausalWAM-Tau0`) | Note |
|---|---|---|
| `tau-0-wm/` | `tau-0-wm/` | wholesale (minus `.git/`, `__pycache__/`, `figures/`, `runner/posttrain_sim.py`, `configs/tau_model/posttrain_sim_taco_play_abs.yaml`) |
| `rl/ercag/` (formal subset) | `ercag/` | import root change `rl.ercag.*` → `ercag.*` |
| `rl/flow_grpo/` (all) | `flow_grpo/` | import root change `rl.flow_grpo.*` → `flow_grpo.*` |
| `adapters/tau0_robotwin/` (all) | `adapters/robotwin/` | import root change `adapters.tau0_robotwin.*` → `adapters.robotwin.*` |
| `eval/` (formal subset, 16 files) | `eval/` | |
| `scripts/` (formal subset, 16 files) | `scripts/` | |
| `data/robotwin_tau0/local_tau_dataset.py` | `data/robotwin_tau0/local_tau_dataset.py` | |
| `configs/` (formal subset, 68 files) | `configs/` | |
| `outputs/checkpoints/*` (5 KEEP entries) | `checkpoints/*` | `outputs/checkpoints/` collapsed to `checkpoints/` |
| `datasets/*` (2 KEEP entries) | `datasets/*` | name preserved |
| `outputs/multitask_init/final_ready_tasks.json` | `outputs/multitask_init/final_ready_tasks.json` | 49-task suite manifest |
| `outputs/v3b1_acvs_positive_neutral/tau_candidates/hold_actions.json` | `outputs/v3b1_acvs_positive_neutral/tau_candidates/hold_actions.json` | Hold reference action |

## 1. Portability conventions (path re-pointing)

All hardcoded `/data/QWW/CausalWAM` paths were removed. Two environment variables are the single
source of truth, exported by the launch scripts (`scripts/0*.sh`) and defaulting sensibly for
direct invocation:

- `CAUSALWAM_ROOT` — this project root. Python default = derived from `__file__` location.
- `ROBOTWIN_ROOT` — external RoboTwin benchmark. Default = `/data/QWW/RoboTwin` (external, not migrated).

Config files (YAML/JSON) use `${CAUSALWAM_ROOT}` / `${ROBOTWIN_ROOT}` placeholders expanded by
`tau-0-wm/utils/config_utils.py::expand_env_vars` (SFT trainer, `TauPolicy`, `TauSimulator`) or by
`_expand()` in `scripts/eval_theta_init_multi_closed_loop.py`. Python entry points resolve the same
variables via `os.environ.get(...)`.

## 2. Core τ0 world model — `tau-0-wm/`

| Source | Destination | Reason | used_by |
|---|---|---|---|
| `tau-0-wm/` (wholesale) | `tau-0-wm/` | Wan2.2-TI2V-5B video-diffusion world backbone (SFT trainer + VAM inference + ER-CAG frozen world model) | `tau-0-wm/main.py`, `runner/posttrain.py`, `web_infer_utils/TauPolicy.py`, `web_infer_utils/simulator/TauSimulator.py` |

Patch applied during migration (does not change model semantics):
- `web_infer_utils/TauPolicy.py` — config load wrapped in `expand_env_vars` (was raw `yaml.load`).
- `web_infer_utils/simulator/TauSimulator.py` — same.

## 3. ER-CAG method core — `ercag/`

| Source | Destination | Reason | used_by |
|---|---|---|---|
| `rl/ercag/official_reward.py` | `ercag/official_reward.py` | official reward wrapper `r_t = float(check_success())` | `eval/r2c_joint_rl_smoke.py`, `scripts/robotwin_theta_init_eval_one.py`, `eval/theta_init_closed_loop_screen.py` |
| `rl/ercag/native_hook.py` | `ercag/native_hook.py` | read-only future hook `video_states_buffer[-1]` | `eval/r2c_joint_rl_smoke.py`, `eval/r2c_native_value_smoke.py` |
| `rl/ercag/value_head.py` | `ercag/value_head.py` | shared ValueHead `[B,3]` scalar head | `eval/r2c_joint_rl_smoke.py`, `eval/r2c_native_value_smoke.py` |
| `rl/ercag/losses.py` | `ercag/losses.py` | `l_val` / `l_pair_future` / gain algebra | `ercag/__init__.py` |
| `rl/ercag/metrics.py` | `ercag/metrics.py` | tie-safe metrics | `ercag/__init__.py` |
| `rl/ercag/shared_environment.py` | `ercag/shared_environment.py` | paper-method F_env reference | `ercag/__init__.py` |
| `rl/ercag/action_residual.py` | `ercag/action_residual.py` | paper-method F_act reference | `ercag/__init__.py` |
| `rl/ercag/ercag_model.py` | `ercag/ercag_model.py` | `forward_pair`/`forward_group`, `_reference_zero_path` | `ercag/__init__.py` |
| `rl/ercag/__init__.py` | `ercag/__init__.py` | package init | all `from ercag.*` imports |

## 4. True Flow-GRPO — `flow_grpo/`

| Source | Destination | Reason | used_by |
|---|---|---|---|
| `rl/flow_grpo/tau_pipeline_with_logprob.py` | `flow_grpo/tau_pipeline_with_logprob.py` | critic-free Flow-GRPO pipeline | `eval/r2c_joint_rl_smoke.py` |
| `rl/flow_grpo/tau_flow_grpo_training.py` | `flow_grpo/tau_flow_grpo_training.py` | FG-B training loop (ACVS-rewarded reference) | release `scripts/04_*` |
| `rl/flow_grpo/tau_flow_grpo_loss.py` | `flow_grpo/tau_flow_grpo_loss.py` | PPO-clipped flow loss | `tau_flow_grpo_training.py` |
| `rl/flow_grpo/tau_flow_grpo_buffer.py` | `flow_grpo/tau_flow_grpo_buffer.py` | trajectory buffer | `tau_flow_grpo_training.py` |
| `rl/flow_grpo/tau_flow_grpo_sde.py` | `flow_grpo/tau_flow_grpo_sde.py` | SDE sampler | `tau_flow_grpo_training.py` |
| `rl/flow_grpo/tau_flow_grpo_stability.py` | `flow_grpo/tau_flow_grpo_stability.py` | stability diagnostic (reference) | `tau_flow_grpo_training.py` |
| `rl/flow_grpo/test_tau_flow_grpo.py` | `flow_grpo/test_tau_flow_grpo.py` | smoke test (reference) | manual |
| `rl/flow_grpo/__init__.py` | `flow_grpo/__init__.py` | package init | all |

## 5. RoboTwin adapter — `adapters/robotwin/`

| Source | Destination | Reason | used_by |
|---|---|---|---|
| `adapters/tau0_robotwin/observation_adapter.py` | `adapters/robotwin/observation_adapter.py` | RoboTwin obs → τ0 input | `eval/theta_init_vam_worker.py`, `eval/vam_server.py`, `scripts/robotwin_theta_init_eval_one.py` |
| `adapters/tau0_robotwin/action_adapter.py` | `adapters/robotwin/action_adapter.py` | τ0 action → RoboTwin action | same + `scripts/robotwin_multitask_collect_one.py` |
| `adapters/tau0_robotwin/rotation_utils.py` | `adapters/robotwin/rotation_utils.py` | rotation conversion | `scripts/convert_robotwin_to_lerobot*.py` |
| `adapters/tau0_robotwin/frame_utils.py` | `adapters/robotwin/frame_utils.py` | world-pose ↔ arm-base | `scripts/build_robotwin_multitask_dataset_one.py` |
| `adapters/tau0_robotwin/gripper_utils.py` | `adapters/robotwin/gripper_utils.py` | gripper convention | same |
| `adapters/tau0_robotwin/contracts.py` | `adapters/robotwin/contracts.py` | shape/order constants | `observation_adapter.py` |
| `adapters/tau0_robotwin/__init__.py` | `adapters/robotwin/__init__.py` | package init | all |

## 6. Evaluation drivers — `eval/` (formal subset)

| Source → Destination | Reason | used_by |
|---|---|---|
| `eval/theta_init_vam_worker.py` | persistent VAM inference worker (pickle protocol) | `scripts/launch_tau_vam_server.py` / `eval_theta_init_multi_closed_loop.py` |
| `eval/theta_init_closed_loop_screen.py` | closed-loop screen driver | `scripts/04_*` |
| `eval/vam_server.py` | VAM subprocess server (turn_switch comparison) | `scripts/05_*` |
| `eval/r2c_joint_rl_smoke.py` | ER-CAG joint RL (policy + ValueHead) driver | `scripts/04_*`, `scripts/05_*` |
| `eval/r2c_native_value_smoke.py` | ValueHead smoke | manual |
| `eval/native_ercag_gain_audit.py` | ER-CAG gain audit | manual |
| `eval/pbb_closed_loop.py`, `eval/pbb2_closed_loop.py` | PB/PB-B2 turn_switch closed-loop | `scripts/05_*` |
| `eval/pbb_productivity.py`, `eval/pbb2_productivity.py` | productivity probe | manual |
| `eval/v3b0_sign_probe.py`, `eval/v3b0_capture_native_snapshots.py`, `eval/v3b0_generate_sde_candidates.py` | outcome probes (never a training reward) | manual |
| `eval/pbc2_run_interventions.py`, `eval/pbc2_capture_snapshots.py`, `eval/pbc2_generate_sde_candidates.py` | PB-C2 intervention probes | manual |

## 7. Launch / prep scripts — `scripts/` (formal subset)

| Source → Destination | Reason |
|---|---|
| `scripts/eval_theta_init_multi_closed_loop.py` | 49-task resumable closed-loop eval orchestrator |
| `scripts/robotwin_theta_init_eval_one.py` | per-task one-process eval child |
| `scripts/launch_multitask_tau_server.py`, `scripts/launch_tau_vam_server.py` | VAM server launchers |
| `scripts/prepare_theta_init_multi_training.py`, `scripts/prepare_multitask_init.py` | SFT dataset + suite freezing |
| `scripts/build_multitask_datasets.py`, `scripts/build_robotwin_multitask_dataset_one.py` | LeRobot root build |
| `scripts/collect_multitask_demos.py`, `scripts/robotwin_multitask_collect_one.py` | demo collection |
| `scripts/compute_multitask_statistics_one.py` | per-task statistics |
| `scripts/finalize_multitask_init_partial.py` | partial report materialization |
| `scripts/convert_robotwin_to_lerobot.py`, `scripts/convert_robotwin_to_lerobot_final.py`, `scripts/convert_robotwin_to_tau0.py` | data conversion |
| `scripts/validate_tau0_checkpoints.py` | checkpoint validation utility |

## 8. Data module

| Source | Destination | Reason | used_by |
|---|---|---|---|
| `data/robotwin_tau0/local_tau_dataset.py` | `data/robotwin_tau0/local_tau_dataset.py` | `CustomLeRobotDataset` (data class) | `configs/data/robotwin_multitask_v0/*.yaml` via `data_class_path: data/example_dataset.py` |

## 9. Configs — `configs/` (68 files)

| Group | Files | Reason |
|---|---|---|
| `configs/checkpoints.yaml` | 1 | checkpoint path map (re-pointed) |
| `configs/training/theta_init_multi_v0.yaml` | 1 | **formal** multi-task SFT init (49-task, task-balanced) |
| `configs/training/pbb2_canonical_turn_switch.yaml` | 1 | **formal** PB-B2 (ER-CAG) turn_switch policy |
| `configs/training/pbb_turn_switch.{yaml,json}` | 2 | PB-B2 policy provenance |
| `configs/training/theta_init_multi_v0_{smoke,smoke500,smoke500_fp32,smoke_fp32}.yaml` | 4 | SFT smoke variants (provenance) |
| `configs/training/theta_init_multi_v0_{technician_repro_100,technician_repro_500,exclusive_replay_2000}.yaml` | 3 | technician reproduction configs (provenance) |
| `configs/data/robotwin_multitask_v0/*.yaml` | 49 | per-task SFT data configs (referenced by `theta_init_multi_v0.yaml`) |
| `configs/data/robotwin_tau0/turn_switch_{pbb_canonical_eef6d,abs_eef6d,pbb_abs_eef6d,v2_abs_eef6d}.yaml` | 4 | turn_switch data configs (canonical stats contract) |
| `configs/runtime/vam_deploy.yaml` | 1 | runtime VAM deploy (VAM worker / server) |
| `configs/runtime/acvs_deploy.yaml` | 1 | frozen world-model infra (scorer role OFF) |
| `configs/runtime/robotwin_tau0_adapter.yaml` | 1 | adapter contract |

## 10. Checkpoints — `checkpoints/`

| Source | Destination | Size | Reason |
|---|---|---|---|
| `outputs/checkpoints/tau0_wm/vam/` | `checkpoints/tau0_wm/vam/` | 11 G | τ0 VAM pretrained (SFT init + eval + ER-CAG backbone) |
| `outputs/checkpoints/tau0_wm/simulator/` | `checkpoints/tau0_wm/simulator/` | 12 G | τ0 simulator backbone (frozen world model; reward head bypassed) |
| `outputs/checkpoints/wan2.2-ti2v-5b/` | `checkpoints/wan2.2-ti2v-5b/` | 14 G | Wan2.2-TI2V-5B shared backbone (VAE + T5-XXL + model) |
| `outputs/theta_init_multi_v0_technician_repro_500/canonical_bf16_seed42/step_500/` | `checkpoints/theta_init_multi_v0/step_500/` | 11 G | validated multi-task SFT init (step_500, clean GPU1 run) |
| `outputs/pbb2_canonical/turn_switch/2026_08_13_03_21_49/step_802/` | `checkpoints/pbb2_turn_switch/step_802/` | 11 G | PB-B2 turn_switch policy (ER-CAG / vanilla closed-loop) |

## 11. Datasets — `datasets/`

| Source | Destination | Size | Reason |
|---|---|---|---|
| `datasets/tau0_robotwin_multitask_v0/` (49 tasks) | `datasets/tau0_robotwin_multitask_v0/` | 5.4 G | formal multi-task SFT training data |
| `datasets/tau0_robotwin_success_v3_lerobot/turn_switch/` | `datasets/tau0_robotwin_success_v3_lerobot/turn_switch/` | <1 M | PB-B2 policy + ER-CAG statistics (`statistics_relative_v2.json`) |

## 12. Reference assets

| Source | Destination | Reason | used_by |
|---|---|---|---|
| `outputs/multitask_init/final_ready_tasks.json` | `outputs/multitask_init/final_ready_tasks.json` | 49-task READY suite manifest (name, instruction, dataset root) | `scripts/eval_theta_init_multi_closed_loop.py` |
| `outputs/v3b1_acvs_positive_neutral/tau_candidates/hold_actions.json` | `outputs/v3b1_acvs_positive_neutral/tau_candidates/hold_actions.json` | Hold Current Pose reference (33-step hold chunk) | `eval/r2c_joint_rl_smoke.py`, `eval/r2c_native_value_smoke.py`, `eval/native_ercag_gain_audit.py` |

## 13. Excluded (not migrated) — reference only

Per `docs/SOURCE_REPO_AUDIT.md` §4 and the KEEP/EXCLUDE table, the following remain only in the
read-only source `/data/QWW/CausalWAM`:

- ACVS scorer / reward head, V3-C ACVS adaptation (`rl/ercag/v3c_*`, `configs/tau_model/v3c_*`)
- F_act / F_env decomposition diagnostics (B15–B24), `JointEffectHead`, `CausalStateEncoder`
- state-history experiment code, early smoke scripts, debug scripts, failed experiments
- PB-A/B/C capture pilots, v0–v3 pilots, smoke-debug-gate-monitor, `.orig`/`.pyc`
- all `datasets/*` except the two KEEP entries (16 historical roots)
- all `outputs/*` except the 5 KEEP checkpoints + 2 reference assets (≈1.4 TB experiment logs)
