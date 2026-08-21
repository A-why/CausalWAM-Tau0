# CausalWAM-Tau0 — Final Experiment Status

This document is the final reproducibility record for the CausalWAM-Tau0
release package, written before large-scale experimental execution on another
server. It records only verified facts. No training/evaluation code was changed
to produce this document.

---

## 1. Project Overview

- **Project name:** CausalWAM-Tau0
- **Main goal:** τ0-based World Action Model (WAM) adaptation for RoboTwin
  manipulation with ER-CAG (Environment-Referenced Causal Action Gain).
- **Current status:** Production experiment ready.

### Final pipeline

```
Tau0 pretrained model
        |
        v
Multi-task initialization / SFT
        |
        v
Vanilla GRPO baseline
        |
        v
ER-CAG optimization
        |
        v
RoboTwin closed-loop evaluation
```

- **Tau0 pretrained model** — Wan2.2-TI2V-5B video-diffusion backbone with the
  τ0 VAM (world action model) and frozen simulator.
- **Multi-task initialization / SFT** — 49-task balanced SFT pass
  (`theta_init_multi_v0/step_500`).
- **Vanilla GRPO baseline** — True Flow-GRPO (critic-free, PPO-clipped,
  `beta_kl=0`), group-relative advantage, official sparse success reward.
- **ER-CAG optimization** — native paired counterfactual reference → shared
  ValueHead → causal action gain `G_i = Q_i − Q_0`.
- **RoboTwin closed-loop evaluation** — 49 tasks, official `check_success()`
  reward, τ0↔RoboTwin action adapter.

---

## 2. Repository Information

- **Project root:** `/data/QWW/CausalWAM-Tau0`
- **Original development repository:** `/data/QWW/CausalWAM` (read-only origin)
- **State:**
  - Original repository untouched (no files modified/removed/renamed).
  - Release package is independent and self-contained (given the two conda
    envs and the external RoboTwin benchmark).

### Directory structure summary

| Directory | Contents |
|---|---|
| `tau-0-wm/` | Wan2.2-TI2V-5B world backbone (SFT trainer + VAM inference) |
| `ercag/` | ER-CAG method core (ValueHead, native future hook, official reward, losses) |
| `flow_grpo/` | True Flow-GRPO (critic-free, PPO-clipped flow loss) |
| `adapters/` | RoboTwin ↔ τ0 observation / action adapters |
| `datasets/` | RoboTwin multi-task SFT data (49 tasks) + turn_switch statistics |
| `configs/` | training / data / runtime configs + `server_large/` variants |
| `scripts/` | launch scripts `00_env_check.sh` … `06_eval_all.sh` |
| `checkpoints/` | 5 KEEP checkpoints (~58 GB) |
| `eval/` | evaluation drivers (closed-loop, joint-RL smoke, probes) |
| `outputs/` | generated artifacts (manifests, snapshots, smoke results) |
| `docs/` | reproducibility and migration documentation |

---

## 3. Environment Record

Validated environment (recorded at smoke-test time):

| Item | Value |
|---|---|
| GPU | 2 × NVIDIA H100 80GB HBM3 |
| NVIDIA driver | 580.173.02 (CUDA runtime 13.0) |
| OS | Linux 7.0.0-29-generic (Ubuntu 24.04.2) |
| Conda env `tau0_wm` | Python 3.10.20 · PyTorch 2.7.1+cu126 · CUDA 12.6 |
| Conda env `robotwin` | Python 3.10.20 · PyTorch 2.4.1+cu121 · CUDA 12.1 |

- `tau0_wm` is used for the τ0 world model, SFT, and ER-CAG / Flow-GRPO RL.
- `robotwin` is used for RoboTwin closed-loop evaluation.
- Both envs live at `/opt/conda/envs/` (override with `CONDA_ROOT` if
  relocated). External RoboTwin benchmark at `$ROBOTWIN_ROOT` (default
  `/data/QWW/RoboTwin`).

**Environment validation:** `scripts/00_env_check.sh` — **PASS**.

---

## 4. Data Record

| Item | Value |
|---|---|
| RoboTwin task count | 49 tasks |
| Demonstrations (current available) | 146 episodes total across 49 tasks (≈3 per task); turn_switch single-task success set has 3 episodes / 241 frames |
| Dataset location | `datasets/tau0_robotwin_multitask_v0/` (49 tasks) · `datasets/tau0_robotwin_success_v3_lerobot/turn_switch/` |
| Manifest location | `outputs/multitask_init/final_ready_tasks.json` (`N_ready = 49`) |

- The multi-task SFT training data is `datasets/tau0_robotwin_multitask_v0/`.
- The turn_switch statistics (`statistics_relative_v2.json`) drive the PB-B2
  policy and ER-CAG experiments.

**Data preprocessing status:** **PASS**.

---

## 5. Checkpoint Record

| Checkpoint | Purpose | Path | Status |
|---|---|---|---|
| Tau0 pretrained (VAM) | τ0 world action model backbone (SFT init + eval + ER-CAG) | `checkpoints/tau0_wm/vam/` | Present |
| SFT initialization | validated multi-task SFT init (49 tasks) | `checkpoints/theta_init_multi_v0/step_500/` | Present |
| Vanilla baseline | PB-B2 turn_switch policy (vanilla / ER-CAG start) | `checkpoints/pbb2_turn_switch/step_802/` | Present |
| ER-CAG checkpoint | produced by `scripts/05_train_ercag.sh` | (output of ER-CAG training) | **Placeholder — not yet produced** |

Supporting pretrained checkpoints (also part of the 5 KEEP entries):

| Checkpoint | Purpose | Path | Status |
|---|---|---|---|
| τ0 simulator (frozen WM) | native future hook `Zhat` (ER-CAG) | `checkpoints/tau0_wm/simulator/` | Present |
| Wan2.2-TI2V-5B backbone | VAE + T5 text encoder + base model | `checkpoints/wan2.2-ti2v-5b/` | Present |

Debug checkpoints are excluded from the release package.

---

## 6. Production Pipeline Status

| Component | Entry | Status |
|---|---|---|
| SFT | `scripts/02_train_sft.sh` | PASS |
| VAM deployment | `scripts/03_launch_vam_server.sh` | PASS |
| Evaluation | `scripts/06_eval_all.sh` | PASS |
| Vanilla | `scripts/04_train_vanilla.sh` | PASS |
| ER-CAG | `scripts/05_train_ercag.sh` | PASS |

All five stages execute end-to-end on the production code paths.

---

## 7. Smoke Test Record

| Section | Result |
|---|---|
| Environment smoke | PASS |
| ER-CAG snapshot | PASS (artifacts: `S0.pkl`, tau candidates, sign probe) |
| Vanilla smoke (20 steps) | PASS |
| ER-CAG smoke (20 steps) | PASS |
| Evaluation smoke | PASS |

### Vanilla prompt-alignment fix

| | Value |
|---|---|
| Before | `"turn on the switch"` (hardcoded literal) |
| After | `"use the robotic arm to click the switch"` (official RoboTwin task instruction) |

- **Determinism:** `6.3e-2 → 0` (max|sampled − stored| action diff, after fix).
- The vanilla rollout now obtains the instruction from the same source as the
  capture/sign-probe pipeline and ER-CAG (`adapt_observation → get_instruction`
  → RoboTwin `description/task_instruction/{task}.json`), guarded by a
  `PROMPT_ALIGNMENT` regression check in the driver.

---

## 8. Reproducibility Guarantees

Verified items:

- ✅ Official reward used (`r_t = float(check_success())` in `{0,1}`, real RoboTwin outcome).
- ✅ Official task instruction used (RoboTwin task metadata, not a hardcoded literal).
- ✅ Action representation unchanged (relative, `eef6d`).
- ✅ Prompt alignment checked (regression gate in the vanilla driver).
- ✅ No stale paths (all paths resolved via `${CAUSALWAM_ROOT}` / `${ROBOTWIN_ROOT}`).
- ✅ No smoke-only launcher remains (smoke reuses the production scripts).
- ✅ No algorithm changes.

---

## 9. Formal Experiment Plan

### Initialization

Multi-task SFT (49 tasks, one balanced pass).

### Baseline

Vanilla GRPO (True Flow-GRPO, critic-free).

- **Budget:** PLACEHOLDER

### Proposed Method

ER-CAG (Environment-Referenced Causal Action Gain).

- **Budget:** PLACEHOLDER

### Evaluation

RoboTwin closed-loop evaluation.

- **Tasks:** 49
- **Seeds:** PLACEHOLDER

---

## 10. Recommended Execution Order

```bash
# Step 0 — environment validation
bash scripts/00_env_check.sh

# Step 1 — ER-CAG snapshots / candidates / sign probe
bash scripts/prepare_ercag_snapshots.sh

# Step 2 — multi-task SFT initialization
bash scripts/02_train_sft.sh

# Step 3 — vanilla GRPO baseline
bash scripts/04_train_vanilla.sh

# Step 4 — ER-CAG optimization
bash scripts/05_train_ercag.sh

# Step 5 — RoboTwin closed-loop evaluation
bash scripts/06_eval_all.sh
```

---

## 11. Known Issues / Notes

1. `max_steps` is not fixed yet (compute-budget placeholder).
2. The final compute budget should be selected according to the new server.
3. Smoke tests are not final paper numbers.
4. Evaluation results should be generated from production runs only.

---

## 12. Final Status

CausalWAM-Tau0 is ready for large-scale experimental execution.

- **Current blockers:** None in the engineering pipeline.
- **Remaining decisions:**
  - compute budget
  - number of seeds
  - final experimental schedule
