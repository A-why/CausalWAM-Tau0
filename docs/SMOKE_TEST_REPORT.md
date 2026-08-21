# CausalWAM-Tau0 — End-to-End Production Smoke Test Report

- **Date:** 2026-08-20
- **Scope:** Verify the full production pipeline (env → snapshots → vanilla → ER-CAG → evaluation) before moving to a large server.
- **Ground rules (applied throughout):**
  - Same production code paths only: `04_train_vanilla.sh`, `05_train_ercag.sh`, `06_eval_all.sh`, `prepare_ercag_snapshots.sh`.
  - Only reducible axes were shrunk: `max_steps`, rollout `k`, task count, eval episodes. Reward, model, action representation, optimizer logic are **unchanged**.
  - No algorithm changes, no refactoring, no `/data/QWW/CausalWAM` modification.
  - Report first error only per section; nothing was silently fixed.

---

## 1. Environment — **PASS**

| Check | Result |
|---|---|
| GPUs | 2× H100 80 GB detected |
| conda env `tau0_wm` (torch 2.7.1+cu126) | present |
| conda env `robotwin` (torch 2.4.1+cu121) | present |
| Path conventions `${CAUSALWAM_ROOT}` / `${ROBOTWIN_ROOT}` | resolve correctly |
| Import re-map (`rl.ercag.*`→`ercag.*`, `rl.flow_grpo.*`→`flow_grpo.*`, `adapters.tau0_robotwin.*`→`adapters.robotwin.*`) | working |

**GPU routing note (not a failure):** GPU0 holds ~50 GB used by an external tenant process (other container/namespace). All single-GPU steps were pinned to `CUDA_VISIBLE_DEVICES=1` (free GPU); the dual-GPU ER-CAG step used `CUDA_VISIBLE_DEVICES=1,0` (policy→GPU1, simulator+ValueHead→GPU0 with ~30 GB free).

---

## 2. Snapshot preparation — **PASS**

`prepare_ercag_snapshots.sh` completed all 5 capture stages at minimum task scope (`turn_switch`).

| Artifact | Status |
|---|---|
| `native_snapshots/S0.pkl … S3.pkl` + `manifest.json` | present (expert success=True, threshold 0.141986) |
| `tau_candidates/candidates.jsonl` | present (hash_match_all=True) |
| `tau_candidates/hold_actions.json` | present (hold Y0≈0) |
| `sign_probe_results.jsonl` | present — 4 candidate records: S0_00 fail, S0_01 fail, S0_02 success, S0_03 success (2 success / 2 fail) |

---

## 3. Vanilla smoke (`04_train_vanilla.sh`, max_steps=20) — **PASS** (1 WARN)

| Check | Result |
|---|---|
| Env init / rollout / reward / optimizer update / checkpoint write | all executed |
| Rewards (from V3-B0 sign probe) | `[0.0, 0.0, 1.0, 1.0]` |
| Group-relative advantage (vanilla GRPO) | `[-0.866, -0.866, 0.866, 0.866]` |
| 20 iterations | completed; loss 0 → −5.76e-4, grad_norm 0.64 |
| Finite all steps | `true` |
| **Determinism check** | **`false` — max_act_diff = 6.3e-2 (threshold 1e-3)** |

**WARN (first error, reported not fixed):** The vanilla driver re-samples the action and compares it to the stored candidate action; the two diverge by 6.3e-2.

**Root cause found:** `flow_grpo/tau_vanilla_grpo_production.py:221` samples with `prompt = cfg["rollout"]["prompt"]` = `"turn on the switch"`, whereas the capture pipeline that generated the stored candidates — and thus the sign-probe rewards — uses the official RoboTwin instruction `"use the robotic arm to click the switch"` via `tau_input["prompt"]` (`eval/v3b1_generate_tau_candidates.py:194`). ER-CAG uses `tau_input["prompt"]` too (`eval/r2c_joint_rl_smoke.py:297`) and achieves determinism 0.0 on the same seed/checkpoint, isolating the prompt as the cause.

**Consequence:** the vanilla baseline reads a group-relative reward that is keyed to the *stored* candidate action, while optimizing a *re-sampled* action conditioned on a different prompt. This is a reward-alignment defect in the vanilla production path, not a smoke-pipeline break. It is a genuine finding to resolve before treating vanilla production rewards as trustworthy.

---

## 4. ER-CAG smoke (`05_train_ercag.sh`, max_steps=20) — **PASS**

| Check | Result |
|---|---|
| Snapshot load / candidate load / causal-gain computation / optimizer update | all executed |
| Determinism (`act_match` maxdiff) | **0.0 → PASS** |
| Native future hook `Zhat` | `(1, 864, 3072)` |
| Value loss Lv | 0.395 → 0.195 across 20 iters |
| Policy loss Lp | −0.261 → −0.947 (decreasing) |
| Causal gain std `G_std` | 0.165 (final) |
| Gradient gates (`gates=1111`) | 4/4 pass on **all** 20 iterations |
| NaN | none |
| Finite all steps | `true` |
| History length | 20 (iters 0–19) |

---

## 5. Evaluation smoke (`06_eval_all.sh`, 1 task / 1 episode) — **PASS** (pipeline)

| Check | Result |
|---|---|
| VAM server (`theta_init_multi_v0/step_500`) | launched, `MULTITASK_TAU_SERVER_READY`, reachable at 127.0.0.1:8765 |
| RoboTwin env | built; `horizon=400`, 400 actions executed, clean teardown |
| Action adapter | valid `(33,20)` → RoboTwin action, `finite_actions=true`, 13 policy calls |
| Official reward | `official_reward` returned binary `{0,1}` (`official_reward_binary=true`) |
| Episode completion | `status=PASS` (ran to completion, no error) |
| Task success (this 1 episode) | `success=0` (reward 0.0) — θ-init policy, not a trained policy |

**Note on exit code:** `06_eval_all.sh` reports `rc=2` because the orchestrator compares completed tasks against the full 49-task suite (`tasks_complete=1 ≠ N_ready=49`). This is the expected outcome of a 1-task smoke, **not** a pipeline failure. The per-task record in `outputs/multitask_init/closed_loop_eval.json` shows a clean `status=PASS`.

---

## Final verdict

```
SMOKE_READY: YES
```

### Remaining blockers

**None (blocking).** All five production stages execute end-to-end on the existing code paths.

### Findings to resolve (non-blocking, do not fix silently in this release)

1. **Vanilla reward-alignment defect** (Section 3): `04_train_vanilla.sh` samples actions with the config literal prompt `"turn on the switch"` instead of the official instruction `"use the robotic arm to click the switch"` used by the capture pipeline. This breaks the vanilla determinism check (6.3e-2) and means the group-relative reward is read for a different action than the one optimized. ER-CAG does not have this defect (it uses `tau_input["prompt"]`).
2. **GPU0 tenant contention**: an external process holds ~50 GB on GPU0. Production runs must keep the `CUDA_VISIBLE_DEVICES` remapping (`=1` single-GPU, `=1,0` ER-CAG) unless that tenant is cleared.
