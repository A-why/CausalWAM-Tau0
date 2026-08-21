# configs/archive — provenance & historical configs

These configs were moved out of `configs/training/` and
`configs/data/robotwin_tau0/` during the production-readiness audit. They are
**kept for provenance** (not deleted) and are **not** referenced by any launcher,
script, or production config.

| Group | Files | Why archived |
|---|---|---|
| `training/pbb2_canonical_turn_switch.yaml` | PB-B2 (ER-CAG) turn_switch policy | stale resume → historical `step_700` (real policy is `checkpoints/pbb2_turn_switch/step_802`) |
| `training/pbb_turn_switch.{yaml,json}` | PB turn_switch policy | provenance of the earlier PB run; stale resume |
| `training/theta_init_multi_v0_{smoke,smoke500,smoke_fp32,smoke500_fp32}.yaml` | SFT smoke variants | smoke scale only; `smoke500_fp32` has stale resume → `step_250` |
| `training/theta_init_multi_v0_{technician_repro_100,technician_repro_500}.yaml` | technician reproduction | provenance of the migrated `step_500` init |
| `training/theta_init_multi_v0_exclusive_replay_2000.yaml` | replay variant | provenance |
| `data/robotwin_tau0/turn_switch_{abs_eef6d,pbb_abs_eef6d,v2_abs_eef6d}.yaml` | turn_switch data variants | historical; the canonical contract is `configs/data/robotwin_tau0/turn_switch_pbb_canonical_eef6d.yaml` |

**To revive any of these** (e.g. continue PB-B2 training), copy the config back
and re-point its `model_path` to the migrated checkpoint and null the
`latest_log_dir` / `optimizer_path` / `latest_global_step` resume fields.
