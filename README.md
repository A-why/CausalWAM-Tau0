# CausalWAM-Tau0

**τ0 World Action Model + RoboTwin manipulation benchmark + ER-CAG**
(Environment-Referenced Causal Action Gain)

Release-prepared project for the Tau0 mainline. This tree is self-contained and
directly migratable to another server (given the two conda envs and the external
RoboTwin benchmark). The historical development source at `/data/QWW/CausalWAM`
is the read-only origin and is left completely unchanged by this release.

## What is here

| Directory | Contents |
|---|---|
| `tau-0-wm/` | Wan2.2-TI2V-5B video-diffusion world backbone (SFT trainer + VAM inference) |
| `ercag/` | ER-CAG method core (ValueHead, native future hook, official reward, losses) |
| `flow_grpo/` | True Flow-GRPO (critic-free, PPO-clipped flow loss) |
| `adapters/robotwin/` | RoboTwin ↔ τ0 observation / action adapters |
| `eval/` | evaluation drivers (closed-loop, joint-RL smoke, probes) |
| `scripts/` | launch scripts `00_env_check.sh` … `06_eval_all.sh` |
| `configs/` | training / data / runtime configs + `server_large/` throughput variants |
| `checkpoints/` | 5 KEEP checkpoints (~58 GB) |
| `datasets/` | RoboTwin multi-task SFT data (49 tasks) + turn_switch statistics |
| `docs/` | `SOURCE_REPO_AUDIT.md`, `MIGRATION_MANIFEST.md` |

## Formal experiments

1. Multi-task τ0 initialization / SFT — `theta_init_multi_v0` (49 tasks, one balanced pass)
2. RoboTwin closed-loop evaluation — 49 tasks × 3 seeds, resumable
3. Vanilla baseline — True Flow-GRPO, critic-free
4. ER-CAG method — native paired counterfactual → shared ValueHead → `G_i = Q_i − Q_0`
5. Multi-task × multi-seed final runs

## Quick start

See [QUICKSTART.md](QUICKSTART.md) (30-minute startup) and [ENVIRONMENT.md](ENVIRONMENT.md).

## Migration provenance

- [docs/SOURCE_REPO_AUDIT.md](docs/SOURCE_REPO_AUDIT.md) — read-only audit + KEEP/EXCLUDE decisions.
- [docs/MIGRATION_MANIFEST.md](docs/MIGRATION_MANIFEST.md) — source → destination record with reasons.
- [CHANGELOG.md](CHANGELOG.md) — release-preparation change log.
