# Quickstart — 30-minute startup

Goal: validate the environment and bring the stack to a ready state. Long
training / evaluation runs are intentionally **not** part of these 30 minutes.

## 0. Prerequisites (one-time)

- Two conda envs (see [ENVIRONMENT.md](ENVIRONMENT.md)):
  - `tau0_wm` — τ0 + ER-CAG / Flow-GRPO
  - `robotwin` — RoboTwin closed-loop
- RoboTwin benchmark at `$ROBOTWIN_ROOT` (default `/data/QWW/RoboTwin`).
- Checkpoints + datasets migrated (inventory in [ENVIRONMENT.md](ENVIRONMENT.md)).

## 1. Environment check (~1 min)

```bash
bash scripts/00_env_check.sh
```

Exit `0` = all critical checks pass (envs, checkpoints, datasets, configs, GPU).

## 2. Data & suite check (~1 min)

```bash
bash scripts/01_prepare_data.sh
```

Verifies the 49-task suite, per-task statistics (20-D), and data configs.

## 3. Launch the τ0 VAM server (~10–15 min first load)

```bash
bash scripts/03_launch_vam_server.sh --background
tail -f logs/03_launch_vam_server_*.log   # wait for MULTITASK_TAU_SERVER_READY
```

The server is a long-running foreground/background process that serves the
shared τ0 checkpoint with per-request normalization.

## 4. Training / evaluation (long-running, run when ready)

```bash
bash scripts/02_train_sft.sh     # multi-task SFT init (theta_init_multi_v0)
bash scripts/06_eval_all.sh      # 49-task closed-loop eval (needs the server from step 3)
```

## Environment variables

- `CAUSALWAM_ROOT` — project root (auto-detected by each script).
- `ROBOTWIN_ROOT` — RoboTwin benchmark (default `/data/QWW/RoboTwin`).
- `CUDA_VISIBLE_DEVICES` — auto-detected from `nvidia-smi` if unset.
- `CONDA_ROOT` — conda root (default `/opt/conda`).

Every script writes a timestamped log and a config snapshot under `logs/`.
