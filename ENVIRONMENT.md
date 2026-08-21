# Environment

## Host

- OS: Linux (7.0.0-29-generic)
- GPUs: 2 × NVIDIA H100 80GB HBM3
- Conda root: `/opt/conda`

## Conda environments

| Env | Python | PyTorch | CUDA | Used for |
|---|---|---|---|---|
| `tau0_wm` | 3.10 | 2.7.1+cu126 | 12.6 | τ0 world model, SFT, ER-CAG / Flow-GRPO RL |
| `robotwin` | 3.10 | 2.4.1+cu121 | 12.1 | RoboTwin closed-loop eval |

Both live at `/opt/conda/envs/`. Override the search root with `CONDA_ROOT` if
they are installed elsewhere.

## External dependency: RoboTwin

RoboTwin is an external manipulation benchmark and is **not** part of this repo.
It lives at `$ROBOTWIN_ROOT` (default `/data/QWW/RoboTwin`). The eval child
imports its `script/` helpers and runs its envs in the `robotwin` conda env.

## Path conventions

Two environment variables are the single source of truth:

- `CAUSALWAM_ROOT` — this project root (default = derived from `__file__`).
- `ROBOTWIN_ROOT` — external benchmark (default `/data/QWW/RoboTwin`).

Config YAML/JSON uses `${CAUSALWAM_ROOT}` / `${ROBOTWIN_ROOT}` placeholders,
expanded by `tau-0-wm/utils/config_utils.py::expand_env_vars` (SFT trainer,
`TauPolicy`, `TauSimulator`) or by `_expand()` in the eval orchestrator. Python
entry points resolve the same variables via `os.environ.get(...)`. The launch
scripts (`scripts/0*.sh`) export both before invoking Python.

## GPU / CUDA

Launch scripts auto-set `CUDA_VISIBLE_DEVICES` if unset. The τ0 VAM server runs
on `cuda:0`; the joint-RL drivers use both GPUs (policy `cuda:0`, simulator +
ValueHead `cuda:1`).

### H100 driver workaround

A post-reboot H100 `misaligned address` driver fault (driver ~580.x / CUDA 13.0)
is avoided by pinning the SDPA attention backend to `math` and using
`AdamW(foreach=False)`. This is applied in `tau-0-wm/main.py` (training) and
`_pin_sdpa()` in the RL drivers; it is an environment-compatibility fix only and
does **not** change the training recipe.

## Disk footprint

| Artifact | Size |
|---|---|
| Checkpoints (`checkpoints/`) | ~58 GB |
| Datasets (`datasets/`) | ~5.4 GB |

## Checkpoint inventory (5 KEEP entries)

| Path | Role |
|---|---|
| `checkpoints/tau0_wm/vam/` | τ0 VAM pretrained (SFT init + eval + ER-CAG backbone) |
| `checkpoints/tau0_wm/simulator/` | τ0 simulator backbone (frozen world model) |
| `checkpoints/wan2.2-ti2v-5b/` | Wan2.2-TI2V-5B backbone (VAE + T5 + model) |
| `checkpoints/theta_init_multi_v0/step_500/` | validated multi-task SFT init |
| `checkpoints/pbb2_turn_switch/step_802/` | PB-B2 turn_switch policy (ER-CAG / vanilla closed-loop) |

## Dataset inventory (2 KEEP entries)

| Path | Role |
|---|---|
| `datasets/tau0_robotwin_multitask_v0/` (49 tasks) | formal multi-task SFT training data |
| `datasets/tau0_robotwin_success_v3_lerobot/turn_switch/` | PB-B2 policy + ER-CAG statistics |
