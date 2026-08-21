#!/usr/bin/env python3
"""FG-C: True Flow-GRPO 1→5→20 Stability Smoke.

Verifies that Flow-GRPO training dynamics are stable across 20 consecutive
optimizer updates. Tests stability, not performance — no success metric required.

Key requirements:
  - 20 optimizer steps with frozen FG-B config
  - 5-observation cycle (O0-O4 from demo frames 0/33/60/84/126)
  - Identity gate per step (max|ratio-1| < 1e-3 before optimizer)
  - Post-update ratio direction diagnostic
  - Checkpoints at step 1, 5, 20
  - UniPC + SDE sanity at step 1, 5, 20
  - Memory + runtime tracking per step
  - No FPO, no ER-CAG, no reference, no KL

Usage:
    python flow_grpo/tau_flow_grpo_stability.py
"""
import sys, os, json, time, math, gc
import numpy as np
import torch

# ===========================================================================
# Section 1: Paths and Constants
# ===========================================================================

CAUSAL_ROOT = os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAU_ROOT = os.path.join(CAUSAL_ROOT, "tau-0-wm")
sys.path.insert(0, TAU_ROOT)
sys.path.insert(0, CAUSAL_ROOT)
sys.path.insert(0, os.path.join(CAUSAL_ROOT, "flow_grpo"))
os.chdir(TAU_ROOT)

# Model
CHECKPOINT_PATH = os.path.join(CAUSAL_ROOT, "outputs/v0d6/turn_switch/2026_08_10_07_56_20/step_100")
VAM_STATS_PATH = os.path.join(CAUSAL_ROOT, "datasets/tau0_robotwin_v2/turn_switch/statistics.json")
ACVS_MODEL_PATH = os.path.join(CAUSAL_ROOT, "checkpoints/tau0_wm/simulator/diffusion_pytorch_model.bin")
ACVS_STATS_PATH = os.path.join(CAUSAL_ROOT, "datasets/tau0_robotwin_v2/turn_switch/statistics.json")

# Observation sources
DEMO_PATH = os.path.join(CAUSAL_ROOT, "datasets/tau0_robotwin_tau30hz_1ep/turn_switch/npz_data/episode_0.npz")
OBS_IMAGE_PATH = os.path.join(CAUSAL_ROOT, "outputs/cache/v0c_tau_input.npz")

# Output
OUT_DIR = os.path.join(CAUSAL_ROOT, "outputs/fgc_flow_grpo_stability")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "step1"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "step5"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "step20"), exist_ok=True)

# ===========================================================================
# Section 2: Hyperparameters (FROZEN from FG-B)
# ===========================================================================

K = 4                    # candidates per observation
L = 5                    # SDE flow steps
BASE_SEED = 200          # base random seed
EXECUTION_STEPS = 33     # action chunk length
ACVS_INFERENCE_STEPS = 10

# GRPO (frozen from FG-B / Wan2.1 official)
CLIP_RANGE = 1e-3        # grpo.py:56 (Wan2.1)
ADV_CLIP_MAX = 5.0       # base.py:97
BETA_KL = 0.0            # FG-C spec: no KL
LR = 1e-6                # FG-B smoke LR
MAX_GRAD_NORM = 1.0      # base.py:89
ADAM_BETAS = (0.9, 0.999)
ADAM_WEIGHT_DECAY = 1e-4
ADAM_EPS = 1e-8

# SDE (frozen from FG-B)
SDE_SHIFT = 1.0
OBS_PROMPT = "turn on the switch"

# Observation cycle: frames from seed100 demo covering early→approach→pre-contact→contact→later
OBS_FRAMES = [0, 33, 60, 84, 126]  # tau30hz_1ep/episode_0 frame indices
OBS_IDS = ["O0_early", "O1_approach", "O2_precontact", "O3_contact", "O4_later"]

# Identity threshold
IDENTITY_THRESHOLD = 1e-3  # max|ratio-1| before abort

N_STEPS = 20

print("=" * 80)
print("FG-C: True Flow-GRPO 1→5→20 Stability Smoke")
print("=" * 80)
print(f"  Steps: {N_STEPS}, K={K}, L={L}")
print(f"  CLIP_RANGE={CLIP_RANGE}, LR={LR}, BETA_KL={BETA_KL}")
print(f"  Observations: {OBS_IDS} at frames {OBS_FRAMES}")
print(f"  Checkpoint: {CHECKPOINT_PATH}")
print(f"  Output: {OUT_DIR}")
print(f"  Identity threshold: |ratio-1| < {IDENTITY_THRESHOLD}")
sys.stdout.flush()

# ===========================================================================
# Section 3: Load VAM (FRESH V0-D6 step100)
# ===========================================================================

print("\n" + "=" * 60)
print("Section 3: Loading VAM (fresh V0-D6 step100)")
print("=" * 60)
sys.stdout.flush()

import utils.model_utils
utils.model_utils.forward_pass = lambda *a, **kw: None
from models.wan_2_2_models.transformers.attention import set_attention_backend
set_attention_backend(attention_impl='sdpa')

from yaml import load, Loader, Dumper, dump
from web_infer_utils.TauPolicy import TauPolicy

vam_cfg = load(open(os.path.join(CAUSAL_ROOT, 'configs/runtime/vam_deploy.yaml')), Loader=Loader)
vam_cfg['diffusion_model']['model_path'] = CHECKPOINT_PATH
vam_cfg['statistics_file'] = VAM_STATS_PATH
vam_cfg['seed'] = BASE_SEED

with open('/tmp/fgc_vam.yaml', 'w') as f:
    dump(vam_cfg, f, Dumper=Dumper)

vam = TauPolicy(
    config_file='/tmp/fgc_vam.yaml', device=torch.device('cuda:0'), rank=0,
    compile_model=False, attention_impl='sdpa',
    enable_self_attn_fused_qkv=True, enable_context_null_cache=True,
)
vam.diffusion_model.eval()
print(f"  VAM loaded — GPU 0: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
sys.stdout.flush()

# ===========================================================================
# Section 4: Load ACVS
# ===========================================================================

print("\n" + "=" * 60)
print("Section 4: Loading ACVS")
print("=" * 60)
sys.stdout.flush()

from web_infer_utils.simulator.TauSimulator import TauSimulator

acvs_cfg = load(open(os.path.join(CAUSAL_ROOT, 'configs/runtime/acvs_deploy.yaml')), Loader=Loader)
acvs_cfg['diffusion_model']['model_path'] = ACVS_MODEL_PATH
acvs_cfg['statistics_file'] = ACVS_STATS_PATH

with open('/tmp/fgc_acvs.yaml', 'w') as f:
    dump(acvs_cfg, f, Dumper=Dumper)

acvs = TauSimulator(config_file='/tmp/fgc_acvs.yaml', device=torch.device('cuda:1'), rank=1)
print(f"  ACVS loaded — GPU 1: {torch.cuda.memory_allocated(1)/1e9:.2f} GB")
sys.stdout.flush()

# ===========================================================================
# Section 5: Build Observation Pool
# ===========================================================================

print("\n" + "=" * 60)
print("Section 5: Building observation pool")
print("=" * 60)
sys.stdout.flush()

from adapters.robotwin.rotation_utils import (
    tau_6d_to_robotwin_quat, reorder_quaternion, quaternion_to_rotation_6d
)

demo = np.load(DEMO_PATH, allow_pickle=True)
demo_states = demo['states']  # (168, 20) float32, converter layout

def build_state_14d(state_phys_20d):
    """Converter 20D → VAM state (14D xyzw + 2D gripper)."""
    left_xyz = state_phys_20d[0:3]
    left_6d = state_phys_20d[3:9]
    right_xyz = state_phys_20d[9:12]
    right_6d = state_phys_20d[12:18]
    left_grip = state_phys_20d[18]
    right_grip = state_phys_20d[19]
    left_quat_wxyz = tau_6d_to_robotwin_quat(left_6d)
    right_quat_wxyz = tau_6d_to_robotwin_quat(right_6d)
    left_quat_xyzw = reorder_quaternion(left_quat_wxyz, 'wxyz', 'xyzw')
    right_quat_xyzw = reorder_quaternion(right_quat_wxyz, 'wxyz', 'xyzw')
    state_14d = np.concatenate([left_xyz, left_quat_xyzw, right_xyz, right_quat_xyzw])
    grip_2d = np.array([left_grip, right_grip])
    return state_14d.astype(np.float64), grip_2d.astype(np.float64)

# Load shared observation image
if os.path.exists(OBS_IMAGE_PATH):
    obs_img_shared = np.load(OBS_IMAGE_PATH, allow_pickle=True)['obs']
    print(f"  Using cached real observation image: shape={obs_img_shared.shape}")
else:
    obs_img_shared = np.zeros((3, 3, 192, 256), dtype=np.float32)
    print(f"  WARNING: No cached image, using dummy zeros")

obs_pool = []
for i, frame_idx in enumerate(OBS_FRAMES):
    state_14d, grip_2d = build_state_14d(demo_states[frame_idx])
    obs_pool.append({
        'obs_id': OBS_IDS[i],
        'frame': frame_idx,
        'state_14d': state_14d,
        'grip_2d': grip_2d,
        'obs_img': obs_img_shared,
    })
    print(f"  {OBS_IDS[i]} (frame {frame_idx}): "
          f"pos_l=[{state_14d[0]:.3f},{state_14d[1]:.3f},{state_14d[2]:.3f}], "
          f"pos_r=[{state_14d[7]:.3f},{state_14d[8]:.3f},{state_14d[9]:.3f}], "
          f"grip=[{grip_2d[0]:.0f},{grip_2d[1]:.0f}]")

sys.stdout.flush()

# ===========================================================================
# Section 6: Pipeline Wrapper
# ===========================================================================

from flow_grpo.tau_pipeline_with_logprob import TauPipelineWithLogprob, sample_k_trajectories
from flow_grpo.tau_flow_grpo_buffer import TauTrajectory, TauTrajectoryGroup, build_trajectory_from_sde_result
from flow_grpo.tau_flow_grpo_loss import compute_grpo_loss

pipeline_wrapper = TauPipelineWithLogprob(vam)

# ===========================================================================
# Section 7: Pre-training Setup
# ===========================================================================

# Trainable params
trainable_params = []
for name, param in vam.diffusion_model.named_parameters():
    if 'action_' in name or (name.find('action_') < 0 and name.find('vlm_interface') < 0):
        trainable_params.append(param)
        param.requires_grad = True
    else:
        param.requires_grad = False

n_trainable = sum(p.numel() for p in trainable_params)
n_total = sum(p.numel() for p in vam.diffusion_model.parameters())
print(f"\n  Trainable params: {n_trainable:,} / {n_total:,}")

# Base parameter snapshot (fp32 on CPU for drift tracking)
param_base_cpu = {}
for name, param in vam.diffusion_model.named_parameters():
    if param.requires_grad:
        param_base_cpu[name] = param.detach().float().cpu().clone()

# Optimizer
optimizer = torch.optim.AdamW(
    trainable_params, lr=LR, betas=ADAM_BETAS,
    weight_decay=ADAM_WEIGHT_DECAY, eps=ADAM_EPS,
)

# Previous step params for incremental delta
param_prev_cpu = {k: v.clone() for k, v in param_base_cpu.items()}

# Metrics log
metrics_path = os.path.join(OUT_DIR, 'metrics.jsonl')
with open(metrics_path, 'w') as f:
    f.write("# FG-C metrics.jsonl — one JSON object per line\n")

# ===========================================================================
# Section 8: Distributed communication stubs (prepared for DDP)
# ===========================================================================

def _get_world_size():
    """Return world_size if distributed is initialized, else 1."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1

def _get_rank():
    """Return rank if distributed is initialized, else 0."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0

def all_gather_scalars(local_values, device='cuda:0'):
    """Gather scalar values across ranks. Returns list of all values."""
    world_size = _get_world_size()
    if world_size == 1:
        return list(local_values) if isinstance(local_values, (list, tuple)) else [local_values]
    # DDP path (future)
    tensor = torch.tensor(local_values, device=device)
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, tensor)
    return torch.cat(gathered).tolist()

# ============================================================================
# Section 9: Training Loop (20 iterations)
# ============================================================================

print("\n" + "=" * 80)
print(f"FG-C: Starting {N_STEPS}-step training loop")
print("=" * 80)
sys.stdout.flush()

all_step_metrics = []
identity_failures = 0
degenerate_groups = 0
step_times = []

for step in range(1, N_STEPS + 1):
    t_step_start = time.monotonic()
    step_metrics = {'step': step}

    # --- A. Select observation ---
    obs_idx = (step - 1) % 5
    obs = obs_pool[obs_idx]
    step_metrics['obs_id'] = obs['obs_id']
    step_metrics['obs_frame'] = obs['frame']

    print(f"\n{'─'*60}")
    print(f"Step {step}/{N_STEPS} | Obs: {obs['obs_id']} (frame {obs['frame']})")
    print(f"{'─'*60}")
    sys.stdout.flush()

    # --- B. Sample K=4 SDE trajectories ---
    # NOTE: _build_conditioning() is called internally by sample_with_logprob()
    # on first call. No explicit pre-call needed.
    t_sde = time.monotonic()
    sde_results = sample_k_trajectories(
        pipeline_wrapper=pipeline_wrapper,
        state_14d=obs['state_14d'],
        gripper_states=obs['grip_2d'],
        obs_img=obs['obs_img'],
        prompt=OBS_PROMPT,
        k=K, base_seed=BASE_SEED,
        num_inference_steps=L,
        execution_steps=EXECUTION_STEPS,
        return_velocities=True,
    )
    t_sde_elapsed = time.monotonic() - t_sde
    step_metrics['runtime_sde'] = t_sde_elapsed

    trajectories = []
    for i, result in enumerate(sde_results):
        traj = build_trajectory_from_sde_result(
            result=result, state_14d=obs['state_14d'],
            gripper_states=obs['grip_2d'], prompt=OBS_PROMPT,
        )
        traj.k_idx = i; traj.seed = result['seed']
        trajectories.append(traj)

    # --- D. True model-recomputation identity check ---
    identity_ratios = []
    identity_vdiffs = []
    for traj in trajectories:
        with torch.no_grad():
            rlp, rv = pipeline_wrapper.recompute_trajectory_logprobs(traj, enable_grad=False)
        log_ratio = rlp - traj.log_probs
        ratio = torch.exp(torch.clamp(log_ratio, min=-20, max=20))
        identity_ratios.extend(ratio.tolist())
        vdiff = (rv - traj.velocities).abs().max().item()
        identity_vdiffs.append(vdiff)

    id_ratio_arr = np.array(identity_ratios)
    id_max_error = np.abs(id_ratio_arr - 1.0).max()
    id_vd_max = max(identity_vdiffs)
    step_metrics['identity_ratio_error'] = float(id_max_error)
    step_metrics['identity_velocity_diff'] = float(id_vd_max)

    print(f"  Identity: max|ratio-1|={id_max_error:.8f}, max|v_diff|={id_vd_max:.8f}")

    if id_max_error > IDENTITY_THRESHOLD:
        print(f"  ⛔ IDENTITY FAIL at step {step}! max|ratio-1|={id_max_error:.6f} > {IDENTITY_THRESHOLD}")
        identity_failures += 1
        # Diagnostic dump
        for i, traj in enumerate(trajectories):
            for l in range(L):
                with torch.no_grad():
                    rlp_i, _ = pipeline_wrapper.recompute_logprob(
                        traj.latents[l], traj.next_latents[l], traj.timesteps[l], l, enable_grad=False)
                ratio_i = torch.exp(torch.clamp(rlp_i - traj.log_probs[l], min=-20, max=20)).item()
                print(f"    k={i} step={l}: ratio={ratio_i:.6f}")
        sys.exit(1)

    # --- E. ACVS scoring ---
    t_acvs = time.monotonic()
    all_Q = []
    resample_attempts = 0

    for k_idx, traj in enumerate(trajectories):
        final_action_phys = pipeline_wrapper.denormalize_action(traj.latents[-1])
        acvs.reset()
        with torch.inference_mode():
            _, reward_k = acvs.play(
                obs=obs['obs_img'], prompt=OBS_PROMPT,
                actions=final_action_phys.cpu().numpy()[:EXECUTION_STEPS],
                num_inference_steps=ACVS_INFERENCE_STEPS,
                execution_step=EXECUTION_STEPS, n_mem=3,
            )
        Q_k = float(np.max(reward_k))
        all_Q.append(Q_k)
        traj.reward = Q_k

    Qs = np.array(all_Q)
    Q_std = Qs.std()

    # Resample if degenerate (up to 2 attempts)
    while Q_std < 1e-8 and resample_attempts < 2:
        print(f"  ACVS Q degenerate (std={Q_std:.2e}), resampling (attempt {resample_attempts+1})...")
        resample_attempts += 1
        new_seed_offset = BASE_SEED * 1000 + 100 * step + resample_attempts * 10
        sde_results = sample_k_trajectories(
            pipeline_wrapper=pipeline_wrapper,
            state_14d=obs['state_14d'], gripper_states=obs['grip_2d'],
            obs_img=obs['obs_img'], prompt=OBS_PROMPT,
            k=K, base_seed=new_seed_offset,
            num_inference_steps=L, execution_steps=EXECUTION_STEPS,
            return_velocities=True,
        )
        trajectories = []
        for i, result in enumerate(sde_results):
            traj = build_trajectory_from_sde_result(
                result=result, state_14d=obs['state_14d'],
                gripper_states=obs['grip_2d'], prompt=OBS_PROMPT,
            )
            traj.k_idx = i; traj.seed = result['seed']
            trajectories.append(traj)
        all_Q = []
        for traj in trajectories:
            final_action_phys = pipeline_wrapper.denormalize_action(traj.latents[-1])
            acvs.reset()
            with torch.inference_mode():
                _, reward_k = acvs.play(
                    obs=obs['obs_img'], prompt=OBS_PROMPT,
                    actions=final_action_phys.cpu().numpy()[:EXECUTION_STEPS],
                    num_inference_steps=ACVS_INFERENCE_STEPS,
                    execution_step=EXECUTION_STEPS, n_mem=3,
                )
            Q_k = float(np.max(reward_k))
            all_Q.append(Q_k)
            traj.reward = Q_k
        Qs = np.array(all_Q)
        Q_std = Qs.std()

    t_acvs_elapsed = time.monotonic() - t_acvs
    step_metrics['runtime_acvs'] = t_acvs_elapsed

    if Q_std < 1e-8:
        degenerate_groups += 1
        print(f"  ⚠ DEGENERATE GROUP (Q_std={Q_std:.2e}) after {resample_attempts} resamples — skipping update")
        step_metrics['degenerate'] = True
        step_metrics['Q'] = all_Q
        step_metrics['Q_mean'] = float(Qs.mean())
        step_metrics['Q_std'] = float(Q_std)
        step_metrics['skipped'] = True
        all_step_metrics.append(step_metrics)
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(step_metrics) + '\n')
        continue

    step_metrics['degenerate'] = False
    step_metrics['skipped'] = False
    step_metrics['Q'] = [float(q) for q in Qs]
    step_metrics['Q_mean'] = float(Qs.mean())
    step_metrics['Q_std'] = float(Q_std)
    step_metrics['Q_range'] = float(Qs.max() - Qs.min())
    step_metrics['resample_attempts'] = resample_attempts

    print(f"  ACVS: Q={[f'{q:.4f}' for q in Qs]}, mean={Qs.mean():.4f}, std={Q_std:.4f}")

    # --- F. Compute advantages ---
    group = TauTrajectoryGroup(
        group_id=f"fgc_step{step}", state_14d=obs['state_14d'],
        gripper_states=obs['grip_2d'], trajectories=trajectories,
    )
    group.compute_advantages(eps=1e-6)
    advantages = np.array([t.advantage for t in trajectories])
    step_metrics['A'] = [float(a) for a in advantages]
    step_metrics['A_mean'] = float(advantages.mean())
    step_metrics['A_std'] = float(advantages.std())
    step_metrics['A_pos'] = int(sum(advantages > 0))
    step_metrics['A_neg'] = int(sum(advantages < 0))
    step_metrics['A_absmax'] = float(np.abs(advantages).max())

    print(f"  Advantages: A={[f'{a:+.4f}' for a in advantages]}, pos={sum(advantages>0)}, neg={sum(advantages<0)}")

    # --- G. GRPO training ---
    t_train = time.monotonic()

    # Recomputation with grad
    t_recomp = time.monotonic()
    all_cur_logps = []
    all_old_logps = []
    all_advantages_t = []

    for k_idx, traj in enumerate(trajectories):
        A_k = torch.tensor(traj.advantage, device='cuda:0', dtype=torch.float32)
        cur_log_probs, _ = pipeline_wrapper.recompute_trajectory_logprobs(traj, enable_grad=True)
        for l in range(L):
            all_cur_logps.append(cur_log_probs[l])
            all_old_logps.append(traj.log_probs[l].detach())
            all_advantages_t.append(A_k)

    t_recomp_elapsed = time.monotonic() - t_recomp
    step_metrics['runtime_recomp'] = t_recomp_elapsed

    # Batched GRPO loss
    log_prob_current = torch.stack(all_cur_logps).unsqueeze(1)  # (K*L, 1)
    log_prob_old = torch.stack(all_old_logps).unsqueeze(1)
    advantages_batch = torch.stack(all_advantages_t)

    loss_dict = compute_grpo_loss(
        log_prob_current=log_prob_current, log_prob_old=log_prob_old,
        advantages=advantages_batch, clip_range=CLIP_RANGE,
        adv_clip_max=ADV_CLIP_MAX, beta_kl=BETA_KL,
    )

    total_loss = loss_dict['loss']
    step_metrics['policy_loss'] = float(total_loss.item())
    step_metrics['clip_fraction'] = float(loss_dict['clipfrac'].item())

    # Backward
    t_backward = time.monotonic()
    optimizer.zero_grad()
    total_loss.backward()
    grad_norm_preclip = torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
    grad_norm_postclip = min(float(grad_norm_preclip), MAX_GRAD_NORM)
    optimizer.step()
    t_backward_elapsed = time.monotonic() - t_backward
    step_metrics['runtime_backward'] = t_backward_elapsed
    step_metrics['runtime_optimizer'] = t_backward_elapsed  # combined

    # Gradient diagnostics
    none_grad = sum(1 for p in trainable_params if p.grad is None)
    nan_grad = sum(1 for p in trainable_params if p.grad is not None and not torch.all(torch.isfinite(p.grad)))
    nonzero_grad = sum(1 for p in trainable_params if p.grad is not None and p.grad.norm() > 0)
    step_metrics['grad_preclip'] = float(grad_norm_preclip)
    step_metrics['grad_postclip'] = float(grad_norm_postclip)
    step_metrics['grad_none'] = none_grad
    step_metrics['grad_nan'] = nan_grad
    step_metrics['grad_nonzero'] = nonzero_grad

    print(f"  Training: loss={total_loss.item():.6f}, grad=[{grad_norm_preclip:.3f}→{grad_norm_postclip:.3f}], "
          f"clipfrac={loss_dict['clipfrac'].item():.3f}, nonzero={nonzero_grad}/{len(trainable_params)}")

    # NaN/Inf check
    if nan_grad > 0 or not torch.isfinite(total_loss):
        print(f"  ⛔ NaN/Inf detected at step {step}! Aborting.")
        sys.exit(1)

    # --- H. Parameter drift ---
    param_delta_base_sq = 0.0
    param_delta_incr_sq = 0.0
    n_params_float = 0
    for name, param in vam.diffusion_model.named_parameters():
        if name in param_base_cpu:
            p_float = param.detach().float().cpu()
            param_delta_base_sq += (p_float - param_base_cpu[name]).pow(2).sum().item()
            if name in param_prev_cpu:
                param_delta_incr_sq += (p_float - param_prev_cpu[name]).pow(2).sum().item()
            param_prev_cpu[name] = p_float.clone()
            n_params_float += p_float.numel()

    delta_base = math.sqrt(max(param_delta_base_sq, 0))
    delta_incr = math.sqrt(max(param_delta_incr_sq, 0))
    rel_delta_base = delta_base / math.sqrt(sum(p.pow(2).sum().item() for p in param_base_cpu.values()))
    step_metrics['param_delta_base'] = float(delta_base)
    step_metrics['param_delta_incremental'] = float(delta_incr)
    step_metrics['param_delta_base_rel'] = float(rel_delta_base)

    print(f"  Param drift: base Δ={delta_base:.6e} (rel={rel_delta_base:.6e}), incr Δ={delta_incr:.6e}")

    # --- I. Post-update ratio on fixed trajectories ---
    vam.diffusion_model.eval()
    post_log_ratios_pos = []
    post_log_ratios_neg = []
    post_all_ratios = []
    post_all_log_ratios = []
    clip_count = 0 ; total_count = 0

    for k_idx, traj in enumerate(trajectories):
        adv = traj.advantage
        with torch.no_grad():
            plp, _ = pipeline_wrapper.recompute_trajectory_logprobs(traj, enable_grad=False)
        for l in range(L):
            lr_val = (plp[l] - traj.log_probs[l]).item()
            r_val = math.exp(min(lr_val, 20))
            post_all_ratios.append(r_val)
            post_all_log_ratios.append(lr_val)
            if adv > 0.01:
                post_log_ratios_pos.append(lr_val)
            elif adv < -0.01:
                post_log_ratios_neg.append(lr_val)
            total_count += 1
            if abs(r_val - 1.0) > CLIP_RANGE:
                clip_count += 1

    post_ratios_arr = np.array(post_all_ratios)
    post_log_ratios_arr = np.array(post_all_log_ratios)
    direction_gap = (np.mean(post_log_ratios_pos) if post_log_ratios_pos else 0.0) - \
                    (np.mean(post_log_ratios_neg) if post_log_ratios_neg else 0.0)

    step_metrics['post_ratio_mean'] = float(post_ratios_arr.mean())
    step_metrics['post_ratio_std'] = float(post_ratios_arr.std())
    step_metrics['post_ratio_min'] = float(post_ratios_arr.min())
    step_metrics['post_ratio_max'] = float(post_ratios_arr.max())
    step_metrics['post_log_ratio_mean'] = float(post_log_ratios_arr.mean())
    step_metrics['post_log_ratio_std'] = float(post_log_ratios_arr.std())
    step_metrics['direction_gap'] = float(direction_gap)
    step_metrics['clip_fraction_post'] = float(clip_count / total_count if total_count else 0)

    print(f"  Post-update: ratio=[{post_ratios_arr.min():.4f},{post_ratios_arr.max():.4f}], "
          f"direction_gap={direction_gap:+.6f}, clip={clip_count}/{total_count}")

    # Ratio instability check
    if post_ratios_arr.min() < 1e-6 or post_ratios_arr.max() > 10.0 or not np.all(np.isfinite(post_ratios_arr)):
        print(f"  ⛔ RATIO INSTABILITY: ratio range [{post_ratios_arr.min():.6f}, {post_ratios_arr.max():.6f}]")
        sys.exit(1)

    # --- J. Action diagnostics ---
    final_actions = torch.stack([t.latents[-1] for t in trajectories])  # (K, 33, 20)
    step_metrics['action_norm_mean'] = float(final_actions.mean())
    step_metrics['action_norm_std'] = float(final_actions.std())
    step_metrics['action_norm_absmax'] = float(final_actions.abs().max())

    # Pairwise diversity
    pairwise_dists = []
    for i in range(K):
        for j in range(i+1, K):
            d = (final_actions[i] - final_actions[j]).pow(2).sum().sqrt().item()
            pairwise_dists.append(d)
    pw_arr = np.array(pairwise_dists)
    step_metrics['candidate_diversity_min'] = float(pw_arr.min())
    step_metrics['candidate_diversity_mean'] = float(pw_arr.mean())
    step_metrics['candidate_diversity_max'] = float(pw_arr.max())

    print(f"  Actions: norm=[{final_actions.mean():.3f}±{final_actions.std():.3f}], "
          f"diversity=[{pw_arr.min():.2f},{pw_arr.max():.2f}]")

    # --- K. Memory ---
    step_metrics['gpu0_allocated'] = float(torch.cuda.memory_allocated(0) / 1e9)
    step_metrics['gpu0_reserved'] = float(torch.cuda.memory_reserved(0) / 1e9)
    step_metrics['gpu1_allocated'] = float(torch.cuda.memory_allocated(1) / 1e9)
    step_metrics['gpu1_reserved'] = float(torch.cuda.memory_reserved(1) / 1e9)

    # --- L. Release buffers ---
    del trajectories, sde_results, all_cur_logps, all_old_logps, all_advantages_t
    del log_prob_current, log_prob_old, advantages_batch
    torch.cuda.empty_cache()
    gc.collect()

    # --- M. Runtime ---
    t_step_total = time.monotonic() - t_step_start
    step_metrics['runtime_step'] = float(t_step_total)
    step_times.append(t_step_total)

    # --- N. Write metrics ---
    all_step_metrics.append(step_metrics)
    with open(metrics_path, 'a') as f:
        f.write(json.dumps(step_metrics) + '\n')

    print(f"  Step {step} done in {t_step_total:.1f}s "
          f"(SDE={t_sde_elapsed:.1f}s, ACVS={t_acvs_elapsed:.1f}s, "
          f"recomp={t_recomp_elapsed:.1f}s, backward={t_backward_elapsed:.1f}s)")

    # --- Checkpoints ---
    if step == 1:
        ckpt_path = os.path.join(OUT_DIR, f"step{step}")
        torch.save({
            'model': vam.diffusion_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'note': 'FLOW-GRPO STABILITY SMOKE — NOT MAIN RESULT',
        }, os.path.join(ckpt_path, 'checkpoint.pt'))
        print(f"  💾 Checkpoint saved: step{step}/")
    elif step == 5:
        ckpt_path = os.path.join(OUT_DIR, f"step{step}")
        torch.save({
            'model': vam.diffusion_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'note': 'FLOW-GRPO STABILITY SMOKE — NOT MAIN RESULT',
        }, os.path.join(ckpt_path, 'checkpoint.pt'))
        print(f"  💾 Checkpoint saved: step{step}/")

    # UniPC + SDE sanity at steps 1, 5, 20
    if step in [1, 5, 20]:
        print(f"\n  --- Sanity checkpoints at step {step} ---")

        # UniPC
        with torch.no_grad():
            unipc_action = pipeline_wrapper.sample_unipc(
                state_14d=obs_pool[0]['state_14d'], gripper_states=obs_pool[0]['grip_2d'],
                obs_img=obs_pool[0]['obs_img'], prompt=OBS_PROMPT,
                num_inference_steps=L, execution_steps=EXECUTION_STEPS,
                seed=BASE_SEED * 10000 + step, shift=SDE_SHIFT,
            )
        unipc_ok = torch.isfinite(unipc_action).all().item() and unipc_action.shape == (33, 20)
        print(f"  UniPC: shape={tuple(unipc_action.shape)}, finite={unipc_ok}, "
              f"range=[{unipc_action.min():.3f},{unipc_action.max():.3f}]")

        # SDE
        gen_sde = torch.Generator(device='cuda:0')
        gen_sde.manual_seed(BASE_SEED * 10000 + step + 1)
        with torch.no_grad():
            sde_test = pipeline_wrapper.sample_with_logprob(
                state_14d=obs_pool[0]['state_14d'], gripper_states=obs_pool[0]['grip_2d'],
                obs_img=obs_pool[0]['obs_img'], prompt=OBS_PROMPT,
                num_inference_steps=L, execution_steps=EXECUTION_STEPS,
                seed=BASE_SEED * 10000 + step + 1, generator=gen_sde,
            )
        sde_ok = torch.isfinite(sde_test['action']).all().item()
        print(f"  SDE:   shape={tuple(sde_test['action'].shape)}, finite={sde_ok}, "
              f"range=[{sde_test['action'].min():.3f},{sde_test['action'].max():.3f}]")

    sys.stdout.flush()

# ===========================================================================
# Section 10: Final Checkpoint (step 20)
# ===========================================================================

ckpt_path = os.path.join(OUT_DIR, "step20")
torch.save({
    'model': vam.diffusion_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'step': 20,
    'note': 'FLOW-GRPO STABILITY SMOKE — NOT MAIN RESULT',
}, os.path.join(ckpt_path, 'checkpoint.pt'))
print(f"\n💾 Final checkpoint saved: step20/")

# ===========================================================================
# Section 11: Final Report
# ===========================================================================

print("\n" + "=" * 80)
print("FG-C: True Flow-GRPO 1→5→20 Stability Smoke — FINAL REPORT")
print("=" * 80)

# Aggregate metrics
completed_steps = sum(1 for m in all_step_metrics if not m.get('skipped', False))
total_skipped = sum(1 for m in all_step_metrics if m.get('skipped', False))

post_ratios_all = []
direction_gaps = []
grad_preclips = []
for m in all_step_metrics:
    if not m.get('skipped') and 'post_ratio_mean' in m:
        post_ratios_all.append(m['post_ratio_mean'])
        direction_gaps.append(m['direction_gap'])
        grad_preclips.append(m['grad_preclip'])

positive_direction_steps = sum(1 for d in direction_gaps if d > 0)

# Runtime
total_runtime = sum(step_times)
mean_step_time = np.mean(step_times) if step_times else 0

# Memory
gpu0_peak = max((m.get('gpu0_allocated', 0) for m in all_step_metrics), default=0)
gpu1_peak = max((m.get('gpu1_allocated', 0) for m in all_step_metrics), default=0)
gpu0_step2 = all_step_metrics[1].get('gpu0_allocated', 0) if len(all_step_metrics) > 1 else 0
gpu0_last = all_step_metrics[-1].get('gpu0_allocated', 0) if all_step_metrics else 0
mem_growth = gpu0_last - gpu0_step2

# Parameter drift
final_delta_base = all_step_metrics[-1].get('param_delta_base', 0) if all_step_metrics else 0
final_delta_rel = all_step_metrics[-1].get('param_delta_base_rel', 0) if all_step_metrics else 0

overall_pass = (identity_failures == 0 and completed_steps == N_STEPS and
                (not post_ratios_all or (min(post_ratios_all) > 0 and max(post_ratios_all) < 10)))

print(f"""
  Experiment: FG-C True τ₀ Flow-GRPO 1→5→20 Stability Smoke
  Base: V0-D6 step100
  Date: 2026-08-12

  --- Flow-GRPO Config ---
  K: 4
  L: 5
  SDE: plain sigma-interpolation
  clip_range: {CLIP_RANGE} (Wan2.1 grpo.py:56)
  optimizer: AdamW(lr={LR}, betas={ADAM_BETAS})
  KL: 0

  --- Steps ---
  completed: {completed_steps}/{N_STEPS}
  skipped (degenerate): {total_skipped}

  --- Identity across training ---
  failed identity steps: {identity_failures}/{N_STEPS}
  max|ratio-1| threshold: {IDENTITY_THRESHOLD}

  --- Gradient ---
  preclip: min={min(grad_preclips):.3f}, max={max(grad_preclips):.3f}, mean={np.mean(grad_preclips):.3f}
  NaN/Inf: {'YES' if any(m.get('grad_nan',0)>0 for m in all_step_metrics) else '0'}

  --- Parameter drift ---
  step1: {all_step_metrics[0].get('param_delta_base', 0) if all_step_metrics else 0:.6e}
  step5: {all_step_metrics[4].get('param_delta_base', 0) if len(all_step_metrics) > 4 else 0:.6e}
  step20: {final_delta_base:.6e}
  relative: {final_delta_rel:.6e}

  --- Post-update ratio ---
  range: [{min(post_ratios_all):.6f}, {max(post_ratios_all):.6f}]
  clip fraction range: [{min(m.get('clip_fraction_post',0) for m in all_step_metrics if 'clip_fraction_post' in m):.3f},
                         {max(m.get('clip_fraction_post',0) for m in all_step_metrics if 'clip_fraction_post' in m):.3f}]

  --- Direction ---
  positive-gap steps: {positive_direction_steps}/{len(direction_gaps)}
  mean: {np.mean(direction_gaps):.6f}
  median: {np.median(direction_gaps):.6f}
  range: [{min(direction_gaps):.6f}, {max(direction_gaps):.6f}]

  --- Runtime ---
  mean step: {mean_step_time:.1f}s
  total: {total_runtime:.1f}s

  --- Memory ---
  GPU0 peak: {gpu0_peak:.1f} GB
  GPU1 peak: {gpu1_peak:.1f} GB
  GPU0 growth (step2→20): {mem_growth:+.2f} GB
  leak: {'YES' if mem_growth > 2.0 else 'NO'}

  --- Formal optimizer ---
  TRUE FLOW-GRPO: {'STABLE' if overall_pass else 'UNSTABLE'}
  FPO used: NO
  ER-CAG used: NO
  Reference used: NO

  --- V3 status ---
  universal floor: UNCHANGED

  --- FG-C ---
  {'PASS' if overall_pass else 'PARTIAL/FAIL'}

  --- Next stage ---
  if PASS: Productive Baseline Repair
  — obtain successful executable turn_switch trajectory/policy
  under current RoboTwin physics
""")

# Save summary
summary_path = os.path.join(OUT_DIR, 'fgc_summary.json')
with open(summary_path, 'w') as f:
    json.dump({
        'experiment': 'FG-C True Flow-GRPO 1→5→20 Stability Smoke',
        'verdict': 'PASS' if overall_pass else 'PARTIAL',
        'completed_steps': completed_steps,
        'total_steps': N_STEPS,
        'identity_failures': identity_failures,
        'degenerate_groups': degenerate_groups,
        'positive_direction_steps': positive_direction_steps,
        'total_direction_steps': len(direction_gaps),
        'mean_direction_gap': float(np.mean(direction_gaps)) if direction_gaps else 0,
        'param_drift_base_final': float(final_delta_base),
        'param_drift_rel_final': float(final_delta_rel),
        'gpu0_peak_gb': float(gpu0_peak),
        'gpu1_peak_gb': float(gpu1_peak),
        'memory_leak': bool(mem_growth > 2.0),
        'total_runtime_s': float(total_runtime),
        'fpo_used': False,
        'er_cag_used': False,
        'reference_used': False,
    }, f, indent=2)

print(f"\nSummary saved: {summary_path}")
print(f"Metrics saved: {metrics_path} ({completed_steps} lines)")
print(f"Outputs: {os.listdir(OUT_DIR)}")
print("\nFG-C Complete — STOP")
print("=" * 80)
