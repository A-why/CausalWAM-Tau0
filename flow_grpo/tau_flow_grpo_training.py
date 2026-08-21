#!/usr/bin/env python3
"""FG-B: True Flow-GRPO Real-ACVS One-Step Update.

Single-file training script implementing the FG-B specification:
- K=4 SDE rollouts with FG-A pipeline wrapper
- Real ACVS rewards (no artificial advantages)
- True model-recomputation identity check (hard gate)
- One optimizer.step with PPO-clipped GRPO loss
- Post-update ratio direction diagnostics
- Output to outputs/fgb_flow_grpo_step1/

Usage:
    python flow_grpo/tau_flow_grpo_training.py
"""
import sys, os, json, time, math
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
os.chdir(TAU_ROOT)  # Required: config paths are relative to tau-0-wm

# Model checkpoint
CHECKPOINT_PATH = os.path.join(CAUSAL_ROOT, "outputs/v0d6/turn_switch/2026_08_10_07_56_20/step_100")
VAM_STATS_PATH = os.path.join(CAUSAL_ROOT, "datasets/tau0_robotwin_v2/turn_switch/statistics.json")
ACVS_MODEL_PATH = os.path.join(CAUSAL_ROOT, "checkpoints/tau0_wm/simulator/diffusion_pytorch_model.bin")
ACVS_STATS_PATH = os.path.join(CAUSAL_ROOT, "datasets/tau0_robotwin_v2/turn_switch/statistics.json")

# Observation source (seed100 proxy)
OBS_CACHE = os.path.join(CAUSAL_ROOT, "outputs/cache/v0c_tau_input.npz")
RESET_CACHE = os.path.join(CAUSAL_ROOT, "outputs/cache/v0c_robotwin_reset.npz")

# Output directory
OUT_DIR = os.path.join(CAUSAL_ROOT, "outputs/fgb_flow_grpo_step1")
os.makedirs(OUT_DIR, exist_ok=True)

# ===========================================================================
# Section 2: Hyperparameters (sourced from official Wan2.1 config)
# ===========================================================================

K = 4                    # candidates per observation
L = 5                    # SDE flow steps
BASE_SEED = 200          # base random seed
EXECUTION_STEPS = 33     # action chunk length
ACVS_INFERENCE_STEPS = 10  # ACVS denoising steps (used in all experiments)

# GRPO hyperparameters (official Wan2.1)
CLIP_RANGE = 1e-3        # grpo.py:56 — Wan2.1-specific (base default is 1e-4)
ADV_CLIP_MAX = 5.0       # base.py:97
BETA_KL = 0.0            # FG-B spec: no KL regularization (Wan2.1 uses 0.004)
LR = 1e-6                # smoke test learning rate
MAX_GRAD_NORM = 1.0      # base.py:89
ADAM_BETAS = (0.9, 0.999)
ADAM_WEIGHT_DECAY = 1e-4
ADAM_EPS = 1e-8

# SDE config (plain sigma-interpolation, matches Wan2.1 pipeline)
SDE_SHIFT = 1.0
SDE_NUM_STEPS = L
SDE_DETERMINISTIC = False

# ACVS config
OBS_PROMPT = "turn on the switch"

print("=" * 80)
print("FG-B: True Flow-GRPO Real-ACVS One-Step Update")
print("=" * 80)
print(f"  K={K}, L={L}, BASE_SEED={BASE_SEED}")
print(f"  CLIP_RANGE={CLIP_RANGE} (Wan2.1 official)")
print(f"  LR={LR}, BETA_KL={BETA_KL}")
print(f"  Checkpoint: {CHECKPOINT_PATH}")
print(f"  Output: {OUT_DIR}")
sys.stdout.flush()

# ===========================================================================
# Section 3: Load VAM (base policy θ_old)
# ===========================================================================

print("\n" + "=" * 60)
print("Section 3: Loading VAM (base policy)")
print("=" * 60)
sys.stdout.flush()

# Import utils (mute forward_pass sentinel, set SDPA)
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

cfg_path = '/tmp/fgb_vam.yaml'
with open(cfg_path, 'w') as f:
    dump(vam_cfg, f, Dumper=Dumper)

vam = TauPolicy(
    config_file=cfg_path, device=torch.device('cuda:0'), rank=0,
    compile_model=False, attention_impl='sdpa',
    enable_self_attn_fused_qkv=True, enable_context_null_cache=True,
)
# Set to eval mode (no dropout — deterministic model forward)
vam.diffusion_model.eval()
print(f"  VAM loaded — GPU 0: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
sys.stdout.flush()

# ===========================================================================
# Section 4: Load ACVS (TauSimulator)
# ===========================================================================

print("\n" + "=" * 60)
print("Section 4: Loading ACVS")
print("=" * 60)
sys.stdout.flush()

from web_infer_utils.simulator.TauSimulator import TauSimulator

acvs_cfg = load(open(os.path.join(CAUSAL_ROOT, 'configs/runtime/acvs_deploy.yaml')), Loader=Loader)
acvs_cfg['diffusion_model']['model_path'] = ACVS_MODEL_PATH
acvs_cfg['statistics_file'] = ACVS_STATS_PATH

acvs_cfg_path = '/tmp/fgb_acvs.yaml'
with open(acvs_cfg_path, 'w') as f:
    dump(acvs_cfg, f, Dumper=Dumper)

# ACVS on cuda:1 (standard placement)
acvs = TauSimulator(config_file=acvs_cfg_path, device=torch.device('cuda:1'), rank=1)
print(f"  ACVS loaded — GPU 1: {torch.cuda.memory_allocated(1)/1e9:.2f} GB")
sys.stdout.flush()

# ===========================================================================
# Section 5: Load Observation (seed100 proxy)
# ===========================================================================

print("\n" + "=" * 60)
print("Section 5: Loading observation (seed100 proxy)")
print("=" * 60)
sys.stdout.flush()

# Use cached real observation (v0c_tau_input.npz — captured from RoboTwin reset)
if os.path.exists(OBS_CACHE):
    obs_data = np.load(OBS_CACHE, allow_pickle=True)
    obs_img = obs_data['obs']  # (3, 3, 192, 256) float32 in [-1, 1]
    state_14d = obs_data['state'].astype(np.float64)  # (14,)
    grip_2d = obs_data['gripper'].astype(np.float64)  # (2,)
    print(f"  Loaded real observation: obs={obs_img.shape}, state={state_14d.shape}")
elif os.path.exists(RESET_CACHE):
    # Build state from reset data (v2c pattern)
    from adapters.robotwin.frame_utils import world_pose_to_arm_base
    from adapters.robotwin.rotation_utils import reorder_quaternion
    from adapters.robotwin.gripper_utils import robotwin_gripper_to_tau

    reset = np.load(RESET_CACHE, allow_pickle=True)
    left = reset['left_endpose']
    right = reset['right_endpose']
    lbp, _ = world_pose_to_arm_base(left[0:3], left[3:7])
    rbp, _ = world_pose_to_arm_base(right[0:3], right[3:7])
    state_14d = np.concatenate([
        lbp, reorder_quaternion(left[3:7], 'wxyz', 'xyzw'),
        rbp, reorder_quaternion(right[3:7], 'wxyz', 'xyzw')
    ]).astype(np.float64)
    grip_2d = robotwin_gripper_to_tau(
        np.array([float(reset['left_gripper']), float(reset['right_gripper'])])
    )
    obs_img = np.zeros((3, 3, 192, 256), dtype=np.float32)
    print(f"  Built from reset: state_14d shape={state_14d.shape}")
else:
    # Fallback: dummy
    state_14d = np.zeros(14, dtype=np.float64)
    grip_2d = np.array([0.0, 0.0], dtype=np.float64)
    obs_img = np.zeros((3, 3, 192, 256), dtype=np.float32)
    print(f"  WARNING: No cached observation found, using dummy zeros")

print(f"  State range: pos=[{state_14d[0:3].min():.3f},{state_14d[0:3].max():.3f}]")
print(f"  Obs range: [{obs_img.min():.3f}, {obs_img.max():.3f}], shape={obs_img.shape}")
sys.stdout.flush()

# ===========================================================================
# Section 6: K=4 SDE Rollouts with pipeline wrapper
# ===========================================================================

print("\n" + "=" * 60)
print("Section 6: K=4 SDE Rollouts")
print("=" * 60)
sys.stdout.flush()

from flow_grpo.tau_pipeline_with_logprob import TauPipelineWithLogprob, sample_k_trajectories
from flow_grpo.tau_flow_grpo_buffer import TauTrajectory, TauTrajectoryGroup, build_trajectory_from_sde_result

pipeline_wrapper = TauPipelineWithLogprob(vam)

t0_rollout = time.monotonic()
sde_results = sample_k_trajectories(
    pipeline_wrapper=pipeline_wrapper,
    state_14d=state_14d,
    gripper_states=grip_2d,
    obs_img=obs_img,
    prompt=OBS_PROMPT,
    k=K,
    base_seed=BASE_SEED,
    num_inference_steps=L,
    return_velocities=True,
)
t_rollout = time.monotonic() - t0_rollout

# Build TauTrajectory objects
trajectories = []
for i, result in enumerate(sde_results):
    traj = build_trajectory_from_sde_result(
        result=result,
        state_14d=state_14d,
        gripper_states=grip_2d,
        prompt=OBS_PROMPT,
    )
    traj.k_idx = i
    traj.seed = result['seed']
    trajectories.append(traj)

# Quick stats
for i, traj in enumerate(trajectories):
    lp_mean = traj.log_probs.mean().item()
    lp_std = traj.log_probs.std().item()
    action_norm = traj.latents[-1]  # x_0 (clean action)
    print(f"  k={i}: seed={traj.seed}, "
          f"logp mean={lp_mean:.4f} std={lp_std:.4f}, "
          f"action range=[{action_norm.min():.3f},{action_norm.max():.3f}]")

print(f"  Rollout time: {t_rollout:.1f}s for {K} trajectories")
print(f"  Conditioning cached: pipeline._cached_cond is {'set' if pipeline_wrapper._cached_cond else 'None'}")
sys.stdout.flush()

# ===========================================================================
# Section 7: True Model-Recomputation Identity Check (HARD GATE)
# ===========================================================================

print("\n" + "=" * 60)
print("Section 7: True Model-Recomputation Identity Check (HARD GATE)")
print("=" * 60)
sys.stdout.flush()

print("  Recomputing logprobs through pipeline wrapper (no_grad, eval mode)...")
sys.stdout.flush()

all_identity_results = []  # list of dicts per transition
velocity_diffs_all = []    # velocity diffs per transition

for k_idx, traj in enumerate(trajectories):
    t0_recomp = time.monotonic()
    recomp_log_probs, recomp_velocities = pipeline_wrapper.recompute_trajectory_logprobs(
        traj,
        enable_grad=False,  # identity check: no grad needed
    )
    t_recomp = time.monotonic() - t0_recomp

    # Per-step diagnostics
    for step_i in range(L):
        stored_lp = traj.log_probs[step_i]
        recomp_lp = recomp_log_probs[step_i]
        stored_v = traj.velocities[step_i]  # (T, D)
        recomp_v = recomp_velocities[step_i]  # (T, D)

        log_ratio = (recomp_lp - stored_lp).item()
        ratio = math.exp(min(log_ratio, 20))  # clamp for safety
        v_diff_max = (stored_v - recomp_v).abs().max().item()

        all_identity_results.append({
            'k': k_idx,
            'step': step_i,
            'stored_logp': stored_lp.item(),
            'recomp_logp': recomp_lp.item(),
            'log_ratio': log_ratio,
            'ratio': ratio,
            'v_diff_max': v_diff_max,
        })
        velocity_diffs_all.append(v_diff_max)

    ratios_k = [r['ratio'] for r in all_identity_results[-L:]]
    print(f"  k={k_idx}: {t_recomp*1000:.0f}ms, "
          f"ratios={[f'{v:.6f}' for v in ratios_k]}")

# Aggregate diagnostics
ratios = np.array([r['ratio'] for r in all_identity_results])
log_ratios = np.array([r['log_ratio'] for r in all_identity_results])
v_diffs = np.array(velocity_diffs_all)

print(f"\n  --- Identity Check Summary (n={len(all_identity_results)} transitions) ---")
print(f"  Ratio:       mean={ratios.mean():.8f}, std={ratios.std():.8f}, "
      f"min={ratios.min():.8f}, max={ratios.max():.8f}")
print(f"  |ratio-1|:   max={np.abs(ratios-1).max():.8f}")
print(f"  log_ratio:   mean={log_ratios.mean():.8f}, std={log_ratios.std():.8f}")
print(f"  vel diff:    max={v_diffs.max():.8f}, mean={v_diffs.mean():.8f}")

# Per-step table
print(f"\n  --- Per-Step Identity ---")
for step_i in range(L):
    step_ratios = [r['ratio'] for r in all_identity_results if r['step'] == step_i]
    step_vdiffs = [r['v_diff_max'] for r in all_identity_results if r['step'] == step_i]
    print(f"  step {step_i}: ratio={np.mean(step_ratios):.8f}±{np.std(step_ratios):.8f}, "
          f"v_diff max={np.max(step_vdiffs):.8f}")

# HARD GATE decision
IDENTITY_PASS = bool((np.abs(ratios - 1).max() < 0.01) and (v_diffs.max() < 1e-4))

if IDENTITY_PASS:
    print(f"\n  >>> IDENTITY CHECK: PASS ✓ (max|ratio-1|={np.abs(ratios-1).max():.8f}, max|v_diff|={v_diffs.max():.8f})")
else:
    print(f"\n  >>> IDENTITY CHECK: FAIL ✗ (max|ratio-1|={np.abs(ratios-1).max():.8f}, max|v_diff|={v_diffs.max():.8f})")
    print(f"  ABORTING: True model-recomputation identity not verified. Do NOT proceed to training.")
    # Save identity check results before aborting
    identity_results_path = os.path.join(OUT_DIR, 'identity_check.json')
    with open(identity_results_path, 'w') as f:
        json.dump({
            'pass': False,
            'ratios': [float(r) for r in ratios],
            'log_ratios': [float(r) for r in log_ratios],
            'velocity_diffs': [float(d) for d in v_diffs],
            'per_transition': all_identity_results,
        }, f, indent=2)
    print(f"  Identity results saved to {identity_results_path}")
    sys.exit(1)

sys.stdout.flush()

# ===========================================================================
# Section 8: ACVS Reward Computation
# ===========================================================================

print("\n" + "=" * 60)
print("Section 8: ACVS Reward Computation")
print("=" * 60)
sys.stdout.flush()

all_Q = []

for k_idx, traj in enumerate(trajectories):
    # Get final clean action (x_0 = last latent, σ≈0)
    final_action_norm = traj.latents[-1]  # (T, D) = (33, 20)
    # Denormalize to physical eef6d space
    final_action_phys = pipeline_wrapper.denormalize_action(final_action_norm)

    # ACVS scoring
    acvs.reset()
    with torch.inference_mode():
        _, reward_k = acvs.play(
            obs=obs_img,
            prompt=OBS_PROMPT,
            actions=final_action_phys.cpu().numpy()[:EXECUTION_STEPS],
            num_inference_steps=ACVS_INFERENCE_STEPS,
            execution_step=EXECUTION_STEPS,
            n_mem=3,
        )
    Q_k = float(np.max(reward_k))
    all_Q.append(Q_k)
    traj.reward = Q_k

    print(f"  k={k_idx}: Q={Q_k:.6f}, "
          f"action phys range=[{final_action_phys.min():.4f},{final_action_phys.max():.4f}]")

Qs = np.array(all_Q)
print(f"\n  Q distribution: mean={Qs.mean():.6f}, std={Qs.std():.6f}, "
      f"min={Qs.min():.6f}, max={Qs.max():.6f}")

# Check for degenerate rewards
if Qs.std() < 1e-8:
    print(f"  WARNING: All K={K} candidates received identical ACVS rewards (std≈0).")
    print(f"  Advantage standardization will fail. Check ACVS sensitivity.")
    # Continue anyway — this is a diagnostic, not a hard failure
sys.stdout.flush()

# ===========================================================================
# Section 9: Vanilla GRPO Advantages
# ===========================================================================

print("\n" + "=" * 60)
print("Section 9: Vanilla GRPO Advantages")
print("=" * 60)
sys.stdout.flush()

# Create group and compute advantages
group = TauTrajectoryGroup(
    group_id="fgb_step1",
    state_14d=state_14d,
    gripper_states=grip_2d,
    trajectories=trajectories,
)
group.compute_advantages(eps=1e-6)

for k_idx, traj in enumerate(trajectories):
    print(f"  k={k_idx}: Q={traj.reward:.6f}, A={traj.advantage:+.6f}")

advantages_arr = np.array([t.advantage for t in trajectories])
print(f"  A distribution: mean={advantages_arr.mean():.6f}, std={advantages_arr.std():.6f}")
print(f"  Positive A: {sum(advantages_arr > 0)}, Negative A: {sum(advantages_arr < 0)}")
sys.stdout.flush()

# ===========================================================================
# Section 10: GRPO Loss + Optimizer Step
# ===========================================================================

print("\n" + "=" * 60)
print("Section 10: GRPO Loss + Optimizer Step")
print("=" * 60)
sys.stdout.flush()

from flow_grpo.tau_flow_grpo_loss import compute_grpo_loss, compute_identity_check

# Collect trainable parameters (action-related params + non-vlm)
trainable_params = []
for name, param in vam.diffusion_model.named_parameters():
    if 'action_' in name or (name.find('action_') < 0 and name.find('vlm_interface') < 0):
        trainable_params.append(param)
        param.requires_grad = True
    else:
        param.requires_grad = False

n_trainable = sum(p.numel() for p in trainable_params)
n_total = sum(p.numel() for p in vam.diffusion_model.parameters())
print(f"  Trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")

# Record pre-update parameter snapshot
param_snapshot = {}
for name, param in vam.diffusion_model.named_parameters():
    if param.requires_grad:
        param_snapshot[name] = param.detach().clone()

# Setup optimizer
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=LR,
    betas=ADAM_BETAS,
    weight_decay=ADAM_WEIGHT_DECAY,
    eps=ADAM_EPS,
)
optimizer.zero_grad()

# Accumulate GRPO loss — batch ALL transitions together
print(f"\n  Computing GRPO loss with clip_range={CLIP_RANGE}...")
sys.stdout.flush()

# Collect all current logprobs and old logprobs as (K*L,) tensors
all_cur_logps = []
all_old_logps = []
all_advantages = []

for k_idx, traj in enumerate(trajectories):
    A_k = torch.tensor(traj.advantage, device='cuda:0', dtype=torch.float32)

    # Recompute logprobs WITH grad
    cur_log_probs, cur_velocities = pipeline_wrapper.recompute_trajectory_logprobs(
        traj,
        enable_grad=True,  # CRITICAL: enable gradient flow for training
    )

    for step_i in range(L):
        all_cur_logps.append(cur_log_probs[step_i])
        all_old_logps.append(traj.log_probs[step_i].detach())
        all_advantages.append(A_k)

# Stack into batched tensors
log_prob_current = torch.stack(all_cur_logps).unsqueeze(1)  # (K*L, 1)
log_prob_old = torch.stack(all_old_logps).unsqueeze(1)      # (K*L, 1)
advantages_batch = torch.stack(all_advantages)               # (K*L,)

loss_dict = compute_grpo_loss(
    log_prob_current=log_prob_current,
    log_prob_old=log_prob_old,
    advantages=advantages_batch,
    clip_range=CLIP_RANGE,
    adv_clip_max=ADV_CLIP_MAX,
    beta_kl=BETA_KL,
)

total_loss = loss_dict['loss']
clipfrac_avg = loss_dict['clipfrac'].item()
ratio_mean_avg = loss_dict['ratio_mean'].item()

print(f"  Loss: {total_loss.item():.6f}")
print(f"  Avg ratio: {ratio_mean_avg:.6f}, clipfrac: {clipfrac_avg:.4f}")

# Backward
total_loss.backward()

# Gradient diagnostics
total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
none_grad = sum(1 for p in trainable_params if p.grad is None)
nan_grad = sum(1 for p in trainable_params if p.grad is not None
               and not torch.all(torch.isfinite(p.grad)))
nonzero_grad = sum(1 for p in trainable_params if p.grad is not None
                   and p.grad.norm() > 0)
print(f"  Grad norm (pre-clip): {total_norm:.4f}")
print(f"  None grad: {none_grad}, NaN/Inf: {nan_grad}, Nonzero: {nonzero_grad}")

# Optimizer step
optimizer.step()

# Compute parameter delta
param_deltas = {}
max_delta = 0.0
for name, param in vam.diffusion_model.named_parameters():
    if name in param_snapshot:
        delta = (param - param_snapshot[name]).abs().max().item()
        param_deltas[name] = delta
        max_delta = max(max_delta, delta)

mean_delta = np.mean(list(param_deltas.values()))
print(f"  Param delta: max={max_delta:.6e}, mean={mean_delta:.6e}")
print(f"  Params changed: {sum(1 for d in param_deltas.values() if d > 0)}/{len(param_deltas)}")

# Put model back in eval mode for post-update checks
vam.diffusion_model.eval()
sys.stdout.flush()

# ===========================================================================
# Section 11: Post-Update Diagnostics
# ===========================================================================

print("\n" + "=" * 60)
print("Section 11: Post-Update Diagnostics")
print("=" * 60)
sys.stdout.flush()

# --- 11a: Ratio Direction Check ---
print("\n  --- 11a: Ratio Direction Check ---")
print("  Recomputing logprobs with UPDATED model on FIXED trajectories...")
sys.stdout.flush()

post_ratios_by_adv = {'positive': [], 'negative': [], 'zero': []}
post_all_ratios = []

for k_idx, traj in enumerate(trajectories):
    with torch.no_grad():
        post_log_probs, post_velocities = pipeline_wrapper.recompute_trajectory_logprobs(
            traj,
            enable_grad=False,
        )

    ratios_k = []
    for step_i in range(L):
        log_ratio = (post_log_probs[step_i] - traj.log_probs[step_i]).item()
        ratio = math.exp(min(log_ratio, 20))
        ratios_k.append(ratio)
        post_all_ratios.append(ratio)

    mean_ratio_k = np.mean(ratios_k)
    adv = traj.advantage
    if adv > 0.01:
        post_ratios_by_adv['positive'].extend(ratios_k)
    elif adv < -0.01:
        post_ratios_by_adv['negative'].extend(ratios_k)
    else:
        post_ratios_by_adv['zero'].extend(ratios_k)

    print(f"  k={k_idx}: A={adv:+.4f}, mean post-ratio={mean_ratio_k:.6f}, "
          f"ratios={[f'{r:.6f}' for r in ratios_k]}")

post_ratios_arr = np.array(post_all_ratios)
print(f"\n  Post-update ratios: mean={post_ratios_arr.mean():.6f}, std={post_ratios_arr.std():.6f}, "
      f"min={post_ratios_arr.min():.6f}, max={post_ratios_arr.max():.6f}")

# Direction diagnostic
for label, rs in post_ratios_by_adv.items():
    if rs:
        mean_r = np.mean(rs)
        print(f"  {label:>8} A: n={len(rs)}, mean ratio={mean_r:.6f} "
              f"({'≥1 ✓' if mean_r >= 1 else '<1'})")

# --- 11b: Updated SDE Sampling ---
print("\n  --- 11b: Updated SDE Sampling ---")
sys.stdout.flush()

gen_new = torch.Generator(device='cuda:0')
gen_new.manual_seed(BASE_SEED * 1000 + K)  # different seed from training

with torch.no_grad():
    new_result = pipeline_wrapper.sample_with_logprob(
        state_14d=state_14d,
        gripper_states=grip_2d,
        obs_img=obs_img,
        prompt=OBS_PROMPT,
        num_inference_steps=L,
        execution_steps=EXECUTION_STEPS,
        seed=BASE_SEED * 1000 + K,
        generator=gen_new,
    )

new_action = new_result['action']
new_logp_mean = torch.stack([lp.flatten()[0] for lp in new_result['all_log_probs']]).mean().item()
print(f"  Updated SDE: action shape={tuple(new_action.shape)}, "
      f"finite={torch.isfinite(new_action).all().item()}, "
      f"logp mean={new_logp_mean:.4f}")

# Compare with pre-update trajectory 0
pre_action_0 = trajectories[0].latents[-1]  # (33, 20)
action_diff = (new_action - pre_action_0).abs().mean().item()
print(f"  Updated vs pre-update k=0: mean|diff|={action_diff:.6f}")
print(f"  Actions distinct: {action_diff > 1e-6}")

# --- 11c: Updated Native UniPC ---
print("\n  --- 11c: Updated Native UniPC ---")
sys.stdout.flush()

with torch.no_grad():
    unipc_action = pipeline_wrapper.sample_unipc(
        state_14d=state_14d,
        gripper_states=grip_2d,
        obs_img=obs_img,
        prompt=OBS_PROMPT,
        num_inference_steps=L,
        execution_steps=EXECUTION_STEPS,
        seed=BASE_SEED * 1000 + K + 1,
    )

print(f"  Updated UniPC: action shape={tuple(unipc_action.shape)}, "
      f"finite={torch.isfinite(unipc_action).all().item()}, "
      f"range=[{unipc_action.min():.4f},{unipc_action.max():.4f}]")

sys.stdout.flush()

# ===========================================================================
# Section 12: Save Outputs
# ===========================================================================

print("\n" + "=" * 60)
print("Section 12: Saving Outputs")
print("=" * 60)
sys.stdout.flush()

# 12a: Config
config_out = {
    'experiment': 'FG-B True Flow-GRPO Real-ACVS One-Step Update',
    'date': '2026-08-11',
    'checkpoint': CHECKPOINT_PATH,
    'hyperparameters': {
        'K': K, 'L': L, 'BASE_SEED': BASE_SEED,
        'EXECUTION_STEPS': EXECUTION_STEPS,
        'ACVS_INFERENCE_STEPS': ACVS_INFERENCE_STEPS,
        'CLIP_RANGE': CLIP_RANGE, 'ADV_CLIP_MAX': ADV_CLIP_MAX,
        'BETA_KL': BETA_KL, 'LR': LR, 'MAX_GRAD_NORM': MAX_GRAD_NORM,
        'ADAM_BETAS': list(ADAM_BETAS), 'ADAM_WEIGHT_DECAY': ADAM_WEIGHT_DECAY,
        'ADAM_EPS': ADAM_EPS,
    },
    'config_sources': {
        'clip_range': 'grpo.py:56 (Wan2.1 official)',
        'adv_clip_max': 'base.py:97',
        'max_grad_norm': 'base.py:89',
        'learning_rate': 'smoke test (official Wan2.1 uses 1e-4)',
    },
}
with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
    json.dump(config_out, f, indent=2)

# 12b: Identity check
identity_out = {
    'pass': IDENTITY_PASS,
    'method': 'true-model-recomputation through cached pipeline conditioning',
    'n_transitions': len(all_identity_results),
    'aggregate': {
        'ratio_mean': float(ratios.mean()), 'ratio_std': float(ratios.std()),
        'ratio_min': float(ratios.min()), 'ratio_max': float(ratios.max()),
        'max_abs_ratio_minus_1': float(np.abs(ratios - 1).max()),
        'log_ratio_mean': float(log_ratios.mean()), 'log_ratio_std': float(log_ratios.std()),
        'velocity_diff_max': float(v_diffs.max()), 'velocity_diff_mean': float(v_diffs.mean()),
    },
    'per_step': {},
    'per_transition': all_identity_results,
}
for step_i in range(L):
    step_ratios = [float(r['ratio']) for r in all_identity_results if r['step'] == step_i]
    step_vdiffs = [float(r['v_diff_max']) for r in all_identity_results if r['step'] == step_i]
    identity_out['per_step'][f'step_{step_i}'] = {
        'ratio_mean': float(np.mean(step_ratios)), 'ratio_std': float(np.std(step_ratios)),
        'v_diff_max': float(np.max(step_vdiffs)), 'n': len(step_ratios),
    }
with open(os.path.join(OUT_DIR, 'identity_check.json'), 'w') as f:
    json.dump(identity_out, f, indent=2)

# 12c: ACVS rewards
acvs_out = {
    'Qs': [float(q) for q in Qs],
    'Q_mean': float(Qs.mean()), 'Q_std': float(Qs.std()),
    'Q_min': float(Qs.min()), 'Q_max': float(Qs.max()),
}
with open(os.path.join(OUT_DIR, 'acvs_rewards.json'), 'w') as f:
    json.dump(acvs_out, f, indent=2)

# 12d: Training results
training_out = {
    'loss': float(total_loss.item()),
    'grad_norm': float(total_norm),
    'clipfrac': float(clipfrac_avg),
    'ratio_mean': float(ratio_mean_avg),
    'param_delta_max': float(max_delta),
    'param_delta_mean': float(mean_delta),
    'n_trainable': n_trainable,
    'n_total': n_total,
    'none_grad': none_grad,
    'nan_grad': nan_grad,
    'nonzero_grad': nonzero_grad,
    'param_deltas_top10': dict(sorted(param_deltas.items(), key=lambda x: -x[1])[:10]),
}
with open(os.path.join(OUT_DIR, 'training.json'), 'w') as f:
    json.dump(training_out, f, indent=2)

# 12e: Post-update diagnostics
post_out = {
    'ratio_direction': {
        'positive_A': {
            'n': len(post_ratios_by_adv['positive']),
            'mean_ratio': float(np.mean(post_ratios_by_adv['positive'])) if post_ratios_by_adv['positive'] else None,
        },
        'negative_A': {
            'n': len(post_ratios_by_adv['negative']),
            'mean_ratio': float(np.mean(post_ratios_by_adv['negative'])) if post_ratios_by_adv['negative'] else None,
        },
        'zero_A': {
            'n': len(post_ratios_by_adv['zero']),
            'mean_ratio': float(np.mean(post_ratios_by_adv['zero'])) if post_ratios_by_adv['zero'] else None,
        },
    },
    'post_ratios_mean': float(post_ratios_arr.mean()),
    'post_ratios_std': float(post_ratios_arr.std()),
    'post_ratios_min': float(post_ratios_arr.min()),
    'post_ratios_max': float(post_ratios_arr.max()),
    'updated_sde': {
        'action_shape': list(new_action.shape),
        'finite': bool(torch.isfinite(new_action).all().item()),
        'logp_mean': float(new_logp_mean),
        'diff_vs_pre_update': float(action_diff),
        'distinct': bool(action_diff > 1e-6),
    },
    'updated_unipc': {
        'action_shape': list(unipc_action.shape),
        'finite': bool(torch.isfinite(unipc_action).all().item()),
        'action_min': float(unipc_action.min()),
        'action_max': float(unipc_action.max()),
    },
}
with open(os.path.join(OUT_DIR, 'post_update_diagnostics.json'), 'w') as f:
    json.dump(post_out, f, indent=2)

# 12f: Full trajectories
torch.save({
    'trajectories': [(t.latents.cpu(), t.next_latents.cpu(), t.log_probs.cpu(),
                      t.velocities.cpu() if t.velocities is not None else None,
                      t.reward, t.advantage, t.seed)
                     for t in trajectories],
    'Qs': Qs.tolist(),
    'advantages': advantages_arr.tolist(),
}, os.path.join(OUT_DIR, 'trajectories.pt'))

# 12g: Structured summary
fgb_results = {
    'experiment': 'FG-B True Flow-GRPO Real-ACVS One-Step Update',
    'verdict': 'PASS' if IDENTITY_PASS and total_norm > 0 and max_delta > 0 else 'PARTIAL',
    'date': '2026-08-11',
    'identity_check': identity_out['aggregate'],
    'acvs_rewards': acvs_out,
    'training': {
        'loss': training_out['loss'],
        'grad_norm': training_out['grad_norm'],
        'param_delta_max': training_out['param_delta_max'],
        'nonzero_grad': training_out['nonzero_grad'],
    },
    'post_update': {
        'ratio_mean': post_out['post_ratios_mean'],
        'sde_valid': post_out['updated_sde']['finite'],
        'unipc_valid': post_out['updated_unipc']['finite'],
    },
    'fpo_used': False,  # FG-B explicitly prohibits FPO
    'er_cag_used': False,
    'reference_used': False,
}
with open(os.path.join(OUT_DIR, 'fgb_results.json'), 'w') as f:
    json.dump(fgb_results, f, indent=2)

print(f"  Saved {len(os.listdir(OUT_DIR))} files to {OUT_DIR}")
sys.stdout.flush()

# ===========================================================================
# Section 13: Final Report
# ===========================================================================

print("\n" + "=" * 80)
print("FG-B: True Flow-GRPO Real-ACVS One-Step Update — FINAL REPORT")
print("=" * 80)
print()
print(f"  Experiment: FG-B")
print(f"  Date: 2026-08-11")
print(f"  Base policy: V0-D6 step100 ({CHECKPOINT_PATH})")
print(f"  Observation: seed100 proxy ({'real obs' if os.path.exists(OBS_CACHE) else 'dummy'})")
print()
print(f"  --- Identity Check (Hard Gate) ---")
print(f"  Method: True model-recomputation through cached pipeline conditioning")
print(f"  N transitions: {len(all_identity_results)}")
print(f"  Ratio: mean={ratios.mean():.8f}, max|ratio-1|={np.abs(ratios-1).max():.8f}")
print(f"  Velocity diff: max={v_diffs.max():.8f}, mean={v_diffs.mean():.8f}")
print(f"  Status: {'✅ PASS' if IDENTITY_PASS else '❌ FAIL'}")
print()
print(f"  --- ACVS Rewards ---")
print(f"  Q values: {[f'{q:.6f}' for q in Qs]}")
print(f"  Q mean={Qs.mean():.6f}, std={Qs.std():.6f}")
print()
print(f"  --- GRPO Training ---")
print(f"  Loss: {total_loss.item():.6f}")
print(f"  Grad norm: {total_norm:.4f}")
print(f"  Param delta: max={max_delta:.6e}, mean={mean_delta:.6e}")
print(f"  Nonzero grad params: {nonzero_grad}/{len(trainable_params)}")
print(f"  Optimizer: AdamW(lr={LR}, betas={ADAM_BETAS})")
print()
print(f"  --- Post-Update ---")
print(f"  Ratio direction:")
for label in ['positive', 'negative', 'zero']:
    rs = post_ratios_by_adv[label]
    if rs:
        print(f"    {label:>8} A: n={len(rs)}, mean ratio={np.mean(rs):.6f}")
print(f"  Updated SDE: finite={post_out['updated_sde']['finite']}, "
      f"logp={new_logp_mean:.4f}")
print(f"  Updated UniPC: finite={post_out['updated_unipc']['finite']}, "
      f"range=[{unipc_action.min():.4f},{unipc_action.max():.4f}]")
print()
print(f"  --- Compliance ---")
print(f"  FPO used: NO")
print(f"  ER-CAG used: NO")
print(f"  Reference used: NO")
print(f"  V3 continued: NO")
print(f"  Multi-step training: NO (one optimizer step)")
print(f"  Pipeline wrapper: YES (shared between sampling and recomputation)")
print()
print(f"  --- Output ---")
print(f"  Directory: {OUT_DIR}")
print(f"  Files: {sorted(os.listdir(OUT_DIR))}")
print()
overall_verdict = "PASS" if (IDENTITY_PASS and total_norm > 0 and max_delta > 0
                              and post_out['updated_sde']['finite']
                              and post_out['updated_unipc']['finite']) else "PARTIAL"
print(f"  OVERALL VERDICT: {overall_verdict}")
print()
print("=" * 80)
print("FG-B Complete")
print("=" * 80)
