"""τ₀ Flow-GRPO Loss — PPO-clipped policy gradient on SDE transition logprobs.

Follows the official Flow-GRPO training loop (train_wan2_1.py lines 943-963):

  ratio = exp(log_prob_current - log_prob_old)
  unclipped = -advantage * ratio
  clipped = -advantage * clip(ratio, 1-ε, 1+ε)
  loss = mean(max(unclipped, clipped))

Optional KL regularization (β=0 for FG-A):
  kl = (μ_cur - μ_ref)² / (2·σ²_trans)
  loss = policy_loss + β * kl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


def compute_grpo_loss(
    log_prob_current: torch.Tensor,
    log_prob_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.0001,   # official default
    adv_clip_max: float = 5.0,
    beta_kl: float = 0.0,
    prev_sample_mean: Optional[torch.Tensor] = None,
    prev_sample_mean_ref: Optional[torch.Tensor] = None,
    trans_std: Optional[torch.Tensor] = None,
) -> dict:
    """Compute the GRPO PPO-clipped loss.

    Args:
        log_prob_current: (B, L) current model log-prob per transition
        log_prob_old: (B, L) old (sampling-time) log-prob per transition
        advantages: (B,) per-candidate advantages (broadcast to L steps)
        clip_range: PPO clipping epsilon
        adv_clip_max: max absolute advantage value
        beta_kl: KL penalty coefficient (0 for FG-A)
        prev_sample_mean: (B, L, T, D) current model transition mean (for KL)
        prev_sample_mean_ref: (B, L, T, D) reference transition mean (for KL)
        trans_std: (B, L) or scalar transition std deviation (for KL)

    Returns:
        dict with keys: loss, policy_loss, kl_loss, approx_kl, clipfrac, ratio_mean
    """
    B, L = log_prob_current.shape

    # Clip advantages
    advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)  # (B,)

    # Broadcast advantages to all timesteps
    advantages = advantages.unsqueeze(1).expand(-1, L)  # (B, L)

    # Importance ratio
    log_ratio = log_prob_current - log_prob_old
    ratio = torch.exp(torch.clamp(log_ratio, min=-20, max=20))  # (B, L)

    # PPO surrogate
    unclipped_loss = -advantages * ratio
    clipped_loss = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

    # KL regularization (β=0 for FG-A)
    kl_loss = torch.tensor(0.0, device=log_prob_current.device)
    if beta_kl > 0 and prev_sample_mean is not None and prev_sample_mean_ref is not None:
        # Mean squared difference over action dimensions
        kl_per_step = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(
            dim=tuple(range(2, prev_sample_mean.ndim))
        )  # (B, L)
        if trans_std is not None:
            while trans_std.ndim < kl_per_step.ndim:
                trans_std = trans_std.unsqueeze(-1)
            kl_per_step = kl_per_step / (2 * (trans_std ** 2))
        kl_loss = torch.mean(kl_per_step)

    loss = policy_loss + beta_kl * kl_loss

    # Diagnostics
    with torch.no_grad():
        approx_kl = 0.5 * torch.mean((log_prob_current - log_prob_old) ** 2)
        clipfrac = torch.mean((torch.abs(ratio - 1.0) > clip_range).float())

    return {
        'loss': loss,
        'policy_loss': policy_loss,
        'kl_loss': kl_loss,
        'approx_kl': approx_kl,
        'clipfrac': clipfrac,
        'ratio_mean': ratio.mean(),
        'ratio_std': ratio.std(),
        'log_ratio_mean': log_ratio.mean(),
    }


def compute_identity_check(
    log_prob_current: torch.Tensor,
    log_prob_old: torch.Tensor,
) -> dict:
    """Verify that with theta_current == theta_old, ratio ≈ 1.

    Args:
        log_prob_current: recomputed logprobs with same model
        log_prob_old: stored logprobs from sampling

    Returns:
        dict with ratio statistics for FG-A gate check
    """
    log_ratio = log_prob_current - log_prob_old
    ratio = torch.exp(torch.clamp(log_ratio, min=-20, max=20))

    return {
        'log_ratio_mean': log_ratio.mean().item(),
        'log_ratio_std': log_ratio.std().item(),
        'log_ratio_min': log_ratio.min().item(),
        'log_ratio_max': log_ratio.max().item(),
        'ratio_mean': ratio.mean().item(),
        'ratio_std': ratio.std().item(),
        'ratio_min': ratio.min().item(),
        'ratio_max': ratio.max().item(),
        'max_abs_ratio_minus_1': (ratio - 1.0).abs().max().item(),
        'n_transitions': log_ratio.numel(),
    }
