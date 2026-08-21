"""τ₀ Flow-GRPO SDE Transition — adapted from official Wan2.1 implementation.

Official source: flow_grpo/diffusers_patch/wan_pipeline_with_logprob.py:sde_step_with_logprob
Commit: 879042cf5707f8b90daa98d147d7deac2317c5da

τ₀ flow convention (verified):
  x_σ = (1-σ)·x₀ + σ·ε   where x₀=clean action, ε~N(0,I)
  v = ε - x₀              target velocity
  Generation: σ:1→0 (noise→clean), 5 steps

SDE transition (Gaussian, isotropic):
  μ = x_σ · (1 + s²/(2σ)·dt) + v̂ · (1 + s²·(1-σ)/(2σ)·dt)
  σ²_trans = (s · sqrt(-dt))²
  s = σ_min + (σ_max - σ_min) · σ
  dt = σ_prev - σ  (< 0 during generation)

log p(x_next | x_curr) = -||x_next - μ||² / (2·σ²_trans) - D·log(σ_trans) - D/2·log(2π)
"""
import math
import torch
import numpy as np
from typing import Optional, Union


def sde_step_with_logprob(
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    model_output: torch.Tensor,
    timestep: Union[float, torch.Tensor],
    sample: torch.Tensor,
    prev_sample: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
    return_dt_and_std_dev_t: bool = False,
    noise_scale: float = 1.0,
):
    """One SDE step with explicit Gaussian log-probability.

    Adapted from official Wan2.1 sde_step_with_logprob. The main difference is
    tensor shapes: τ₀ action is (B, T, D) instead of (B, C, F, H, W).

    Args:
        sigmas: full sigma array (1D tensor, length = num_train_timesteps)
        timesteps: scheduler timesteps (1D tensor, length = num_steps)
        model_output: velocity prediction v̂_θ(x_σ, σ), shape (B, T, D)
        timestep: current timestep (scalar or shape (B,))
        sample: current action latent x_σ, shape (B, T, D)
        prev_sample: if given, use as x_next (for logprob under fixed trajectory)
        generator: random number generator for noise
        deterministic: if True, use ODE step (no noise)
        return_dt_and_std_dev_t: if True, also return dt and std_dev_t
        noise_scale: V4-B6 exploration knob — temperature scaling of the SDE transition
            std ``std_dev_t`` (1.0 = default Wan2.1 noise; >1.0 widens candidate spread).
            Applies to BOTH the injected noise and the transition std used in log_prob,
            so the log-probability stays consistent with the actual transition.

    Returns:
        prev_sample: x_{σ-Δ}, shape (B, T, D)
        log_prob: per-sample log probability, shape (B,)
        prev_sample_mean: μ, shape (B, T, D)
        std_dev_t * sqrt(-dt): transition std per sample
    """
    # Cast to float32 for numerical stability (following official convention)
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    device = sample.device
    sigmas = sigmas.to(device)
    timesteps = timesteps.to(device)

    # Map timestep → index in sigmas array
    if isinstance(timestep, torch.Tensor) and timestep.numel() == 1:
        t_val = timestep.item()
    elif isinstance(timestep, torch.Tensor):
        t_val = timestep[0].item()
    else:
        t_val = float(timestep)
    step_index = int((timesteps == t_val).nonzero(as_tuple=True)[0].item())
    prev_step_index = step_index + 1

    # sigma values
    sigma = sigmas[step_index]       # current sigma (float)
    sigma_prev = sigmas[prev_step_index]  # next sigma (float, smaller)
    sigma_max = sigmas[1].item()     # second-largest sigma
    sigma_min = sigmas[-1].item()    # last sigma (~0)

    dt = sigma_prev - sigma          # negative during generation (σ decreasing)

    # Diffusion coefficient: interpolates between sigma_min at σ=0 and sigma_max at σ=1
    std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma

    # SDE mean (same formula as official Wan2.1, adapted for 3D tensor)
    # View as scalars for broadcasting: (1, 1, 1) for (B, T, D) tensors
    coeff_x = 1.0 + (std_dev_t ** 2) / (2.0 * sigma) * dt
    coeff_v = 1.0 + (std_dev_t ** 2) * (1.0 - sigma) / (2.0 * sigma) * dt
    prev_sample_mean = sample * coeff_x + model_output * coeff_v * dt

    # Transition std with V4-B6 exploration temperature (noise_scale==1.0 -> unchanged).
    trans_std = noise_scale * std_dev_t * math.sqrt(-dt)

    if deterministic:
        # ODE step: dx = v·dt
        prev_sample = sample + dt * model_output
    elif prev_sample is not None:
        # Use given prev_sample (for computing logprob under known trajectory)
        prev_sample = prev_sample
    else:
        # Sample from SDE transition
        noise = torch.randn(
            model_output.shape,
            generator=generator,
            device=device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + trans_std * noise

    # Gaussian log-probability
    # Note: official uses prev_sample.detach() in numerator
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2.0 * (trans_std ** 2))
        - math.log(trans_std)
        - 0.5 * math.log(2.0 * math.pi)
    )
    # Mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    if return_dt_and_std_dev_t:
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, math.sqrt(-dt)
    return prev_sample, log_prob, prev_sample_mean, trans_std
