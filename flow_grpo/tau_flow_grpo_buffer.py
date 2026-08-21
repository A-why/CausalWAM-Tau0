"""τ₀ Flow-GRPO Trajectory Buffer — stores rollout data for training.

Stores per-transition data following the official Wan2.1 training structure:
  - latents[:, :-1]: x_t before each step
  - next_latents[:, 1:]: x_{t+1} after each step
  - log_probs: old log-probabilities from sampling
  - timesteps: scheduler timesteps for each step
  - rewards: per-candidate ACVS scores
  - advantages: standardized group advantages

For K=4 candidates per observation:
  - Group by (state, observation) for advantage standardization
  - Store observation state for re-encoding during training
"""
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class TauTrajectory:
    """One SDE action-flow trajectory.

    Stores the full denoising trajectory from noise (σ=1) to clean action (σ=0).
    L = num_inference_steps = 5 flow steps.
    """
    # Conditioning (needed to recompute logprob during training)
    state_14d: np.ndarray         # (14,) robot state in physical xyzw format
    gripper_states: np.ndarray    # (2,) gripper state
    text_prompt: str              # text instruction

    # Trajectory data
    latents: torch.Tensor         # (L, T, D) = (5, 33, 20) — x at each sigma BEFORE step
    next_latents: torch.Tensor    # (L, T, D) = (5, 33, 20) — x at each sigma AFTER step
    log_probs: torch.Tensor       # (L,) — old log-prob per step
    timesteps: torch.Tensor       # (L,) — scheduler timestep per step
    sigmas: List[float]           # [σ_0, ..., σ_L] — sigma values (L+1 entries)
    reward: Optional[float] = None     # ACVS score (filled after rollout)
    advantage: Optional[float] = None  # group-standardized advantage
    velocities: Optional[torch.Tensor] = None  # (L, T, D) velocity predictions
    noise: Optional[torch.Tensor] = None       # (L, T, D) SDE noise sampled per step

    # Debug
    k_idx: int = -1               # candidate index in group
    seed: int = -1                # random seed used

    def get_log_probs(self) -> torch.Tensor:
        """Return old logprobs as tensor."""
        return self.log_probs

    def get_timesteps_tensor(self) -> torch.Tensor:
        """Return timesteps as tensor."""
        return self.timesteps


@dataclass
class TauTrajectoryGroup:
    """K=4 trajectories sharing the same observation (same state, same RNG)."""
    group_id: str
    state_14d: np.ndarray
    gripper_states: np.ndarray
    trajectories: List[TauTrajectory] = field(default_factory=list)

    @property
    def k(self) -> int:
        return len(self.trajectories)

    def compute_advantages(self, eps: float = 1e-6) -> None:
        """Standardize rewards within group to get advantages."""
        rewards = torch.tensor([t.reward for t in self.trajectories], dtype=torch.float32)
        mean_r = rewards.mean()
        std_r = rewards.std()
        advantages = (rewards - mean_r) / (std_r + eps)
        for traj, adv in zip(self.trajectories, advantages):
            traj.advantage = float(adv.item())

    def to_training_batch(self) -> dict:
        """Collate group trajectories into training batch format.

        Returns dict with keys matching official training loop structure:
            latents: (K, L, T, D) — x_t before each step
            next_latents: (K, L, T, D) — x_{t+1} after each step
            log_probs: (K, L) — old logprobs
            timesteps: (K, L) — timesteps
            advantages: (K,) — per-candidate advantages
        """
        K = len(self.trajectories)
        L = len(self.trajectories[0].latents)  # 5

        latents = torch.stack([t.latents for t in self.trajectories])         # (K, L, T, D)
        next_latents = torch.stack([t.next_latents for t in self.trajectories])
        log_probs = torch.stack([t.log_probs for t in self.trajectories])     # (K, L)
        timesteps = torch.stack([t.timesteps for t in self.trajectories])     # (K, L)
        advantages = torch.tensor([t.advantage for t in self.trajectories])   # (K,)

        return {
            'latents': latents,
            'next_latents': next_latents,
            'log_probs': log_probs,
            'timesteps': timesteps,
            'advantages': advantages,
        }


def build_trajectory_from_sde_result(
    result: dict,
    state_14d: np.ndarray,
    gripper_states: np.ndarray,
    prompt: str = "turn on the switch",
) -> TauTrajectory:
    """Build a TauTrajectory from a pipeline_with_logprob result.

    Args:
        result: output of TauPipelineWithLogprob.sample_with_logprob()
        state_14d: (14,) robot state
        gripper_states: (2,) gripper state
        prompt: text instruction

    Returns:
        TauTrajectory ready for buffer storage
    """
    all_latents = result['all_latents']        # list of (L+1) tensors [x_1, ..., x_0]
    all_log_probs = result['all_log_probs']    # list of L tensors
    timesteps = result['timesteps']            # tensor (L,)

    L = len(timesteps)
    # latents: x BEFORE each step = [x_1, x_σ2, ..., x_σL]
    latents = torch.stack([all_latents[i].squeeze(0) for i in range(L)])        # (L, T, D)
    # next_latents: x AFTER each step = [x_σ2, ..., x_0]
    next_latents = torch.stack([all_latents[i+1].squeeze(0) for i in range(L)])  # (L, T, D)
    # log_probs: scalar per step
    log_probs = torch.stack([lp.squeeze() if lp.ndim > 0 else lp for lp in all_log_probs])  # (L,)

    # Optionally store velocities
    all_velocities = result.get('all_velocities')
    if all_velocities is not None:
        velocities = torch.stack([v.squeeze(0) for v in all_velocities])  # (L, T, D)
    else:
        velocities = None

    return TauTrajectory(
        state_14d=state_14d,
        gripper_states=gripper_states,
        text_prompt=prompt,
        latents=latents,
        next_latents=next_latents,
        log_probs=log_probs,
        timesteps=timesteps,
        sigmas=[float(s) for s in result['sigmas']],
        velocities=velocities,
        k_idx=result.get('k_idx', -1),
        seed=result.get('seed', -1),
    )
