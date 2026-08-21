"""τ₀ Flow-GRPO Pipeline with Logprob — wraps τ₀ pipeline for SDE action generation.

Reuses official τ₀ condition construction (text encoding, VAE encoding, state
normalization, context KV cache). Replaces the UniPC deterministic step with
SDE stochastic step that computes and returns log-probabilities.

FG-B extensions (2026-08-11):
  - _build_conditioning(): cache all conditioning state from first sampling call
  - _model_forward_step(): reusable model forward with step-index-aware flags
  - recompute_logprob(): replay model forward through cached cond, then SDE step
  - recompute_trajectory_logprobs(): convenience for full trajectory recomputation

DOES NOT:
  - Reconstruct raw diffusion_model forward from scratch
  - Hardcode video latent shapes
  - Touch Tau tracked source or RoboTwin tracked source
"""
import sys, os, math, time
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

# Make τ₀ pipeline importable
_TAU_ROOT = os.path.join(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tau-0-wm")
if _TAU_ROOT not in sys.path:
    sys.path.insert(0, _TAU_ROOT)

from tau_flow_grpo_sde import sde_step_with_logprob


class TauPipelineWithLogprob:
    """Wraps a TauPolicy pipeline to produce SDE trajectories with logprobs.

    Usage:
        pipeline_wrapper = TauPipelineWithLogprob(tau_policy)
        result = pipeline_wrapper.sample_with_logprob(
            state=state_14d,
            gripper=grip_2d,
            num_inference_steps=5,
            generator=torch.Generator(),
        )
        # result['action']     : (33, 20) final action (normalized)
        # result['all_latents']: list of 6 tensors [x_1, x_σ2, ..., x_0]
        # result['all_log_probs']: list of 5 tensors (log p at each step)
        # result['timesteps']  : tensor of timesteps used

    For recomputation (FG-B):
        # After sampling, conditioning is auto-cached
        log_prob, velocity = pipeline_wrapper.recompute_logprob(
            x_t, x_next, timestep, step_index, enable_grad=False)
        # Transparently replays the EXACT same model forward path
    """

    def __init__(self, tau_policy):
        """
        Args:
            tau_policy: TauPolicy instance with loaded model, text_encoder, pipeline
        """
        self.policy = tau_policy
        self.pipeline = tau_policy.pipeline
        self.device = tau_policy.device
        self.dtype = tau_policy.dtype

        # FG-B: cached conditioning state (populated by _build_conditioning)
        self._cached_cond = None
        # FG-B: cached video buffers from step 0 (for recomputation)
        self._cached_video_buffers = None  # (video_states_buffer, action_context_kv_cache)
        # V4-B6: text-encoder output cache (prompt -> context). The task prompt is constant
        # across a candidate group, so re-running T5 per candidate is redundant and repeatedly
        # exposes a flaky post-reboot CUDA "misaligned address" fault in the T5 relative-pos
        # bucket. Cache keyed on the exact prompt string.
        self._text_cache = {}

    # ------------------------------------------------------------------
    # Internal helpers (shared between sampling and recomputation)
    # ------------------------------------------------------------------

    def _encode_text(self, prompt: str):
        """Reuse official text encoding (cached per prompt; V4-B6)."""
        cached = self._text_cache.get(prompt)
        if cached is not None:
            return cached
        context = self.pipeline._encode_single_text(
            prompt,
            offload_model=False,
            use_cache=False,
        )
        self._text_cache[prompt] = context
        return context

    def _encode_obs(self, obs_img):
        """Encode observation image through VAE.

        Args:
            obs_img: shape (V, C, H, W) as torch tensor, range [-1, 1] (raw format from TauPolicy)
        Returns:
            z: list of encoded latents
        """
        # Apply same preprocessing as TauPolicy.play():
        # (V,C,H,W) -> unsqueeze(2) -> (V,C,1,H,W) -> transpose(0,1) -> (C,V,1,H,W)
        V_in, C_in, H_in, W_in = obs_img.shape
        obs_5d = obs_img.unsqueeze(2).transpose(0, 1)  # -> (C, V, 1, H, W)
        C, V, T, H, W = obs_5d.shape

        img = list(obs_5d.unbind(dim=1))  # list of (C, T, H, W)
        z = self.pipeline.vae.encode(img)
        z = torch.stack(z, dim=1)  # C, V, T_lat, H_lat, W_lat

        # Rearrange to pack video views into width dimension (same as pipeline)
        from einops import rearrange
        z = [rearrange(z, "c v t h w -> c t h (v w)")]
        return z, V

    def _compute_seq_len(self, z):
        """Compute packed sequence length for the transformer."""
        # z[0] shape: (C, T_lat, H_lat, W_lat_packed)
        _, T_lat, H_lat, W_lat = z[0].shape
        F = (T_lat - 1) * self.pipeline.vae_stride[0] + 1
        seq_len = T_lat * H_lat * 1 * W_lat // (
            self.pipeline.patch_size[1] * self.pipeline.patch_size[2]
        )
        import math
        seq_len = int(math.ceil(seq_len / self.pipeline.sp_size)) * self.pipeline.sp_size
        return seq_len, T_lat, H_lat, W_lat

    # ------------------------------------------------------------------
    # FG-B: Conditioning cache (built once, reused for all recomputations)
    # ------------------------------------------------------------------

    def _build_conditioning(
        self,
        state_14d: np.ndarray,
        gripper_states: np.ndarray,
        obs_img: Optional[np.ndarray] = None,
        prompt: str = "turn on the switch",
        num_inference_steps: int = 5,
        execution_steps: int = 33,
        shift: float = 1.0,
    ) -> dict:
        """Build and cache ALL conditioning state for sampling and recomputation.

        Called automatically by the first sample_with_logprob() call.
        Subsequent calls to recompute_logprob() reuse the cached state.

        Returns:
            dict with keys:
                z, seq_len, arg_c, video_timestep_packed,
                history_action_state, scheduler, timesteps, sigmas,
                execution_steps
        """
        param_dtype = self.pipeline.param_dtype

        # ---- 1. Text encoding ----
        context = self._encode_text(prompt)

        # ---- 2. Observation encoding ----
        if obs_img is None:
            obs_img = torch.zeros(3, 3, 192, 256, dtype=torch.float32, device=self.device)
        elif isinstance(obs_img, np.ndarray):
            obs_img = torch.from_numpy(obs_img).to(self.device, dtype=torch.float32)
        z, V = self._encode_obs(obs_img)

        # ---- 3. Sequence length ----
        seq_len, T_lat, H_lat, W_lat = self._compute_seq_len(z)

        # ---- 4. Normalize state (same as TauPolicy.play) ----
        state_t = torch.from_numpy(state_14d).float().unsqueeze(0)  # (1, 14)
        grip_t = torch.from_numpy(gripper_states).float().unsqueeze(0)  # (1, 2)

        from adapters.robotwin.rotation_utils import quaternion_to_rotation_6d
        state_rot_l_6d = quaternion_to_rotation_6d(state_t[:, 3:7])
        state_rot_r_6d = quaternion_to_rotation_6d(state_t[:, 10:14])
        state_6d = torch.cat((
            state_t[:, :3], state_rot_l_6d,
            grip_t[:, :1],
            state_t[:, 7:10], state_rot_r_6d,
            grip_t[:, 1:],
        ), dim=-1)  # (1, 20)

        sta_mean_t = torch.tensor(self.policy.sta_mean[None, :])
        sta_std_t = torch.tensor(self.policy.sta_std[None, :])
        history_action_state = ((state_6d - sta_mean_t) / sta_std_t).unsqueeze(0)  # (1, 1, 20)
        history_action_state = history_action_state.to(self.device, dtype=param_dtype)

        # ---- 5. Scheduler and timesteps ----
        from models.wan_2_2_models.scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.pipeline.num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(num_inference_steps, device=self.device, shift=shift)
        timesteps = scheduler.timesteps
        sigmas = scheduler.sigmas.to(self.device)

        # ---- 6. Video timestep (constant 1000 for action-only mode) ----
        video_timestep_packed = torch.full((1, seq_len), 1000, device=self.device, dtype=param_dtype)

        # ---- 7. arg_c for model forward ----
        arg_c = {
            'context': [context[0]],
            'seq_len': seq_len,
        }

        cond = {
            'z': z,
            'seq_len': seq_len,
            'arg_c': arg_c,
            'video_timestep_packed': video_timestep_packed,
            'history_action_state': history_action_state,
            'scheduler': scheduler,
            'timesteps': timesteps,
            'sigmas': sigmas,
            'execution_steps': execution_steps,
        }

        # Cache for later recomputation
        self._cached_cond = cond
        self._cached_video_buffers = None  # reset; will be populated by step 0

        return cond

    # ------------------------------------------------------------------
    # FG-B: Model forward step (shared between sampling and recomputation)
    # ------------------------------------------------------------------

    def _model_forward_step(
        self,
        cond: dict,
        action_states: torch.Tensor,
        timestep: torch.Tensor,
        step_index: int,
        video_states_buffer=None,
        action_context_kv_cache=None,
    ) -> dict:
        """Execute one model forward step with the EXACT τ₀ pipeline calling convention.

        Args:
            cond: conditioning dict from _build_conditioning()
            action_states: (1, execution_steps, 20) current action latent
            timestep: scalar tensor — current flow timestep value
            step_index: 0-based step index (0 = first step, builds video buffers)
            video_states_buffer: cached video hidden states (None on step 0, reused after)
            action_context_kv_cache: cached action cross-attn KV (None on step 0)

        Returns:
            dict with keys: 'action' (velocity_pred), 'video_states_buffer',
                            'action_context_kv_cache'
        """
        execution_steps = cond['execution_steps']
        param_dtype = self.pipeline.param_dtype

        compute_video = (step_index == 0)   # video forward only on first step
        store_buffer = (step_index == 0)    # store video buffer only on first step

        latent_model_input = [cond['z'][0].to(self.device)]
        action_timestep = torch.full(
            (1, execution_steps), timestep.item(),
            device=self.device, dtype=param_dtype
        )

        with torch.amp.autocast('cuda', dtype=param_dtype):
            noise_pred = self.pipeline.model(
                latent_model_input,
                cond['video_timestep_packed'],
                action_states=action_states,
                action_timestep=action_timestep,
                return_video=compute_video,
                return_action=True,
                store_buffer=store_buffer,
                video_states_buffer=video_states_buffer,
                action_context_kv_cache=action_context_kv_cache,
                history_action_state=cond['history_action_state'],
                **cond['arg_c'],
            )

        return {
            'action': noise_pred['action'],  # (1, execution_steps, 20) — velocity_pred
            'video_states_buffer': noise_pred.get('video_states_buffer'),
            'action_context_kv_cache': noise_pred.get('action_context_kv_cache'),
        }

    # ------------------------------------------------------------------
    # Sampling (refactored to use _build_conditioning + _model_forward_step)
    # ------------------------------------------------------------------

    def sample_with_logprob(
        self,
        state_14d: np.ndarray,
        gripper_states: np.ndarray,
        obs_img: Optional[np.ndarray] = None,
        prompt: str = "turn on the switch",
        num_inference_steps: int = 5,
        execution_steps: int = 33,
        seed: int = 200,
        generator: Optional[torch.Generator] = None,
        shift: float = 1.0,
        return_velocities: bool = False,
        noise_scale: float = 1.0,
    ) -> dict:
        """Generate one action trajectory with SDE logprobs.

        Args:
            state_14d: robot state (14,) — xyz+quat_xyzw for left/right arms
            gripper_states: gripper states (2,)
            obs_img: observation image (V, C, H, W) or None for dummy
            prompt: text instruction
            num_inference_steps: number of flow steps (default 5)
            execution_steps: number of action steps to generate (default 33)
            seed: random seed
            generator: torch Generator
            shift: flow scheduler shift parameter
            noise_scale: V4-B6 exploration knob — SDE transition noise temperature (1.0 default)

        Returns:
            dict with:
                'action': final action, shape (33, 20), normalized
                'all_latents': list of tensors [x_1, x_σ2, ..., x_0]
                'all_log_probs': list of tensors [logp_1, ..., logp_5]
                'timesteps': tensor of scheduler timesteps
                'sigmas': sigma values
        """
        if generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        # ---- Build/cache conditioning (shared with recompute) ----
        cond = self._build_conditioning(
            state_14d=state_14d,
            gripper_states=gripper_states,
            obs_img=obs_img,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            execution_steps=execution_steps,
            shift=shift,
        )
        param_dtype = self.pipeline.param_dtype
        timesteps = cond['timesteps']
        sigmas = cond['sigmas']

        # ---- Initialize action noise ----
        noise_action = torch.randn(
            1, execution_steps, 20,
            dtype=param_dtype, generator=generator, device=self.device
        )

        # ---- SDE Action Flow Loop ----
        action_states = noise_action  # x_σ at σ=1
        all_latents = [action_states.clone()]
        all_log_probs = []
        all_velocities = [] if return_velocities else None

        video_states_buffer = None
        action_context_kv_cache = None

        for i, t in enumerate(timesteps):
            # Model forward (reuses extracted method)
            with torch.no_grad():
                fwd_out = self._model_forward_step(
                    cond=cond,
                    action_states=action_states,
                    timestep=t,
                    step_index=i,
                    video_states_buffer=video_states_buffer,
                    action_context_kv_cache=action_context_kv_cache,
                )

            velocity_pred = fwd_out['action']

            if return_velocities:
                all_velocities.append(velocity_pred.clone().detach())

            if i == 0:
                video_states_buffer = fwd_out['video_states_buffer']
                action_context_kv_cache = fwd_out['action_context_kv_cache']
                # Cache for recomputation
                self._cached_video_buffers = (
                    video_states_buffer,
                    action_context_kv_cache,
                )

            # ---- SDE step with logprob ----
            prev_action, log_prob, prev_mean, trans_std = sde_step_with_logprob(
                sigmas=sigmas,
                timesteps=timesteps,
                model_output=velocity_pred,
                timestep=t,
                sample=action_states,
                prev_sample=None,  # sample fresh SDE noise
                generator=generator,
                deterministic=False,
                return_dt_and_std_dev_t=False,
                noise_scale=noise_scale,
            )

            all_latents.append(prev_action.clone())
            all_log_probs.append(log_prob)

            action_states = prev_action

        # ---- Final action ----
        final_action = action_states.squeeze(0)  # (33, 20)

        result = {
            'action': final_action,
            'all_latents': all_latents,
            'all_log_probs': all_log_probs,
            'timesteps': timesteps,
            'sigmas': sigmas,
        }
        if return_velocities:
            result['all_velocities'] = all_velocities
        return result

    # ------------------------------------------------------------------
    # FG-B: Recomputation methods (replay model forward through cached cond)
    # ------------------------------------------------------------------

    def recompute_logprob(
        self,
        x_t: torch.Tensor,
        x_next: torch.Tensor,
        timestep: torch.Tensor,
        step_index: int,
        enable_grad: bool = False,
    ):
        """Recompute log-probability for one SDE transition using cached pipeline state.

        Replays the EXACT model forward through the same conditioning (VAE encoding,
        text, state normalization, video buffers, KV cache) as the sampling phase.
        Then feeds the velocity prediction into sde_step_with_logprob.

        CRITICAL: _build_conditioning() must have been called first (via
        sample_with_logprob()). The cached cond and video buffers from the sampling
        step 0 are reused to guarantee byte-identical conditioning.

        Args:
            x_t: (T, D) or (1, T, D) — action latent BEFORE the SDE step
            x_next: (T, D) or (1, T, D) — action latent AFTER the SDE step
            timestep: scalar tensor — flow timestep value for this step
            step_index: 0-based step index (determines video compute/store flags)
            enable_grad: if True, run without torch.no_grad() for gradient flow

        Returns:
            (log_prob, velocity_pred) where:
                log_prob: scalar tensor — log p(x_next | x_t)
                velocity_pred: (1, T, D) — fresh model velocity prediction
        """
        if self._cached_cond is None:
            raise RuntimeError(
                "No cached conditioning. Call sample_with_logprob() first to "
                "populate _cached_cond and _cached_video_buffers."
            )

        cond = self._cached_cond
        sigmas = cond['sigmas']
        timesteps = cond['timesteps']

        # Ensure x_t is (1, T, D)
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(0)
        if x_next.dim() == 2:
            x_next = x_next.unsqueeze(0)

        # Get cached video buffers for step > 0
        video_states_buffer = None
        action_context_kv_cache = None
        if self._cached_video_buffers is not None:
            video_states_buffer, action_context_kv_cache = self._cached_video_buffers

        # Model forward
        def _forward():
            return self._model_forward_step(
                cond=cond,
                action_states=x_t,
                timestep=timestep,
                step_index=step_index,
                video_states_buffer=video_states_buffer,
                action_context_kv_cache=action_context_kv_cache,
            )

        if enable_grad:
            fwd_out = _forward()
        else:
            with torch.no_grad():
                fwd_out = _forward()

        velocity_pred = fwd_out['action']

        # SDE step with logprob (using prev_sample=x_next for the given trajectory)
        _, log_prob, prev_mean, trans_std = sde_step_with_logprob(
            sigmas=sigmas,
            timesteps=timesteps,
            model_output=velocity_pred,
            timestep=timestep,
            sample=x_t,
            prev_sample=x_next,  # use the actual x_next from sampling
            generator=None,
            deterministic=False,
            return_dt_and_std_dev_t=False,
        )

        return log_prob, velocity_pred

    def recompute_trajectory_logprobs(
        self,
        traj,  # TauTrajectory
        enable_grad: bool = False,
    ):
        """Recompute log-probabilities for all L steps in a trajectory.

        Uses cached pipeline conditioning and video buffers. Each step replays
        the model forward with the exact same inputs as the sampling phase.

        Args:
            traj: TauTrajectory with latents (L, T, D), timesteps (L,), etc.
            enable_grad: if True, run without torch.no_grad() for gradient flow

        Returns:
            (log_probs, recomputed_velocities) where:
                log_probs: (L,) tensor of recomputed log-probabilities
                recomputed_velocities: (L, T, D) tensor of fresh velocity predictions
        """
        if self._cached_cond is None:
            raise RuntimeError(
                "No cached conditioning. Call sample_with_logprob() first."
            )

        L = len(traj.timesteps)
        log_probs_list = []
        velocities_list = []

        for i in range(L):
            x_t = traj.latents[i]       # (T, D)
            x_next = traj.next_latents[i]  # (T, D)
            t = traj.timesteps[i]       # scalar

            log_prob, velocity_pred = self.recompute_logprob(
                x_t=x_t,
                x_next=x_next,
                timestep=t,
                step_index=i,
                enable_grad=enable_grad,
            )

            # Ensure scalar
            if log_prob.ndim > 0:
                log_prob = log_prob.flatten()[0]
            log_probs_list.append(log_prob)

            velocities_list.append(velocity_pred.squeeze(0).clone().detach())

        log_probs = torch.stack(log_probs_list)  # (L,)
        velocities = torch.stack(velocities_list)  # (L, T, D)

        return log_probs, velocities

    # ------------------------------------------------------------------
    # Utility: denormalize action to physical space
    # ------------------------------------------------------------------

    def denormalize_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized action back to physical eef6d space.

        Args:
            action_norm: (T, 20) normalized action

        Returns:
            (T, 20) physical eef6d action
        """
        act_mean = torch.tensor(
            self.policy.act_mean, device=action_norm.device, dtype=torch.float32
        )
        act_std = torch.tensor(
            self.policy.act_std, device=action_norm.device, dtype=torch.float32
        )
        # act_mean/act_std are (1, 1, C) — squeeze to (C,)
        if act_mean.dim() == 3:
            act_mean = act_mean.squeeze(0).squeeze(0)
            act_std = act_std.squeeze(0).squeeze(0)
        return action_norm.float() * act_std + act_mean

    # ------------------------------------------------------------------
    # Native UniPC sampling (non-SDE baseline, for sanity checks)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_unipc(
        self,
        state_14d: np.ndarray,
        gripper_states: np.ndarray,
        obs_img: Optional[np.ndarray] = None,
        prompt: str = "turn on the switch",
        num_inference_steps: int = 5,
        execution_steps: int = 33,
        seed: int = 200,
        shift: float = 1.0,
    ) -> torch.Tensor:
        """Generate one action trajectory using native UniPC (deterministic, no logprob).

        This is the non-SDE baseline for sanity checks — verifies the pipeline
        still works after refactoring.

        Returns:
            action: (33, 20) normalized action
        """
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        cond = self._build_conditioning(
            state_14d=state_14d,
            gripper_states=gripper_states,
            obs_img=obs_img,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            execution_steps=execution_steps,
            shift=shift,
        )
        param_dtype = self.pipeline.param_dtype
        timesteps = cond['timesteps']
        scheduler = cond['scheduler']

        # Initialize action noise
        noise_action = torch.randn(
            1, execution_steps, 20,
            dtype=param_dtype, generator=generator, device=self.device
        )
        action_states = noise_action

        video_states_buffer = None
        action_context_kv_cache = None

        for i, t in enumerate(timesteps):
            fwd_out = self._model_forward_step(
                cond=cond,
                action_states=action_states,
                timestep=t,
                step_index=i,
                video_states_buffer=video_states_buffer,
                action_context_kv_cache=action_context_kv_cache,
            )
            velocity_pred = fwd_out['action']

            if i == 0:
                video_states_buffer = fwd_out['video_states_buffer']
                action_context_kv_cache = fwd_out['action_context_kv_cache']

            # Native UniPC step (deterministic)
            action_states = scheduler.step(
                velocity_pred, t, action_states, return_dict=False
            )[0]

        return action_states.squeeze(0)  # (33, 20)


def sample_k_trajectories(
    pipeline_wrapper: TauPipelineWithLogprob,
    state_14d: np.ndarray,
    gripper_states: np.ndarray,
    k: int = 4,
    base_seed: int = 200,
    num_inference_steps: int = 5,
    return_velocities: bool = False,
    **kwargs,
) -> list[dict]:
    """Generate K SDE trajectories for one observation.

    Each trajectory uses a different random seed for different SDE noise.

    Args:
        pipeline_wrapper: TauPipelineWithLogprob instance
        state_14d: robot state (14,)
        gripper_states: gripper states (2,)
        k: number of trajectories
        base_seed: base random seed (each candidate gets base_seed * 1000 + k)
        num_inference_steps: flow steps

    Returns:
        list of k result dicts (each has action, all_latents, all_log_probs, timesteps)
    """
    results = []
    for i in range(k):
        seed = base_seed * 1000 + i
        gen = torch.Generator(device=pipeline_wrapper.device)
        gen.manual_seed(seed)

        result = pipeline_wrapper.sample_with_logprob(
            state_14d=state_14d,
            gripper_states=gripper_states,
            seed=seed,
            generator=gen,
            num_inference_steps=num_inference_steps,
            return_velocities=return_velocities,
            **kwargs,
        )
        result['k_idx'] = i
        result['seed'] = seed
        results.append(result)

    return results
