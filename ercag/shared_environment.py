"""F_env — action-independent shared environment branch.

Runs the native WanModel video backbone with ZERO action conditioning and returns the
final hidden representation ``Z_env`` (the transformer hidden state before the video head),
together with the head inputs ``e`` (time embedding) and ``grid_sizes`` so the formal future
``Z_i = Z_env + DeltaZ_i`` can be decoded through the SAME native head/unpatchify downstream.

Requires one minimal Tau source hook: ``model_sim.py`` exposes ``head_inputs = (e, grid_sizes)``
in ``store_buffer`` mode (documented in V4-B report). No other Tau change.

Word fidelity:
    - ACTION ENTERS AT: nowhere in F_env — ``action_states`` is fixed to zeros and is
      identical for every candidate, so F_env carries no candidate-action information.
    - ENV SHARED UP TO: the full 30-block video backbone (action-free modulation). All
      candidate/reference futures share this single Z_env (computed once per group).
"""
import torch
import torch.nn as nn


class SharedEnvironment(nn.Module):
    def __init__(self, backbone: nn.Module, act_cond_in_dim: int = 26,
                 action_chunk: int = 33, n_obs_frames: int = 3):
        super().__init__()
        self.backbone = backbone
        self.act_cond_in_dim = act_cond_in_dim
        self.action_chunk = action_chunk
        self.n_obs_frames = n_obs_frames

    def forward(self, x_t, t, context, seq_len, n_mem=0):
        r"""Compute the action-independent shared environment representation + Z_t.

        Args:
            x_t (list[Tensor]): noisy patch latents, one ``[C, F, H, W]`` per batch item
                (constructed from the SAME shared noise xi for the whole candidate group).
                The leading ``n_obs_frames`` frames are the CLEAN observation (never noised).
            t (Tensor): diffusion timestep, ``[B]`` (expanded internally) or ``[B, seq_len]``.
            context (list[Tensor]): text embeddings, one ``[L, text_dim]`` per batch item.
            seq_len (int): max sequence length for positional encoding.
            n_mem (int): number of memory frames.

        Returns:
            dict:
                - ``z_env`` (Tensor): ``[B, L, dim]`` final hidden, action-independent.
                - ``e`` (Tensor): ``[B, L, dim]`` head time embedding (float32).
                - ``grid_sizes`` (Tensor): patch grid ``[B, 3]`` for unpatchify.
                - ``z_state`` (Tensor): ``[B, n_obs*spatial, dim]`` deterministic state
                  representation Z_t (obs-token patch embeddings; action- AND xi-independent).
        """
        B = len(x_t)
        action_states = torch.zeros(
            B, self.action_chunk, self.act_cond_in_dim,
            dtype=x_t[0].dtype, device=x_t[0].device,
        )
        out = self.backbone(
            x=x_t,
            t=t,
            context=context,
            seq_len=seq_len,
            action_states=action_states,
            return_video=False,
            return_reward=False,
            store_buffer=True,
            n_mem=n_mem,
        )
        # final hidden = output of the last transformer block (before the video head)
        z_env = out["video_states_buffer"][-1]
        e, grid_sizes = out["head_inputs"]
        # deterministic state representation Z_t = obs-token patch embeddings
        z_state = self._state_repr(x_t)
        return {"z_env": z_env, "e": e, "grid_sizes": grid_sizes, "z_state": z_state}

    def _state_repr(self, x_t):
        r"""Deterministic observation/history encoding Z_t.

        Z_t = ``backbone.patch_embedding`` applied to the CLEAN observation frames (frames
        ``0 .. n_obs_frames-1`` of ``x_t``). These frames are the mem latent, never noised by
        xi and never conditioned on the candidate action (action enters the backbone ONLY via
        the time-embedding modulation ``e += action_states_e``, which the patch embedding
        bypasses). So Z_t is action-independent AND xi-independent by construction — the native
        deterministic state representation (§6-7: reuse the native embedding, no new encoder).

        Returns:
            Tensor: ``[B, n_obs_frames*spatial, dim]`` (frame-major token order, matches
                ``z_env``'s leading observation tokens).
        """
        zs = []
        for u in x_t:
            obs_u = u[:, :self.n_obs_frames, :, :]                  # [C, n_obs, H, W] clean
            p = self.backbone.patch_embedding(obs_u.unsqueeze(0))   # [1, dim, n_obs, H', W']
            p = p.flatten(2).transpose(1, 2)                        # [1, n_obs*spatial, dim]
            zs.append(p)
        return torch.cat(zs, dim=0)                                 # [B, n_obs*spatial, dim]
