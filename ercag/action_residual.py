"""F_act — state-conditioned ACTION-DIFFERENTIATED residual branch (ActionResidualAdapter, V4-B5).

V4-B4 localized the V4-B3 collapse: the old ``DeltaZ_i = ResidualProjection(action_feat * (1 +
state_feat))`` with ``action_feat = temporal_proj(mean_11(action_proj(a_i)))`` drowned the action
signal. Two structural bugs:

1. **33 -> 3 mean aggregation** — averaging 11 control steps washes out *which* step matters, so
   distinct SDE candidates produced near-collinear action_feat (cos ~0.98).
2. **shared multiplicative state gate** — ``action_feat * (1 + state_feat)`` adds a large
   candidate-INDEPENDENT component (state_feat is identical for every candidate), leaving ~0.5%
   per-action energy in the residual.

V4-B5 (Word §16-§20) replaces this with an additive, action-identity-preserving fusion while
keeping the formal ``DeltaZ_i = F_act(Z_t, a_i)``:

    a_base      = ActionTemporalEncoder(a_i)                       # [B, 3, dim] action-specific
    state_corr  = StateActionCorrection(Z_t, a_base)               # final layer non-zero init (V4-B15)
    DeltaZ      = ActionProjection(a_base) + StateCorrectionProjection(state_corr)
                                                                   # ^ identity-init   ^ zero-init

At init ``ActionProjection == I`` and ``StateCorrectionProjection == 0``, so ``DeltaZ == a_base`` — a
purely action-specific residual with an explicit action identity bypass (Word §18). The state branch is a
zero-init *correction* projection, learned gradually, and can never swamp the action signal at init.

V4-B15: the final ``state_corr`` layer was previously zero-init too; together with the zero-init
``state_out`` that made the two layers zero each other's gradient (mutual zero-gradient fixed point), so
the state branch could never train. The final ``state_corr`` layer is now left at its default non-zero
init (bias still zero) so the state branch receives gradient while step-0 output stays ``a_base``.

ActionTemporalEncoder reuses the NATIVE Tau per-step action embedding (Word §8, §11, §13):

    a_emb = backbone.act_cond_proj_in(a_i)      # pretrained Linear(26 -> 96 -> 96), per-step tokens
    a_emb = rope_apply(a_emb, act_cond_freqs)   # native per-step rotary position
    a     = action_step_proj(a_emb)             # 96 -> dim (per-step, no pooling yet)
    a_base= temporal_weighted_pool(a)           # 33 -> 3, learnable weights aligned to
                                                 # FRAME_CONTROL_MAPPING {1:7, 2:19, 3:31}

The 33->3 mapping is a LEARNED weighted pool (softmax over the 33 steps, initialized as Gaussians
centered at the control indices 7/19/31) — NOT a mean. It preserves per-step action structure and
lets the model attend to the steps that actually move the switch. If the native modules are not
provided (isolated structural tests), a self-contained fallback embedding is used.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_temporal_logits(n_action_steps, n_future_frames, centers, sigma):
    """Gaussian init concentrated at FRAME_CONTROL_MAPPING control indices (NOT a uniform mean)."""
    pos = torch.arange(n_action_steps, dtype=torch.float32)
    centers = torch.tensor(centers, dtype=torch.float32)
    return -((pos[None, :] - centers[:, None]) / sigma) ** 2   # [n_future_frames, n_steps]


class ActionResidualAdapter(nn.Module):
    def __init__(self, dim: int, act_cond_in_dim: int = 26,
                 n_action_steps: int = 33, n_obs_frames: int = 3,
                 n_future_frames: int = 3, spatial_per_frame: int = 144,
                 act_embed_dim: int = 96,
                 native_action_embed: nn.Module = None,
                 act_cond_freqs: torch.Tensor = None,
                 frame_control_mapping: tuple = (7, 19, 31),
                 temporal_sigma: float = 1.0,
                 readout: str = "global_mean",
                 state_spatial_grid: tuple = None):
        """State-conditioned, action-differentiated residual (additive fusion + identity bypass).

        Args:
            dim: environment/token hidden dim (3072).
            act_cond_in_dim: action step dim (26 = 20 rel-eef6d + 6 whole_body).
            n_action_steps: action chunk length (33).
            n_obs_frames: latent observation frames (3).
            n_future_frames: latent future frames (3).
            spatial_per_frame: tokens per latent frame (144).
            act_embed_dim: per-step action embedding width (96, matches native act_cond_proj_in).
            native_action_embed: the backbone's pretrained ``act_cond_proj_in`` (reused verbatim);
                if None, a self-contained fallback is built (isolated tests only).
            act_cond_freqs: the backbone's native per-step RoPE freqs (``act_cond_freqs``), optional.
            frame_control_mapping: future frame (1-based) -> action control index (qpos trace).
            temporal_sigma: Gaussian width for the temporal-pooling init. V4-B5 set this to 1.0
                (was 6.0): sigma=6 spreads weight over ~13 steps and re-introduces the "33->3
                mean" averaging that §10 forbids (a_base cos 0.979, specific_fraction 2.0%).
                sigma=1.0 concentrates on the control index +/- ~2 steps (a_base cos 0.924,
                specific_fraction 7.3%).
            readout: state readout mode. "global_mean" (default, V4-B15) mean-pools z_state to
                [B, 3, dim] before the state_corr interaction. "structured" (V4-B16) adaptively
                2x2-pools each frame into 4 coarse spatial tokens, runs the shared
                state_corr/state_out interaction PER TOKEN, then means the corrections AFTER the
                interaction (move the mean from before to after the state_corr/state_out).
                "vector" (V4-B19) treats z_state as a single causal state vector Z_t_enc
                [B, dim] (the output of a CausalStateEncoder), broadcasts it over the 3 future
                frames, and feeds state_corr/state_out as in global_mean.
            state_spatial_grid: (H', W') latent spatial grid per frame (e.g. (6, 24)) required by
                the "structured" readout to reshape [B, n_obs*spatial, dim] -> [B, F, H', W', dim].
                None means global-mean readout only.
        """
        super().__init__()
        self.dim = dim
        self.n_action_steps = n_action_steps
        self.n_obs_frames = n_obs_frames
        self.n_future_frames = n_future_frames
        self.spatial_per_frame = spatial_per_frame
        self.act_embed_dim = act_embed_dim
        self.readout = readout
        self.state_spatial_grid = state_spatial_grid
        if readout == "structured" and state_spatial_grid is None:
            raise ValueError("structured readout requires state_spatial_grid=(H', W')")

        # ---- (A) native per-step action embedding (reused pretrained, or fallback) ----
        if native_action_embed is not None:
            self.native_action_embed = native_action_embed       # pretrained act_cond_proj_in
            self.act_cond_freqs = act_cond_freqs                 # RoPE freqs (optional)
        else:
            self.native_action_embed = nn.Sequential(
                nn.Linear(act_cond_in_dim, act_embed_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(act_embed_dim, act_embed_dim),
            )
            self.act_cond_freqs = None

        # ---- (B) per-step projection to dim (keep per-step structure, no pooling yet) ----
        # V4-B5 fix: bias=False. The default bias is a candidate-INDEPENDENT constant
        # (~5.6x the weight contribution) that drowns the per-action differences: with bias
        # the per-step cosine contracted 0.841 -> 0.960, without it 0.841 -> 0.840.
        self.action_step_proj = nn.Linear(act_embed_dim, dim, bias=False)

        # ---- (C) learned temporal weighted pooling 33 -> 3 (Gaussian init at control indices) ----
        assert len(frame_control_mapping) == n_future_frames
        self.temporal_logits = nn.Parameter(
            _init_temporal_logits(n_action_steps, n_future_frames,
                                  frame_control_mapping, temporal_sigma))

        # ---- (D) ActionProjection: explicit action identity bypass (identity-init) ----
        self.action_out = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.action_out.weight)

        # ---- (E) StateActionCorrection: state+action -> correction ----
        self.state_corr = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        # V4-B15 FIX (BREAK MUTUAL ZERO INITIALIZATION): the final state_corr layer was zero-init,
        # which — in series with the zero-init state_out below — made each zero the other's gradient
        # (a mutual zero-gradient fixed point => the state branch never trained, i.e. F_ACT_IGNORES_STATE).
        # Keep this layer's weight at the default nn.Linear init (non-zero, same scheme as state_corr[0])
        # so state_out receives a first-step gradient; keep the bias zero. state_out stays zero-init so
        # the step-0 correction remains exactly 0 (forward-compatible with the old init).
        nn.init.zeros_(self.state_corr[-1].bias)

        # ---- (F) StateCorrectionProjection: zero-init (correction == 0 at init) ----
        self.state_out = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.state_out.weight)

        # diagnostic: last state-correction norm (shared for all candidates at a given Z_t)
        self._last_state_corr_norm = 0.0
        # diagnostic: per-structured-token corrections [B, M, 3, dim] (structured readout only)
        self._last_structured_tokens = None

    def _rope(self, a, dtype):
        """Native per-step rotary position (lazy import; skipped when freqs is None)."""
        from models.wan_2_2_models.transformers.model_sim import rope_apply
        freqs = self.act_cond_freqs.to(device=a.device)
        grid = torch.stack([
            torch.tensor([a.shape[1]], dtype=torch.long, device=a.device)
            for _ in range(a.shape[0])])
        a = rope_apply(a.unsqueeze(dim=-2), grid, freqs)         # [B, 33, 1, act_embed_dim] float32
        return a.squeeze(dim=-2).to(dtype=dtype)                 # [B, 33, act_embed_dim]

    def encode_action_steps(self, action):
        """Per-step action tokens [B, 33, dim] (native embed -> rope -> per-step proj)."""
        a = self.native_action_embed(action)                     # [B, 33, act_embed_dim]
        if self.act_cond_freqs is not None:
            a = self._rope(a, action.dtype)
        return self.action_step_proj(a)                          # [B, 33, dim]

    def encode_action(self, action):
        """ActionTemporalEncoder -> a_base [B, 3, dim] (learned temporal weighted pool)."""
        a = self.encode_action_steps(action)                     # [B, 33, dim]
        w = F.softmax(self.temporal_logits, dim=1)               # [3, 33]
        return torch.einsum("hf,bfd->bhd", w, a)                 # [B, 3, dim]

    def structured_pool(self, z_state):
        """[B, n_obs*spatial, dim] -> [B, M, dim] coarse 2x2 per-frame tokens (frame-major, cell row-major).

        Recovers the latent spatial grid (F, H', W') from the frame-major flatten order used by the
        tau patch embedding (`patch_embedding(obs).flatten(2).transpose(1, 2)` => t = f*H'*W' + h*W' + w),
        then adaptive-average-pools each frame to a fixed 2x2 grid. M = n_obs_frames * 4.
        """
        H, W = self.state_spatial_grid
        B, nF, D = z_state.shape[0], self.n_obs_frames, self.dim
        x = z_state.reshape(B, nF, H, W, D)                 # [B, F, H, W, D]
        x = x.reshape(B * nF, H, W, D).permute(0, 3, 1, 2)  # [B*F, D, H, W]
        x = F.adaptive_avg_pool2d(x, (2, 2))                # [B*F, D, 2, 2]
        x = x.permute(0, 2, 3, 1).reshape(B, nF, 2, 2, D)   # [B, F, 2, 2, D]
        return x.reshape(B, nF * 4, D)                      # [B, M, D]

    def _structured_correction(self, z_state, a_base, B):
        """Per-token state_corr/state_out, then mean over the M structured tokens (V4-B16 readout).

        The SAME state_corr/state_out/action branches are reused — the only change is moving the
        spatial mean from BEFORE the interaction to AFTER it: each coarse token z_k (broadcast over
        the 3 future frames, concatenated with the shared a_base) produces c_k = state_out(state_corr(
        [z_k; a_base])), and state_correction = mean_k(c_k).
        """
        z_struct = self.structured_pool(z_state)           # [B, M, D]
        M = z_struct.shape[1]
        z_exp = z_struct.unsqueeze(2).expand(-1, -1, self.n_future_frames, -1)  # [B, M, 3, D]
        a_exp = a_base.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, 3, D]
        h = self.state_corr(torch.cat([z_exp, a_exp], dim=-1))  # [B, M, 3, D]
        c = self.state_out(h)                              # [B, M, 3, D]
        self._last_structured_tokens = c.detach()
        return c.mean(dim=1)                               # [B, 3, D]

    def forward(self, z_env: torch.Tensor, action: torch.Tensor,
                z_state: torch.Tensor = None):
        r"""State-conditioned action-differentiated residual.

        Args:
            z_env: ``[B, L, dim]`` shared environment representation (context only).
            action: ``[B, n_action_steps, act_cond_in_dim]`` normalized action chunk.
            z_state: ``[B, n_obs*spatial, dim]`` deterministic state repr Z_t (optional).

        Returns:
            Tensor ``[B, L, dim]`` — zero on observation tokens, per-future-frame residual
            broadcast over spatial tokens.
        """
        B, L, D = z_env.shape
        assert D == self.dim, (D, self.dim)
        n_total_frames = self.n_obs_frames + self.n_future_frames
        assert L % n_total_frames == 0
        spatial = L // n_total_frames
        n_obs_tokens = self.n_obs_frames * spatial

        a_base = self.encode_action(action)                      # [B, 3, dim] action-specific
        action_residual = self.action_out(a_base)                # [B, 3, dim] identity @ init

        if z_state is not None:
            if self.readout == "vector":
                # V4-B19: z_state is Z_t_enc [B, dim] — a single causal state vector (Encoder output).
                assert z_state.shape[1] == self.dim, (z_state.shape, self.dim)
                s = z_state.unsqueeze(1).expand(-1, self.n_future_frames, -1)      # [B, 3, dim]
                state_corr = self.state_corr(torch.cat([s, a_base], dim=-1))       # [B, 3, dim]
                correction = self.state_out(state_corr)              # [B, 3, dim] zero @ init
            else:
                assert z_state.shape[1] == self.n_obs_frames * spatial, \
                    (z_state.shape, self.n_obs_frames * spatial)
                if self.readout == "structured":
                    correction = self._structured_correction(z_state, a_base, B)   # [B, 3, dim]
                else:
                    s = z_state.reshape(B, self.n_obs_frames, spatial, D).mean(dim=2)  # [B, 3, dim]
                    state_corr = self.state_corr(torch.cat([s, a_base], dim=-1))       # [B, 3, dim]
                    correction = self.state_out(state_corr)              # [B, 3, dim] zero @ init
            self._last_state_corr_norm = float(correction.detach().float().norm())
            delta = action_residual + correction                 # additive fusion
        else:
            self._last_state_corr_norm = 0.0
            delta = action_residual                              # action-only path

        obs = z_env.new_zeros(B, n_obs_tokens, D)
        future = delta.repeat_interleave(spatial, dim=1)         # [B, 3*spatial, D]
        return torch.cat([obs, future], dim=1)                   # [B, L, D]
