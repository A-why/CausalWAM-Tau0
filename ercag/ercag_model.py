"""ERCAGModel — Shared Environment + Action Residual ER-CAG world/value model.

FORMAL single-path architecture (V4-B6, exact reference-zero path + state-conditioned residual):

    Z_env  = F_env(Z_t, xi_t)            (native backbone, zero action, store_buffer)
    DeltaZ_i = F_act(Z_t, a_i)           (state-conditioned ActionResidualAdapter)
    Z_i     = Z_env + DeltaZ_i            (additive fusion H)          <- candidate ONLY
    Z_0     = H(Z_env, 0) = Z_env          (FORMAL reference: EXACT zero residual)
    DeltaZ_hold = F_act(Z_t, a_0)         (LEARNED hold residual; L_zero ONLY, never in Z_0)
    v_i / v_0 = head(Z_i / Z_0, e)       (SAME native downstream as Tau)
    V_i / V_0 = ValueHead(Z_i / Z_0)      (shared per-future-frame value [B, H])
    G_hat   = V_i - V_0                    ([B, H] short-horizon gain)

V4-B6 Word-fidelity correction (§2-§11): the reference future is ``Z_0 = H(Z_env, 0)`` — the
action residual is EXACT zero, so ``L_ref`` and the reference value term ``MSE(V_0, p_r)``
supervise F_env directly and receive ZERO gradient w.r.t. F_act. The hold action ``a_0`` is only
ever routed through ``F_act`` to produce ``hold_residual``, which feeds L_zero (the structural
constraint ``F_act(Z_t, a_0) ≈ 0``) — it is NEVER added back into the reference future.

The world future prediction (v_i / v_0) and the per-frame value (V_i / V_0) BOTH consume the
SAME formal future representation Z_i / Z_0. There is no separate native action-conditioned
world path — the action enters the world prediction ONLY through DeltaZ_i (Word §2-6).

Formal API:
    out = model.forward_pair(x_t, t, context, seq_len, candidate_action=a_i,
                             reference_action=a_0)
    # -> z_env, z_state, e, grid_sizes, candidate_residual, reference_residual (=0),
    #    hold_residual, candidate_future, reference_future (=z_env),
    #    v_cand, v_ref, V_i, V_0, G_hat

    out = model.forward_group(x_t, t, context, seq_len, actions=actions[K], reference_action=a_0)

Shared randomness: x_t is the noisy latent constructed ONCE from a single shared flow noise
xi, so the whole candidate group shares the same xi; F_env runs once per group.
No training logic here; optimizer.step is the caller's responsibility.
"""
import torch
import torch.nn as nn

from .shared_environment import SharedEnvironment
from .action_residual import ActionResidualAdapter
from .value_head import ValueHead


class ERCAGModel(nn.Module):
    def __init__(self, backbone: nn.Module, dim: int, act_cond_in_dim: int = 26,
                 action_chunk: int = 33, value_hidden_dim: int = 1024,
                 n_obs_frames: int = 3, n_future_frames: int = 3):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.act_cond_in_dim = act_cond_in_dim
        self.action_chunk = action_chunk
        self.n_obs_frames = n_obs_frames
        self.n_future_frames = n_future_frames

        self.f_env = SharedEnvironment(backbone, act_cond_in_dim, action_chunk,
                                       n_obs_frames=n_obs_frames)
        # V4-B5: reuse the NATIVE per-step action embedding + RoPE freqs for F_act (Word §8/§11).
        # If the backbone lacks them (mini test backbones), F_act falls back to a self-contained
        # embedding, so structural tests still run in isolation.
        self.f_act = ActionResidualAdapter(
            dim, act_cond_in_dim, n_action_steps=action_chunk,
            n_obs_frames=n_obs_frames, n_future_frames=n_future_frames,
            native_action_embed=getattr(backbone, "act_cond_proj_in", None),
            act_cond_freqs=getattr(backbone, "act_cond_freqs", None))
        self.value_head = ValueHead(dim, n_obs_frames=n_obs_frames,
                                    n_future_frames=n_future_frames,
                                    hidden_dim=value_hidden_dim)

    # ---- environment path (action-independent) -------------------------------
    def forward_env(self, x_t, t, context, seq_len, n_mem=0):
        """Shared environment representation + head inputs + Z_t (computed once per group)."""
        return self.f_env(x_t, t, context, seq_len, n_mem=n_mem)

    # ---- native downstream (head + unpatchify), reused verbatim --------------
    def _decode(self, z, e, grid_sizes):
        """Map a future hidden state through the SAME native head/unpatchify -> velocity."""
        x = self.backbone.head(z, e)
        return [u.float() for u in self.backbone.unpatchify(x, grid_sizes)]

    # ---- value path (shared env + state-conditioned action residual) ---------
    def _value_path(self, z_env, action, z_state=None):
        delta = self.f_act(z_env, action, z_state)
        z_future = z_env + delta
        v = self.value_head(z_future)          # [B, H] per-future-frame value
        return delta, z_future, v

    # ---- FORMAL reference-zero path (V4-B6 §2-§11) --------------------------
    def _reference_zero_path(self, z_env):
        r"""Reference future with EXACT zero action residual.

        ``Z_0 = H(Z_env, 0) = Z_env``. The hold action ``a_0`` is NOT routed through this path;
        ``F_act(Z_t, a_0)`` is computed separately (``hold_residual``) for L_zero only. This keeps
        ``L_ref`` and ``MSE(V_0, p_r)`` free of any F_act gradient.
        """
        v = self.value_head(z_env)             # [B, H] per-future-frame value
        return z_env, v

    def forward_pair(self, x_t, t, context, seq_len, candidate_action, reference_action,
                     n_mem=0):
        r"""Candidate/reference pair sharing one Z_env, one Z_t and one xi (baked into x_t).

        Returns:
            dict: z_env, z_state, e, grid_sizes, candidate_residual, reference_residual (=0),
            hold_residual, candidate_future, reference_future (=z_env), v_cand, v_ref, V_i, V_0,
            G_hat.
            v_cand/v_ref are lists of ``[C_out, F, H, W]`` velocity tensors decoded
            from the FORMAL futures Z_i / Z_0 (not the native action path).
            V_i/V_0/G_hat are PER-FUTURE-FRAME ``[B, H]`` (short-horizon value, H=3).
        """
        env = self.forward_env(x_t, t, context, seq_len, n_mem=n_mem)
        z_env = env["z_env"]
        z_state = env["z_state"]

        delta_i, z_i, v_i = self._value_path(z_env, candidate_action, z_state)
        # FORMAL reference: exact zero residual (Z_0 = H(Z_env, 0)); hold residual separate.
        z_0, v_0 = self._reference_zero_path(z_env)
        delta_hold = self.f_act(z_env, reference_action, z_state)   # L_zero only

        v_i_dec = self._decode(z_i, env["e"], env["grid_sizes"])
        v_0_dec = self._decode(z_0, env["e"], env["grid_sizes"])

        return {
            "z_env": z_env,
            "z_state": z_state,
            "e": env["e"],
            "grid_sizes": env["grid_sizes"],
            "candidate_residual": delta_i,
            "reference_residual": torch.zeros_like(delta_i),   # FORMAL exact zero
            "hold_residual": delta_hold,                       # F_act(Z_t,a0), L_zero only
            "candidate_future": z_i,
            "reference_future": z_0,
            "v_cand": v_i_dec,
            "v_ref": v_0_dec,
            "V_i": v_i,
            "V_0": v_0,
            "G_hat": v_i - v_0,
        }

    def forward(self, x_t, t, context, seq_len, candidate_action, reference_action, n_mem=0):
        """nn.Module/DDP entry point — delegates to forward_pair (single candidate pair)."""
        return self.forward_pair(x_t, t, context, seq_len, candidate_action,
                                 reference_action, n_mem=n_mem)

    def forward_group(self, x_t, t, context, seq_len, actions, reference_action, n_mem=0):
        r"""K candidates + 1 reference sharing one Z_env and one Z_t.

        Returns:
            dict: z_env, z_state, e, grid_sizes, V ``[K, B, H]``, V_0 ``[B, H]``,
            G_hat ``[K, B, H]``, residuals ``[K, B, L, dim]``, futures ``[K, B, L, dim]``,
            v_cand (list of K velocity lists), v_ref (velocity list),
            reference_future (=z_env), reference_residual (=0), hold_residual (F_act(Z_t,a0)).
        """
        env = self.forward_env(x_t, t, context, seq_len, n_mem=n_mem)
        z_env = env["z_env"]
        z_state = env["z_state"]
        # FORMAL reference: exact zero residual (Z_0 = H(Z_env, 0)); hold residual separate.
        z_0, v_0 = self._reference_zero_path(z_env)
        delta_hold = self.f_act(z_env, reference_action, z_state)   # L_zero only
        v_0_dec = self._decode(z_0, env["e"], env["grid_sizes"])

        vs, residuals, futures, vdecs = [], [], [], []
        for a_i in actions:
            delta_i, z_i, v_i = self._value_path(z_env, a_i, z_state)
            vdec_i = self._decode(z_i, env["e"], env["grid_sizes"])
            residuals.append(delta_i)
            futures.append(z_i)
            vs.append(v_i)
            vdecs.append(vdec_i)

        V = torch.stack(vs, dim=0)              # [K, B, H]
        residuals = torch.stack(residuals, dim=0)
        futures = torch.stack(futures, dim=0)
        return {
            "z_env": z_env,
            "z_state": z_state,
            "e": env["e"],
            "grid_sizes": env["grid_sizes"],
            "V": V,
            "V_0": v_0,
            "G_hat": V - v_0.unsqueeze(0),
            "residuals": residuals,
            "futures": futures,
            "v_cand": vdecs,
            "v_ref": v_0_dec,
            "reference_future": z_0,
            "reference_residual": torch.zeros_like(delta_hold),   # FORMAL exact zero
            "hold_residual": delta_hold,                          # F_act(Z_t,a0), L_zero only
        }
