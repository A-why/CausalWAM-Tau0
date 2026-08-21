"""Shared per-frame ValueHead over native Tau future temporal slices.

Word §15 requires candidate/reference futures to use the SAME value scale, so a single
ValueHead object (shared parameters) maps each future frame's pooled tokens -> a scalar value.
V4-B3 (Fix B) replaces the V4-B2 single global scalar (mean-pool over ALL 864 tokens) with a
PER-FUTURE-FRAME value (§22): the same MLP is applied to each of the 3 future latent frames'
pooled hidden states, giving ``V_hat_h`` for h = 1..3.

    V_i = ValueHead(Z_i)      [B, H]     (H = 3 future frames; shared params across frames)
    V_0 = ValueHead(Z_0)      [B, H]

The formal MAINLINE-R2C/R4 target is derived only from the official sparse success
interface. The architecture contains no task identity, privileged task state,
task-specific head, or historical separate-ACVS reward head.
"""
import torch
import torch.nn as nn


class ValueHead(nn.Module):
    def __init__(self, dim: int, n_obs_frames: int = 3, n_future_frames: int = 3,
                 hidden_dim: int = 1024):
        super().__init__()
        self.n_obs_frames = n_obs_frames
        self.n_future_frames = n_future_frames
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_future: torch.Tensor):
        r"""Map a future representation to a per-future-frame value.

        Args:
            z_future (Tensor): ``[B, L, dim]`` future representation (Z_env + DeltaZ).

        Returns:
            Tensor: ``V`` of shape ``[B, n_future_frames]`` — one value per future temporal
            slice (observation tokens excluded), using the SAME shared MLP for every frame.
        """
        B, L, D = z_future.shape
        n_total = self.n_obs_frames + self.n_future_frames
        spatial = L // n_total
        z = z_future.reshape(B, n_total, spatial, D)[:, self.n_obs_frames:, :, :]
        z = z.mean(dim=2)                        # [B, n_future_frames, D]
        v = self.mlp(self.norm(z))               # [B, n_future_frames, 1]
        return v.squeeze(-1)                     # [B, n_future_frames]
