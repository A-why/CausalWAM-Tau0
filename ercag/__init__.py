"""V4-B — Shared Environment Baseline + Action Residual ER-CAG world/value model.

Word-document formal method (SINGLE path) mapped onto the Tau (WanModel) world backbone:

    Z_t          -> observation/history latent (VAE-encoded patch latent)
    F_env        -> native video backbone run with ZERO action conditioning (shared env repr)
    xi_t         -> shared flow noise (sampled once per group, baked into x_t)
    F_act        -> ActionResidualAdapter (lightweight action-conditioned residual)
    H            -> Z_i = Z_env + DeltaZ_i  (additive composition)
    downstream   -> Z_i -> native head/unpatchify -> velocity v_i (SAME native path)
    a_0          -> Hold Current Pose (33-step hold chunk)
    ValueHead    -> shared head: Q = ValueHead(Z_future)
    L_pair       -> MSE(Delta_future_pred, Delta_future_true)  (future-difference)

Submodules:
    shared_environment.py   F_env (+ head inputs e, grid_sizes)
    action_residual.py      F_act (ActionResidualAdapter)
    value_head.py           shared ValueHead
    ercag_model.py          ERCAGModel (forward_pair / forward_group)
    losses.py               L_dyn / L_ref / L_zero / L_val / L_pair (formal)
"""

from .shared_environment import SharedEnvironment
from .action_residual import ActionResidualAdapter
from .value_head import ValueHead
from .ercag_model import ERCAGModel
from .losses import (
    flow_matching_loss,
    recover_x0_from_velocity,
    l_dyn,
    l_ref,
    l_zero,
    l_val,
    l_pair_future,
    l_gain_diagnostic,
    ercag_total_loss,
    construct_noisy_latent,
)

__all__ = [
    "SharedEnvironment",
    "ActionResidualAdapter",
    "ValueHead",
    "ERCAGModel",
    "flow_matching_loss",
    "recover_x0_from_velocity",
    "l_dyn",
    "l_ref",
    "l_zero",
    "l_val",
    "l_pair_future",
    "l_gain_diagnostic",
    "ercag_total_loss",
    "construct_noisy_latent",
]
