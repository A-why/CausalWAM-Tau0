"""V4-B formal losses: L_dyn / L_ref / L_zero / L_val / L_pair (+ L_total).

Flow-matching convention (matches Tau reward-branch training, V3-C):
    x_t     = (1 - sigma) * x0 + sigma * noise
    target  = noise - x0              (velocity field the model predicts)
    loss    = MSE(model_velocity, target)
    x0_pred = x_t - sigma * v_pred    (Tau FlowMatchEulerDiscreteScheduler recovery)

The KEY V4-B change (§11-14): L_pair is the Word "paired future difference" —
    Delta_future_pred = x0_pred(candidate) - x0_pred(reference)
    Delta_future_true = x0(candidate)     - x0(reference)
    L_pair = MSE(Delta_future_pred, Delta_future_true)
in the SAME native future (clean latent) space. The scalar value-gain MSE
MSE(Q_i - Q_0, Y_i - Y_0) is retained only as a DIAGNOSTIC metric (l_gain_diagnostic),
not a loss term — L_val already supervises Q_i / Q_0 separately.
"""
import torch
import torch.nn.functional as F


def construct_noisy_latent(clean, noise, sigma, obs_clean_frames=1):
    r"""Bake a shared flow noise xi into a clean latent to form the noisy input x_t.

    Args:
        clean (Tensor): ``[B, C, F, H, W]`` clean latent (observation broadcast over frames).
        noise (Tensor): ``[B, C, F, H, W]`` shared flow noise xi.
        sigma (float): noise level in [0, 1].
        obs_clean_frames (int): number of leading (observation/memory) frames kept clean.

    Returns:
        Tensor: ``x_t`` of shape ``[B, C, F, H, W]``.
    """
    mask = torch.zeros_like(clean)
    if obs_clean_frames < clean.shape[2]:
        mask[:, :, obs_clean_frames:] = 1.0
    x_t = (1.0 - sigma) * clean + sigma * noise
    x_t = clean * (1.0 - mask) + x_t * mask
    return x_t


def _stack(v):
    if isinstance(v, list):
        return torch.stack(v, dim=0)
    return v


def flow_matching_loss(v_pred, x0, noise):
    r"""Native flow-matching MSE: MSE(v_pred, noise - x0)."""
    v = _stack(v_pred).float()
    target = (noise - _stack(x0)).float()
    return F.mse_loss(v, target)


def recover_x0_from_velocity(v_pred, x_t, sigma):
    r"""Recover the clean latent x0 from the predicted velocity (Tau scheduler formula).

    ``x0 = x_t - sigma * v_pred`` (FlowMatchEulerDiscreteScheduler.stochastic_sampling branch).

    Args:
        v_pred (Tensor or list): predicted velocity ``[B, C, F, H, W]``.
        x_t (Tensor or list): noisy latent ``[B, C, F, H, W]``.
        sigma (float): noise level.

    Returns:
        Tensor: recovered clean latent ``[B, C, F, H, W]`` (float32).
    """
    return _stack(x_t).float() - sigma * _stack(v_pred).float()


def l_dyn(v_cand, x0_cand, noise):
    """L_dyn: candidate future dynamics loss (formal residual path -> velocity)."""
    return flow_matching_loss(v_cand, x0_cand, noise)


def l_ref(v_ref, x0_ref, noise):
    """L_ref: reference (Hold) future dynamics loss — trains F_env via the formal path."""
    return flow_matching_loss(v_ref, x0_ref, noise)


def l_zero(hold_residual):
    r"""L_zero: mean ||DeltaZ_hold||^2 — the hold action a_0 induces ~zero residual.

    V4-B6 (§2-§11): ``hold_residual = F_act(Z_t, a_0)`` is the LEARNED hold residual, computed
    separately from the FORMAL reference path. The formal reference future is ``Z_0 = H(Z_env, 0)``
    (exact zero), so L_zero is the ONLY place a_0's residual appears, as a structural constraint.
    """
    return (hold_residual.float() ** 2).mean()


def l_val(Q_i, Q_0, Y_i, Y_0):
    r"""L_val: MSE(Q_i, Y_i) + MSE(Q_0, Y_0) with a shared value scale.

    V4-B3 (Fix B): ``Q_i/Q_0`` and ``Y_i/Y_0`` are now PER-FUTURE-FRAME tensors ``[B, H]``
    (``V_hat_h`` vs ``p_h``, h = 1..3), so ``mse_loss`` reduces over all B*H elements —
    i.e. the per-frame MSE over the short-horizon slices. The same function handles the
    legacy scalar ``[B, 1]`` case unchanged.
    """
    return F.mse_loss(Q_i.float(), Y_i.float()) + F.mse_loss(Q_0.float(), Y_0.float())


def l_pair_future(v_cand, v_ref, x_t, x0_cand, x0_ref, sigma):
    r"""L_pair (Word formal): MSE(Delta_future_pred, Delta_future_true).

    Both candidate and reference futures are recovered to the SAME clean-latent space
    (``x0_pred = x_t - sigma * v``), then differenced — matching Word §11-12.
    """
    x0_cand_pred = recover_x0_from_velocity(v_cand, x_t, sigma)
    x0_ref_pred = recover_x0_from_velocity(v_ref, x_t, sigma)
    delta_pred = x0_cand_pred - x0_ref_pred
    delta_true = _stack(x0_cand).float() - _stack(x0_ref).float()
    return F.mse_loss(delta_pred, delta_true)


def l_gain_diagnostic(Q_i, Q_0, Y_i, Y_0):
    r"""DIAGNOSTIC ONLY (not a loss): value-gain MSE(Q_i-Q_0, Y_i-Y_0)."""
    return F.mse_loss((Q_i - Q_0).float(), (Y_i - Y_0).float())


def ercag_total_loss(
    v_cand=None, v_ref=None,
    x_t=None, x0_cand=None, x0_ref=None, noise=None, sigma=1.0,
    hold_residual=None,
    Q_i=None, Q_0=None, Y_i=None, Y_0=None,
    lambda_ref=1.0, lambda_zero=0.1, lambda_val=1.0, lambda_pair=1.0,
):
    r"""L_total = L_dyn + λ_ref L_ref + λ_zero L_zero + λ_val L_val + λ_pair L_pair.

    V4-B6 (§2-§11): ``v_ref`` / ``Q_0`` now come from the FORMAL reference path ``Z_0 = H(Z_env, 0)``
    (exact zero residual), so ``L_ref`` and the reference value term carry no F_act gradient. The
    hold action's residual ``F_act(Z_t, a_0)`` is passed separately as ``hold_residual`` and feeds
    L_zero ONLY. L_pair is the FORMAL future-difference loss (l_pair_future), not value-gain.
    Returns (loss, components) where components is a dict of each finite loss term.
    """
    components = {}
    loss = None

    if v_cand is not None and x0_cand is not None and noise is not None:
        d = l_dyn(v_cand, x0_cand, noise)
        components["l_dyn"] = d
        loss = d if loss is None else loss + d

    if v_ref is not None and x0_ref is not None and noise is not None:
        r = lambda_ref * l_ref(v_ref, x0_ref, noise)
        components["l_ref"] = r
        loss = r if loss is None else loss + r

    if hold_residual is not None:
        z = lambda_zero * l_zero(hold_residual)
        components["l_zero"] = z
        loss = z if loss is None else loss + z

    if Q_i is not None and Q_0 is not None and Y_i is not None and Y_0 is not None:
        v = lambda_val * l_val(Q_i, Q_0, Y_i, Y_0)
        components["l_val"] = v
        loss = v if loss is None else loss + v

    if v_cand is not None and v_ref is not None and x_t is not None \
            and x0_cand is not None and x0_ref is not None:
        p = lambda_pair * l_pair_future(v_cand, v_ref, x_t, x0_cand, x0_ref, sigma)
        components["l_pair"] = p
        loss = p if loss is None else loss + p

    return loss, components
