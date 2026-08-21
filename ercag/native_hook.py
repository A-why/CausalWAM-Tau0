"""R2C §9 — READ-ONLY native future token hook (monkey-patch, no WAM source change).

The tau-0-wm simulator backbone (``model_sim.py``) already natively supports
``store_buffer=True`` -> ``final_output['video_states_buffer']``, a list of the
transformer's video-token hidden sequences ``x`` at every block, each of shape
``[B, seq_len, 3072]``. The last entry (``video_states_buffer[-1]``) is the final
future hidden representation ``Zhat [B, seq_len, 3072]`` that the shared ValueHead
reads.

The simulator *pipeline* (``textimage2video_sim.py``) does not thread ``store_buffer``
through its ``infer()``, so we capture the buffer with a read-only monkey-patch on the
model's ``forward`` — exactly the R1 ``_patch_seed`` pattern: no tau-0-wm source file
is modified, no weight / denoising / latent-coordinate change, no trainable module.
"""
from __future__ import annotations

import torch


def enable_native_future_hook(sim) -> None:
    """Wrap the simulator's backbone forward to also record video_states_buffer.

    After this, every ``sim.play(...)`` call stashes the full per-block video
    hidden-state list in ``sim._last_video_states_buffer`` (list of [B, seq_len, 3072]).
    ``sim.play`` still returns ``(pred_final_frame, reward)`` unchanged.
    """
    model = sim.diffusion_model
    if getattr(model, "_native_future_hooked", False):
        return

    orig_forward = model.forward

    def forward_hooked(*args, **kwargs):
        # force buffer capture; return_video is left as the caller set it (default True)
        kwargs["store_buffer"] = True
        out = orig_forward(*args, **kwargs)
        if isinstance(out, dict) and "video_states_buffer" in out:
            sim._last_video_states_buffer = out["video_states_buffer"]
        return out

    model.forward = forward_hooked
    model._native_future_hooked = True


def disable_native_future_hook(sim) -> None:
    """Restore the original forward (idempotent)."""
    model = sim.diffusion_model
    if getattr(model, "_native_future_hooked", False) and hasattr(model, "_orig_forward"):
        model.forward = model._orig_forward
        model._native_future_hooked = False


def get_native_future_hidden(sim, last: bool = True) -> torch.Tensor:
    """Return the native future hidden token sequence captured by the last play().

    Args:
        sim: TauSimulator whose forward was wrapped by enable_native_future_hook.
        last: True -> video_states_buffer[-1] (final block, the cleanest future repr).

    Returns:
        Tensor [B, seq_len, 3072] (padded to the transformer's seq_len).
    """
    buf = getattr(sim, "_last_video_states_buffer", None)
    if buf is None:
        raise RuntimeError("No video_states_buffer captured. Call enable_native_future_hook "
                           "then sim.play(...) first.")
    out = buf[-1] if last else buf
    return out.detach().float()
