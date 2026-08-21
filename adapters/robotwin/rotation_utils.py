"""
Rotation conversion utilities for τ₀ ↔ RoboTwin adapter.

Reuses τ₀'s official 6D rotation functions where available.
Adds matrix→quaternion conversion (not provided by τ₀).
"""
import numpy as np
import torch

# Import τ₀ official utilities
import sys, os
sys.path.insert(0, os.path.join(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "tau-0-wm"))
from utils.action_space_utils import (
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    quaternion_to_rotation_6d,
)


def rotation_matrix_to_quaternion(R: np.ndarray, order: str = "wxyz") -> np.ndarray:
    """Convert rotation matrix to quaternion.

    Uses the standard algorithm from:
    Bar-Itzhack, "New Method for Extracting the Quaternion from a Rotation Matrix" (2000),
    improved with Shepperd's method for numerical stability.

    Args:
        R: (..., 3, 3) rotation matrix
        order: "wxyz" (default, RoboTwin/transforms3d) or "xyzw" (τ₀)

    Returns:
        (..., 4) quaternion
    """
    R = np.asarray(R)
    orig_shape = R.shape
    R_flat = R.reshape(-1, 3, 3)

    quats = np.zeros((R_flat.shape[0], 4), dtype=R.dtype)

    for i in range(R_flat.shape[0]):
        m = R_flat[i]
        # Shepperd's method: choose the largest of trace, m00, m11, m22
        trace = m[0, 0] + m[1, 1] + m[2, 2]

        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            quats[i, 0] = 0.25 * s  # w
            quats[i, 1] = (m[2, 1] - m[1, 2]) / s  # x
            quats[i, 2] = (m[0, 2] - m[2, 0]) / s  # y
            quats[i, 3] = (m[1, 0] - m[0, 1]) / s  # z
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            quats[i, 0] = (m[2, 1] - m[1, 2]) / s  # w
            quats[i, 1] = 0.25 * s  # x
            quats[i, 2] = (m[0, 1] + m[1, 0]) / s  # y
            quats[i, 3] = (m[0, 2] + m[2, 0]) / s  # z
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            quats[i, 0] = (m[0, 2] - m[2, 0]) / s  # w
            quats[i, 1] = (m[0, 1] + m[1, 0]) / s  # x
            quats[i, 2] = 0.25 * s  # y
            quats[i, 3] = (m[1, 2] + m[2, 1]) / s  # z
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            quats[i, 0] = (m[1, 0] - m[0, 1]) / s  # w
            quats[i, 1] = (m[0, 2] + m[2, 0]) / s  # x
            quats[i, 2] = (m[1, 2] + m[2, 1]) / s  # y
            quats[i, 3] = 0.25 * s  # z

        # Normalize
        quats[i] = quats[i] / np.linalg.norm(quats[i])

    quats = quats.reshape(orig_shape[:-2] + (4,))

    # Reorder if needed
    if order == "xyzw":
        quats = quats[..., [1, 2, 3, 0]]  # wxyz → xyzw
    # Default: wxyz

    return quats


def reorder_quaternion(q: np.ndarray, from_order: str, to_order: str) -> np.ndarray:
    """Reorder quaternion between xyzw and wxyz conventions.

    Args:
        q: (..., 4) quaternion
        from_order: "xyzw" or "wxyz"
        to_order: "xyzw" or "wxyz"

    Returns:
        (..., 4) reordered quaternion
    """
    if from_order == to_order:
        return q.copy()

    if from_order == "xyzw" and to_order == "wxyz":
        # [x,y,z,w] → [w,x,y,z]
        return q[..., [3, 0, 1, 2]]
    elif from_order == "wxyz" and to_order == "xyzw":
        # [w,x,y,z] → [x,y,z,w]
        return q[..., [1, 2, 3, 0]]
    else:
        raise ValueError(f"Unknown order: {from_order} → {to_order}")


def tau_6d_to_robotwin_quat(rot6d: np.ndarray) -> np.ndarray:
    """Convert τ₀ 6D rotation to RoboTwin quaternion (wxyz).

    Uses τ₀'s official rotation_6d_to_matrix, then converts to quaternion.

    Args:
        rot6d: (..., 6) τ₀ 6D rotation

    Returns:
        (..., 4) quaternion in wxyz order (RoboTwin convention)
    """
    if not isinstance(rot6d, torch.Tensor):
        rot6d_t = torch.from_numpy(np.asarray(rot6d, dtype=np.float32))
    else:
        rot6d_t = rot6d

    # τ₀ official: 6D → rotation matrix
    R = rotation_6d_to_matrix(rot6d_t)  # (..., 3, 3)
    R_np = R.detach().cpu().numpy()

    # Matrix → quaternion (wxyz for RoboTwin)
    q = rotation_matrix_to_quaternion(R_np, order="wxyz")

    return q


def robotwin_quat_to_tau_6d(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert RoboTwin quaternion (wxyz) to τ₀ 6D rotation.

    Args:
        q_wxyz: (..., 4) quaternion in wxyz order

    Returns:
        (..., 6) τ₀ 6D rotation
    """
    # Convert to xyzw for τ₀
    q_xyzw = reorder_quaternion(q_wxyz, "wxyz", "xyzw")
    q_t = torch.from_numpy(np.asarray(q_xyzw, dtype=np.float32))

    # τ₀ official: quaternion → 6D rotation
    rot6d = quaternion_to_rotation_6d(q_t, quat_order="xyzw")

    return rot6d.detach().cpu().numpy()
