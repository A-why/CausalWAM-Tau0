"""
Gripper conversion between τ₀ and RoboTwin conventions.

Verified from source code:
- τ₀: range [0, 120], 0=open, 120=close (per README and TauPolicy)
- RoboTwin: range [0.0, 1.0], 0=close, 1=open (per robot.py get_left_gripper_val)

The mapping is both range-scaling AND direction-inverting.
"""
import numpy as np


# Verified conventions
TAU_GRIPPER_OPEN = 0.0
TAU_GRIPPER_CLOSE = 120.0
TAU_GRIPPER_RANGE = (0.0, 120.0)

RTW_GRIPPER_OPEN = 1.0
RTW_GRIPPER_CLOSE = 0.0
RTW_GRIPPER_RANGE = (0.0, 1.0)


def robotwin_gripper_to_tau(rtw_gripper: np.ndarray) -> np.ndarray:
    """Convert RoboTwin gripper [0=close, 1=open] → τ₀ gripper [0=open, 120=close].

    Args:
        rtw_gripper: (...,) or (..., 2) RoboTwin gripper values

    Returns:
        Same shape, τ₀ gripper values
    """
    rtw_gripper = np.asarray(rtw_gripper, dtype=np.float64)
    # Invert direction: RoboTwin 0(close) → τ₀ 120(close), RoboTwin 1(open) → τ₀ 0(open)
    # Formula: tau = (1 - rtw) * 120
    return (1.0 - rtw_gripper) * 120.0


def tau_gripper_to_robotwin(tau_gripper: np.ndarray) -> np.ndarray:
    """Convert τ₀ gripper [0=open, 120=close] → RoboTwin gripper [0=close, 1=open].

    Args:
        tau_gripper: (...,) or (..., 2) τ₀ gripper values

    Returns:
        Same shape, RoboTwin gripper values
    """
    tau_gripper = np.asarray(tau_gripper, dtype=np.float64)
    # Invert direction: τ₀ 0(open) → RoboTwin 1(open), τ₀ 120(close) → RoboTwin 0(close)
    # Formula: rtw = 1 - tau/120
    return 1.0 - tau_gripper / 120.0


def tau_action_gripper_to_robotwin(tau_action_grip: np.ndarray) -> np.ndarray:
    """Convert τ₀ action gripper [0=open, 1=close] → RoboTwin gripper [0=close, 1=open].

    τ₀ action gripper is [0, 1], not [0, 120].

    Args:
        tau_action_grip: τ₀ action gripper values in [0, 1] (0=open, 1=close)

    Returns:
        RoboTwin gripper values in [0, 1] (0=close, 1=open)
    """
    tau_action_grip = np.asarray(tau_action_grip, dtype=np.float64)
    # Simple inversion: tau 0(open) → rtw 1(open), tau 1(close) → rtw 0(close)
    return 1.0 - tau_action_grip
