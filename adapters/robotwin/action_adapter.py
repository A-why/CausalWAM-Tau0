"""
Convert τ₀ VAM action (33, 20) eef6d → RoboTwin EE action (16).
"""
import numpy as np
from .contracts import (
    TAU_ACTION_DIM, TAU_ACTION_CHUNK,
    TAU_LEFT_XYZ, TAU_LEFT_6D, TAU_LEFT_GRIPPER,
    TAU_RIGHT_XYZ, TAU_RIGHT_6D, TAU_RIGHT_GRIPPER,
    RTW_EE_ACTION_DIM, RTW_EE_LEFT_ARM, RTW_EE_LEFT_GRIPPER,
    RTW_EE_RIGHT_ARM, RTW_EE_RIGHT_GRIPPER,
)
from .rotation_utils import tau_6d_to_robotwin_quat
from .gripper_utils import tau_action_gripper_to_robotwin
from .frame_utils import arm_base_pose_to_world


def adapt_tau_action_to_robotwin(tau_action: np.ndarray) -> np.ndarray:
    """Convert τ₀ VAM action chunk (T, 20) eef6d → RoboTwin EE action (T, 16).

    τ₀ (20-dim arm-base eef6d):
        [left_xyz(3), left_6d(6), left_gripper(1), right_xyz(3), right_6d(6), right_gripper(1)]

    RoboTwin EE (16-dim global/world control):
        [left_xyz(3), left_quat_wxyz(4), left_gripper(1), right_xyz(3), right_quat_wxyz(4), right_gripper(1)]

    Args:
        tau_action: (T, 20) numpy array, τ₀ VAM output

    Returns:
        (T, 16) numpy array, RoboTwin-compatible EE action
    """
    tau_action = np.asarray(tau_action, dtype=np.float32)
    T = tau_action.shape[0]
    assert tau_action.shape[1] == TAU_ACTION_DIM, f"Expected last dim {TAU_ACTION_DIM}, got {tau_action.shape[1]}"

    rtw_action = np.zeros((T, RTW_EE_ACTION_DIM), dtype=np.float32)

    # Left arm
    left_xyz = tau_action[:, TAU_LEFT_XYZ[0]:TAU_LEFT_XYZ[1]]           # (T, 3)
    left_6d = tau_action[:, TAU_LEFT_6D[0]:TAU_LEFT_6D[1]]              # (T, 6)
    left_grip = tau_action[:, TAU_LEFT_GRIPPER]                          # (T,)

    # Right arm
    right_xyz = tau_action[:, TAU_RIGHT_XYZ[0]:TAU_RIGHT_XYZ[1]]         # (T, 3)
    right_6d = tau_action[:, TAU_RIGHT_6D[0]:TAU_RIGHT_6D[1]]            # (T, 6)
    right_grip = tau_action[:, TAU_RIGHT_GRIPPER]                         # (T,)

    # Convert Tau arm-base poses to RoboTwin's global SAPIEN control frame.
    # The transform is common to the frozen Aloha-Agilex embodiment and has no
    # task-specific path.
    left_quat_base = tau_6d_to_robotwin_quat(left_6d)     # (T, 4) wxyz
    right_quat_base = tau_6d_to_robotwin_quat(right_6d)   # (T, 4) wxyz
    left_world = [
        arm_base_pose_to_world(position, quaternion)
        for position, quaternion in zip(left_xyz, left_quat_base)
    ]
    right_world = [
        arm_base_pose_to_world(position, quaternion)
        for position, quaternion in zip(right_xyz, right_quat_base)
    ]
    left_xyz = np.asarray([pose[0] for pose in left_world], dtype=np.float32)
    left_quat = np.asarray([pose[1] for pose in left_world], dtype=np.float32)
    right_xyz = np.asarray([pose[0] for pose in right_world], dtype=np.float32)
    right_quat = np.asarray([pose[1] for pose in right_world], dtype=np.float32)

    # Convert gripper: τ₀ action [0=open, 1=close] → RoboTwin [0=close, 1=open]
    left_grip_rtw = tau_action_gripper_to_robotwin(left_grip)
    right_grip_rtw = tau_action_gripper_to_robotwin(right_grip)

    # Assemble RoboTwin EE action
    rtw_action[:, RTW_EE_LEFT_ARM[0]:RTW_EE_LEFT_ARM[1]] = np.concatenate([left_xyz, left_quat], axis=1)
    rtw_action[:, RTW_EE_LEFT_GRIPPER] = left_grip_rtw
    rtw_action[:, RTW_EE_RIGHT_ARM[0]:RTW_EE_RIGHT_ARM[1]] = np.concatenate([right_xyz, right_quat], axis=1)
    rtw_action[:, RTW_EE_RIGHT_GRIPPER] = right_grip_rtw

    return rtw_action


def adapt_tau_action_single(tau_action: np.ndarray, step: int = 0) -> np.ndarray:
    """Extract single step from τ₀ action chunk and convert to RoboTwin format.

    Args:
        tau_action: (T, 20) τ₀ action chunk
        step: which step to use (default: first)

    Returns:
        (16,) RoboTwin EE action
    """
    chunk = adapt_tau_action_to_robotwin(tau_action)
    return chunk[step]  # (16,)


def build_hold_action(robotwin_state: dict) -> np.ndarray:
    """Build a hold/no-op action from current RoboTwin state.

    Args:
        robotwin_state: dict with current left/right EEF poses and gripper values

    Returns:
        (16,) RoboTwin EE action that holds current pose
    """
    endpose = robotwin_state.get("endpose", {})

    left = np.asarray(endpose["left_endpose"], dtype=np.float32)     # [xyz + quat_wxyz]
    right = np.asarray(endpose["right_endpose"], dtype=np.float32)   # [xyz + quat_wxyz]
    left_grip = np.float32(endpose.get("left_gripper", 1.0))
    right_grip = np.float32(endpose.get("right_gripper", 1.0))

    action = np.zeros(16, dtype=np.float32)
    action[0:7] = left        # xyz + quat_wxyz
    action[7] = left_grip      # gripper
    action[8:15] = right       # xyz + quat_wxyz
    action[15] = right_grip    # gripper

    return action


def compute_action_delta(
    action_rtw: np.ndarray,
    current_state: dict,
) -> dict:
    """Compute translation, rotation, and gripper deltas for a RoboTwin action.

    Args:
        action_rtw: (16,) or (T, 16) RoboTwin EE action
        current_state: dict with left_endpose, right_endpose, grippers

    Returns:
        dict with delta metrics
    """
    if action_rtw.ndim == 1:
        action_rtw = action_rtw[np.newaxis, :]  # (1, 16)

    endpose = current_state.get("endpose", {})
    cur_left = np.asarray(endpose["left_endpose"], dtype=np.float64)
    cur_right = np.asarray(endpose["right_endpose"], dtype=np.float64)
    cur_left_grip = np.float64(endpose.get("left_gripper", 1.0))
    cur_right_grip = np.float64(endpose.get("right_gripper", 1.0))

    # First action step
    act_left_pos = action_rtw[0, 0:3]
    act_left_quat = action_rtw[0, 3:7]
    act_left_grip = action_rtw[0, 7]
    act_right_pos = action_rtw[0, 8:11]
    act_right_quat = action_rtw[0, 11:15]
    act_right_grip = action_rtw[0, 15]

    # Translation delta
    left_dxyz = act_left_pos - cur_left[0:3]
    right_dxyz = act_right_pos - cur_right[0:3]

    # Rotation delta (angular, approximate)
    from .rotation_utils import reorder_quaternion, quaternion_to_matrix
    import torch

    cur_left_q_xyzw = reorder_quaternion(cur_left[3:7], "wxyz", "xyzw")
    act_left_q_wxyz = act_left_quat
    act_left_q_xyzw = reorder_quaternion(act_left_q_wxyz, "wxyz", "xyzw")

    cur_right_q_xyzw = reorder_quaternion(cur_right[3:7], "wxyz", "xyzw")
    act_right_q_wxyz = act_right_quat
    act_right_q_xyzw = reorder_quaternion(act_right_q_wxyz, "wxyz", "xyzw")

    # Compute angular difference using matrix dot product
    def angular_distance_deg(q1_xyzw, q2_xyzw):
        q1_t = torch.from_numpy(q1_xyzw.astype(np.float32))
        q2_t = torch.from_numpy(q2_xyzw.astype(np.float32))
        R1 = quaternion_to_matrix(q1_t)
        R2 = quaternion_to_matrix(q2_t)
        R_diff = R1.T @ R2
        trace = torch.clamp(R_diff.trace(), -1.0, 3.0)
        angle_rad = torch.acos((trace - 1.0) / 2.0)
        return float(angle_rad * 180.0 / np.pi)

    left_drot_deg = angular_distance_deg(cur_left_q_xyzw, act_left_q_xyzw)
    right_drot_deg = angular_distance_deg(cur_right_q_xyzw, act_right_q_xyzw)

    return {
        "left_dxyz": left_dxyz,
        "left_dxyz_norm": float(np.linalg.norm(left_dxyz)),
        "right_dxyz": right_dxyz,
        "right_dxyz_norm": float(np.linalg.norm(right_dxyz)),
        "left_drot_deg": left_drot_deg,
        "right_drot_deg": right_drot_deg,
        "left_dgrip": float(act_left_grip - cur_left_grip),
        "right_dgrip": float(act_right_grip - cur_right_grip),
    }
