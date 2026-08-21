"""
Coordinate frame transforms between RoboTwin world/base and τ₀ arm-base frames.

RoboTwin:
- EEF observation (get_arm_pose → _trans_endpose): GLOBAL SAPIEN frame
- Control (take_action ee): GLOBAL SAPIEN frame
- Entity origin: arm base in world frame (from entity_origion_pose)

τ₀:
- State/action: "Arm Base link" (arm-base-relative)

For Aloha-Agilex dual-arm: both arms share the SAME entity origin
(single URDF entity, is_dual_arm=True).
"""
import numpy as np
import torch

# Verified from Aloha-Agilex config (assets/embodiments/aloha-agilex/config.yml)
# robot_pose: [[0, -0.65, 0, 1, 0, 0, 1]] → [0, -0.65, 0, 0.707, 0, 0, 0.707] wxyz
ARMS_BASE_POSITION = np.array([0.0, -0.65, 0.0], dtype=np.float64)
ARMS_BASE_QUAT_WXYZ = np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float64)


def get_arm_base_transform():
    """Return the arm-base pose in world frame as (position, quat_wxyz)."""
    return ARMS_BASE_POSITION.copy(), ARMS_BASE_QUAT_WXYZ.copy()


def world_pose_to_arm_base(position_world, quat_wxyz_world):
    """Convert a pose from world frame to arm-base frame.

    T_base = inv(T_entity_origin) @ T_world

    Args:
        position_world: (3,) xyz in world frame
        quat_wxyz_world: (4,) quaternion wxyz in world frame

    Returns:
        position_base: (3,) xyz in arm-base frame
        quat_wxyz_base: (4,) quaternion wxyz in arm-base frame
    """
    from transforms3d.quaternions import quat2mat, mat2quat

    p_world = np.asarray(position_world, dtype=np.float64)
    q_world = np.asarray(quat_wxyz_world, dtype=np.float64)

    # Entity origin in world
    p_origin = ARMS_BASE_POSITION
    q_origin = ARMS_BASE_QUAT_WXYZ
    R_origin = quat2mat(q_origin)  # world → entity rotation

    # Position: p_base = R_origin^T @ (p_world - p_origin)
    p_base = R_origin.T @ (p_world - p_origin)

    # Rotation: R_base = R_origin^T @ R_world
    R_world_mat = quat2mat(q_world)
    R_base = R_origin.T @ R_world_mat
    q_base = mat2quat(R_base)  # returns wxyz

    return p_base, q_base


def arm_base_pose_to_world(position_base, quat_wxyz_base):
    """Convert a pose from arm-base frame to world frame.

    T_world = T_entity_origin @ T_base

    Args:
        position_base: (3,) xyz in arm-base frame
        quat_wxyz_base: (4,) quaternion wxyz in arm-base frame

    Returns:
        position_world: (3,) xyz in world frame
        quat_wxyz_world: (4,) quaternion wxyz in world frame
    """
    from transforms3d.quaternions import quat2mat, mat2quat

    p_base = np.asarray(position_base, dtype=np.float64)
    q_base = np.asarray(quat_wxyz_base, dtype=np.float64)

    p_origin = ARMS_BASE_POSITION
    q_origin = ARMS_BASE_QUAT_WXYZ
    R_origin = quat2mat(q_origin)  # world → entity rotation

    # Position: p_world = p_origin + R_origin @ p_base
    p_world = p_origin + R_origin @ p_base

    # Rotation: R_world = R_origin @ R_base
    R_base_mat = quat2mat(q_base)
    R_world = R_origin @ R_base_mat
    q_world = mat2quat(R_world)

    return p_world, q_world


def world_action_to_ee_control(raw_ee_pose_world, gripper_world):
    """Convert a world-frame EEF pose to the RoboTwin EE control format.

    This is a pass-through since RoboTwin EE control already uses world frame.
    Just assembles the 16-dim action.

    Args:
        raw_ee_pose_world: (7,) [xyz(3), quat_wxyz(4)]
        gripper_world: float, RoboTwin gripper [0,1]

    Returns:
        (16,) RoboTwin EE action
    """
    action = np.zeros(16, dtype=np.float32)
    action[0:7] = raw_ee_pose_world
    action[7] = gripper_world
    return action
