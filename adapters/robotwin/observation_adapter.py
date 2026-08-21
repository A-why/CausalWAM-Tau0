"""
Convert RoboTwin observation → τ₀ VAM input.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import cv2
from .contracts import (
    TAU_IMAGE_SHAPE, TAU_STATE_DIM, TAU_GRIPPER_DIM, TAU_QUAT_ORDER,
)
from .rotation_utils import reorder_quaternion
from .gripper_utils import robotwin_gripper_to_tau
from .frame_utils import world_pose_to_arm_base


# Default camera mapping: τ₀ view index → RoboTwin camera name
# τ₀ uses 3 views; RoboTwin provides head_camera, left_camera, right_camera
DEFAULT_CAMERA_MAPPING = {
    0: "head_camera",
    1: "left_camera",
    2: "right_camera",
}


def adapt_camera_images(
    robotwin_obs: dict,
    camera_mapping: dict = None,
    target_shape: tuple = TAU_IMAGE_SHAPE,
) -> np.ndarray:
    """Extract and convert RoboTwin camera images to τ₀ format.

    RoboTwin: uint8 (H, W, 3), range [0, 255], BGR/generic
    τ₀:       float32 (V, 3, H, W), range [-1, 1]

    Args:
        robotwin_obs: RoboTwin observation dict from get_obs()
        camera_mapping: {tau_view_idx: robotwin_camera_name}
        target_shape: (H, W) for τ₀

    Returns:
        numpy array of shape (V, 3, H, W), float32, range [-1, 1]
    """
    if camera_mapping is None:
        camera_mapping = DEFAULT_CAMERA_MAPPING

    H_tau, W_tau = target_shape
    views = []

    for view_idx in sorted(camera_mapping.keys()):
        cam_name = camera_mapping[view_idx]
        cam_data = robotwin_obs.get("observation", {}).get(cam_name, {})
        rgb = cam_data.get("rgb")

        if rgb is None:
            raise KeyError(f"Camera '{cam_name}' not found in observation. Available: "
                           f"{list(robotwin_obs.get('observation', {}).keys())}")

        # RoboTwin RGB: uint8 (H, W, 3)
        if rgb.shape[-1] != 3:
            if rgb.shape[0] == 3:
                rgb = np.transpose(rgb, (1, 2, 0))
            else:
                raise ValueError(f"Unexpected image shape: {rgb.shape}")

        # Resize to τ₀ target
        rgb_resized = cv2.resize(rgb, (W_tau, H_tau), interpolation=cv2.INTER_LINEAR)

        # Convert: uint8 [0,255] → float32 [-1,1] using official τ₀ formula
        img = rgb_resized.astype(np.float32) / 127.5 - 1.0

        # HWC → CHW
        img = np.transpose(img, (2, 0, 1))

        views.append(img)

    return np.stack(views, axis=0)  # (V, 3, H, W)


def adapt_state(robotwin_obs: dict) -> np.ndarray:
    """Convert RoboTwin EEF poses to τ₀ state format.

    RoboTwin endpose: world-frame {left/right: [xyz(3)+quat_wxyz(4)]}
    τ₀ state: arm-base [left_xyz(3)+left_quat_xyzw(4)+right_xyz(3)+right_quat_xyzw(4)] = 14 dims

    Returns:
        numpy array of shape (14,), float64
    """
    endpose = robotwin_obs.get("endpose", {})

    left = np.asarray(endpose["left_endpose"], dtype=np.float64)   # world [xyz + quat_wxyz]
    right = np.asarray(endpose["right_endpose"], dtype=np.float64) # world [xyz + quat_wxyz]

    # RoboTwin observations use the global SAPIEN frame, whereas Tau's
    # canonical physical state/action contract uses the shared arm-base frame.
    # This transform is embodiment-wide and deliberately has no task branch.
    left_pos, left_quat_wxyz = world_pose_to_arm_base(left[:3], left[3:7])
    right_pos, right_quat_wxyz = world_pose_to_arm_base(right[:3], right[3:7])

    # Reorder quaternion: wxyz → xyzw
    left_quat_xyzw = reorder_quaternion(left_quat_wxyz, "wxyz", "xyzw")
    right_quat_xyzw = reorder_quaternion(right_quat_wxyz, "wxyz", "xyzw")

    state = np.concatenate([
        left_pos,   # arm-base xyz
        left_quat_xyzw,  # quat xyzw
        right_pos,  # arm-base xyz
        right_quat_xyzw,  # quat xyzw
    ])

    assert state.shape == (TAU_STATE_DIM,), f"Expected ({TAU_STATE_DIM},), got {state.shape}"
    return state


def adapt_gripper(robotwin_obs: dict) -> np.ndarray:
    """Convert RoboTwin gripper values to τ₀ gripper format.

    RoboTwin: [left_gripper, right_gripper], range [0, 1], 0=close, 1=open
    τ₀:       [left_gripper, right_gripper], range [0, 120], 0=open, 120=close

    Returns:
        numpy array of shape (2,), float64
    """
    endpose = robotwin_obs.get("endpose", {})
    rtw_grip = np.array([
        endpose.get("left_gripper", 0.0),
        endpose.get("right_gripper", 0.0),
    ], dtype=np.float64)

    tau_grip = robotwin_gripper_to_tau(rtw_grip)
    assert tau_grip.shape == (TAU_GRIPPER_DIM,)
    return tau_grip


@lru_cache(maxsize=None)
def _official_task_instruction(task_name: str) -> str:
    resource = Path(os.path.join(os.environ.get("ROBOTWIN_ROOT", "/data/QWW/RoboTwin"), "description/task_instruction")) / f"{task_name}.json"
    if resource.exists():
        instruction = str(json.loads(resource.read_text())["full_description"]).strip()
        return instruction.replace("{a}", "one arm")
    return task_name.replace("_", " ")


def get_instruction(robotwin_obs: dict = None, task_name: str = "") -> str:
    """Get an observation-provided or official resource-backed instruction."""
    # RoboTwin tasks provide instruction via env.get_instruction()
    if robotwin_obs is not None and hasattr(robotwin_obs, 'get'):
        instr = robotwin_obs.get("instruction")
        if instr:
            return instr

    return _official_task_instruction(task_name)


def adapt_observation(robotwin_obs: dict, task_name: str = "") -> dict:
    """Full RoboTwin observation → τ₀ VAM input.

    Returns dict with keys matching TauPolicy.play() args:
        obs, state, gripper_states, prompt
    """
    return {
        "obs": adapt_camera_images(robotwin_obs),
        "state": adapt_state(robotwin_obs),
        "gripper_states": adapt_gripper(robotwin_obs),
        "prompt": get_instruction(robotwin_obs, task_name),
    }
