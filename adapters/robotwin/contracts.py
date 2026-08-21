"""
Exact data contracts for τ₀ ↔ RoboTwin adapter.

All values verified from source code (not guessed).
"""

# === τ₀ VAM (pretrained model, eef6d) ===
TAU_IMAGE_SHAPE = (192, 256)  # H, W
TAU_IMAGE_DTYPE_IN = "float32"  # Expected by TauPolicy.play()
TAU_IMAGE_RANGE = (-1.0, 1.0)
TAU_STATE_DIM = 14  # left_xyz(3)+left_quat_xyzw(4)+right_xyz(3)+right_quat_xyzw(4)
TAU_GRIPPER_DIM = 2
TAU_GRIPPER_RANGE = (0, 120)  # 0=open, 120=close
TAU_QUAT_ORDER = "xyzw"  # τ₀ uses xyzw (source: quaternion_to_matrix default)
TAU_ACTION_DIM = 20  # eef6d: left_xyz(3)+left_6d(6)+left_gripper(1)+right_xyz(3)+right_6d(6)+right_gripper(1)
TAU_ACTION_CHUNK = 33

# Layout indices for τ₀ eef6d action:
TAU_LEFT_XYZ = (0, 3)
TAU_LEFT_6D = (3, 9)
TAU_LEFT_GRIPPER = 9
TAU_RIGHT_XYZ = (10, 13)
TAU_RIGHT_6D = (13, 19)
TAU_RIGHT_GRIPPER = 19

# === RoboTwin (Aloha-Agilex, D435 cameras) ===
RTW_CAMERA_NAMES = ["head_camera", "left_camera", "right_camera"]
RTW_CAMERA_SHAPE = {"D435": (240, 320), "L515": (180, 320)}  # H, W
RTW_IMAGE_DTYPE = "uint8"
RTW_IMAGE_RANGE = (0, 255)
RTW_QUAT_ORDER = "wxyz"  # transforms3d convention
RTW_GRIPPER_RANGE = (0.0, 1.0)  # 0=close, 1=open
RTW_EE_DIM_PER_ARM = 7  # xyz(3) + quat_wxyz(4)
RTW_EE_ACTION_DIM = 16  # left_xyz(3)+left_quat_wxyz(4)+left_gripper(1)+right_xyz(3)+right_quat_wxyz(4)+right_gripper(1)

# RoboTwin EE action layout (for action_type='ee'):
RTW_EE_LEFT_ARM = (0, 7)    # xyz(3) + quat_wxyz(4)
RTW_EE_LEFT_GRIPPER = 7
RTW_EE_RIGHT_ARM = (8, 15)  # xyz(3) + quat_wxyz(4)
RTW_EE_RIGHT_GRIPPER = 15
