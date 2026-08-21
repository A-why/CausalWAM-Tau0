#!/usr/bin/env python3
"""V0-D0: Convert RoboTwin demo → τ₀ LeRobot format. Run in robotwin env (needs transforms3d)."""
import sys, os, json, pickle, glob, argparse
import numpy as np

sys.path.insert(0, os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tau-0-wm"))
sys.path.insert(0, os.path.join(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "adapters/robotwin"))

from frame_utils import world_pose_to_arm_base, get_arm_base_transform
from rotation_utils import quaternion_to_rotation_6d, reorder_quaternion
from gripper_utils import robotwin_gripper_to_tau
from contracts import TAU_QUAT_ORDER


def load_episode_frames(raw_dir, episode_idx=0):
    """Load all per-frame pkl files for an episode."""
    frame_dir = os.path.join(raw_dir, ".cache", f"episode{episode_idx}")
    pkl_files = sorted(glob.glob(os.path.join(frame_dir, "*.pkl")),
                       key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frames = []
    for pf in pkl_files:
        with open(pf, 'rb') as f:
            frames.append(pickle.load(f))
    return frames


def compute_ee_action_from_endpose(frames):
    """Reconstruct EE actions from consecutive EEF poses.

    action[t] = endpose[t+1] (absolute EE target for next step).
    Last frame: pad with final pose.
    """
    actions = []
    for t in range(len(frames)):
        if t < len(frames) - 1:
            ep = frames[t + 1]["endpose"]
        else:
            ep = frames[t]["endpose"]  # Pad last frame
        actions.append({
            "left_endpose": np.asarray(ep["left_endpose"], dtype=np.float64),
            "right_endpose": np.asarray(ep["right_endpose"], dtype=np.float64),
            "left_gripper": float(ep.get("left_gripper", 1.0)),
            "right_gripper": float(ep.get("right_gripper", 1.0)),
        })
    return actions


def convert_to_tau_format(frames, actions):
    """Convert RoboTwin frames + actions to τ₀ eef6d format.

    Returns list of dicts with: state_20d, action_20d, images
    """
    samples = []
    for t in range(len(frames)):
        # Current state (from endpose)
        ep = frames[t]["endpose"]
        left = np.asarray(ep["left_endpose"], dtype=np.float64)   # xyz+quat_wxyz
        right = np.asarray(ep["right_endpose"], dtype=np.float64)
        left_grip_rtw = float(ep.get("left_gripper", 1.0))
        right_grip_rtw = float(ep.get("right_gripper", 1.0))

        # Convert to arm-base frame
        left_base_pos, left_base_quat_wxyz = world_pose_to_arm_base(left[0:3], left[3:7])
        right_base_pos, right_base_quat_wxyz = world_pose_to_arm_base(right[0:3], right[3:7])

        # Convert quaternion wxyz → xyzw for τ₀
        left_base_quat_xyzw = reorder_quaternion(left_base_quat_wxyz, "wxyz", "xyzw")
        right_base_quat_xyzw = reorder_quaternion(right_base_quat_wxyz, "wxyz", "xyzw")

        # Convert quaternion → 6D rotation
        import torch
        left_6d = quaternion_to_rotation_6d(
            torch.from_numpy(left_base_quat_xyzw.astype(np.float32)).unsqueeze(0),
            quat_order="xyzw"
        ).squeeze(0).numpy()
        right_6d = quaternion_to_rotation_6d(
            torch.from_numpy(right_base_quat_xyzw.astype(np.float32)).unsqueeze(0),
            quat_order="xyzw"
        ).squeeze(0).numpy()

        # Convert gripper to τ₀ state format [0,120]
        left_grip_tau = robotwin_gripper_to_tau(np.array([left_grip_rtw]))[0]
        right_grip_tau = robotwin_gripper_to_tau(np.array([right_grip_rtw]))[0]

        # Build state: [left_xyz(3), left_6d(6), right_xyz(3), right_6d(6), left_grip(1), right_grip(1)] = 20D
        state_20d = np.concatenate([
            left_base_pos, left_6d,       # 3+6=9
            right_base_pos, right_6d,     # 3+6=9
            [left_grip_tau],              # 1
            [right_grip_tau],             # 1
        ])  # 20D

        # Build action (from pre-computed EE actions, same format)
        act = actions[t]
        act_left = np.asarray(act["left_endpose"], dtype=np.float64)
        act_right = np.asarray(act["right_endpose"], dtype=np.float64)
        act_left_grip_rtw = float(act["left_gripper"])
        act_right_grip_rtw = float(act["right_gripper"])

        act_left_base_pos, act_left_base_quat_wxyz = world_pose_to_arm_base(act_left[0:3], act_left[3:7])
        act_right_base_pos, act_right_base_quat_wxyz = world_pose_to_arm_base(act_right[0:3], act_right[3:7])

        act_left_base_quat_xyzw = reorder_quaternion(act_left_base_quat_wxyz, "wxyz", "xyzw")
        act_right_base_quat_xyzw = reorder_quaternion(act_right_base_quat_wxyz, "wxyz", "xyzw")

        act_left_6d = quaternion_to_rotation_6d(
            torch.from_numpy(act_left_base_quat_xyzw.astype(np.float32)).unsqueeze(0),
            quat_order="xyzw"
        ).squeeze(0).numpy()
        act_right_6d = quaternion_to_rotation_6d(
            torch.from_numpy(act_right_base_quat_xyzw.astype(np.float32)).unsqueeze(0),
            quat_order="xyzw"
        ).squeeze(0).numpy()

        # Action gripper in τ₀ action format [0,1]
        from gripper_utils import tau_action_gripper_to_robotwin
        # Invert: rtw [0,1] → tau action [0,1]
        act_left_grip_tau = 1.0 - act_left_grip_rtw  # Same as tau_action_gripper_to_robotwin inverse
        act_right_grip_tau = 1.0 - act_right_grip_rtw

        action_20d = np.concatenate([
            act_left_base_pos, act_left_6d,        # 9
            act_right_base_pos, act_right_6d,      # 9
            [act_left_grip_tau],                    # 1
            [act_right_grip_tau],                   # 1
        ])  # 20D

        # Extract camera images
        obs = frames[t].get("observation", {})
        images = {}
        for cam_name in ["head_camera", "left_camera", "right_camera"]:
            cam_data = obs.get(cam_name, {})
            rgb = cam_data.get("rgb")
            if rgb is not None:
                images[cam_name] = rgb.astype(np.uint8)

        samples.append({
            "state_20d": state_20d.astype(np.float32),
            "action_20d": action_20d.astype(np.float32),
            "images": images,
            "frame_idx": t,
        })

    return samples


def compute_statistics(samples):
    """Compute mean/std for state and action across all samples."""
    states = np.stack([s["state_20d"] for s in samples])  # (N, 20)
    actions = np.stack([s["action_20d"] for s in samples])

    stats = {
        "action": {
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
        },
        "state": {
            "mean": states.mean(axis=0).tolist(),
            "std": states.std(axis=0).tolist(),
        },
    }

    # Check for low variance
    action_std = np.array(stats["action"]["std"])
    state_std = np.array(stats["state"]["std"])
    low_var_action = np.where(action_std < 1e-4)[0]
    low_var_state = np.where(state_std < 1e-4)[0]
    if len(low_var_action) > 0:
        print(f"  WARNING: Low variance action dims: {low_var_action.tolist()} (std={action_std[low_var_action]})")
    if len(low_var_state) > 0:
        print(f"  WARNING: Low variance state dims: {low_var_state.tolist()} (std={state_std[low_var_state]})")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=os.path.join(os.environ.get("ROBOTWIN_ROOT", "/data/QWW/RoboTwin"), "data/turn_switch/demo_clean_1ep"))
    parser.add_argument("--output-dir", default=os.path.join(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets/tau0_robotwin_tiny/turn_switch"))
    parser.add_argument("--episode-idx", type=int, default=0)
    args = parser.parse_args()

    print("=" * 60)
    print("V0-D0: RoboTwin → τ₀ Data Converter")
    print("=" * 60)

    # Load
    print(f"\nLoading frames from {args.raw_dir}...")
    frames = load_episode_frames(args.raw_dir, args.episode_idx)
    print(f"  Loaded {len(frames)} frames")

    # Reconstruct EE actions
    print("Reconstructing EE actions from endpose...")
    actions = compute_ee_action_from_endpose(frames)

    # Convert
    print("Converting to τ₀ eef6d format...")
    samples = convert_to_tau_format(frames, actions)
    print(f"  Converted {len(samples)} samples")

    # Validate
    print("\nValidation:")
    for i in [0, len(samples)//2, len(samples)-1]:
        s = samples[i]
        finite_s = np.all(np.isfinite(s["state_20d"]))
        finite_a = np.all(np.isfinite(s["action_20d"]))
        print(f"  Frame {i}: state finite={finite_s}, action finite={finite_a}, "
              f"state range=[{s['state_20d'].min():.3f},{s['state_20d'].max():.3f}], "
              f"action range=[{s['action_20d'].min():.3f},{s['action_20d'].max():.3f}]")

    # Statistics
    print("\nComputing statistics...")
    stats = compute_statistics(samples)
    print(f"  State mean[:3]: {stats['state']['mean'][:3]}")
    print(f"  Action mean[:3]: {stats['action']['mean'][:3]}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_data = {
        "samples": [{"state_20d": s["state_20d"], "action_20d": s["action_20d"]} for s in samples],
        "stats": stats,
    }
    np.savez(os.path.join(args.output_dir, "episode_0.npz"),
             states=np.stack([s["state_20d"] for s in samples]),
             actions=np.stack([s["action_20d"] for s in samples]))

    with open(os.path.join(args.output_dir, "statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Save preview images
    preview_dir = os.path.join(args.output_dir, "preview")
    os.makedirs(preview_dir, exist_ok=True)
    import cv2
    for idx, label in [(0, "start"), (len(samples)//2, "middle"), (len(samples)-1, "end")]:
        for cam_name in ["head_camera", "left_camera", "right_camera"]:
            if cam_name in samples[idx]["images"]:
                img = samples[idx]["images"][cam_name]
                cv2.imwrite(os.path.join(preview_dir, f"frame_{idx}_{cam_name}.png"),
                           cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"\nSaved to {args.output_dir}")
    print("Converter: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
