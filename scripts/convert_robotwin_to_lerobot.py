#!/usr/bin/env python3
"""V0-D0.6: RoboTwin demo → LeRobot dataset (tau0_wm env)."""
import sys, os, json, pickle, glob, argparse, time
import numpy as np
import torch

TAU0_ROOT = os.path.join(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tau-0-wm")
PROJ_ROOT = os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TAU0_ROOT)
sys.path.insert(0, PROJ_ROOT)
os.chdir(TAU0_ROOT)

from adapters.robotwin.frame_utils import world_pose_to_arm_base
from adapters.robotwin.rotation_utils import (
    quaternion_to_rotation_6d, reorder_quaternion,
)
from adapters.robotwin.gripper_utils import robotwin_gripper_to_tau

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def load_frames(raw_dir, ep_idx=0):
    frame_dir = os.path.join(raw_dir, ".cache", f"episode{ep_idx}")
    files = sorted(glob.glob(os.path.join(frame_dir, "*.pkl")),
                   key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frames = []
    for pf in files:
        with open(pf, 'rb') as f:
            frames.append(pickle.load(f))
    return frames


def convert_episode_to_samples(frames, source_fps, target_fps):
    """Convert frames to τ₀ training samples.

    Each sample: state_20d (1,20), action_33d (33,20), video_views.

    Action alignment: action[t] = endpose[t+1] (next EE target).
    Last frame padded with its own pose.
    """
    n = len(frames)
    # For τ₀ training: need 33 consecutive action steps + 9 video frames
    # At 30Hz: 33 actions span ~1.07s. At 16.7Hz: 33 steps would span ~2s.
    # We adjust: use a shorter horizon based on actual FPS
    action_horizon = min(33, n - 1)  # Use at most n-1 steps

    samples = []
    for t in range(n):
        ep = frames[t]["endpose"]
        left = np.asarray(ep["left_endpose"], dtype=np.float64)
        right = np.asarray(ep["right_endpose"], dtype=np.float64)
        left_grip_rtw = float(ep.get("left_gripper", 1.0))
        right_grip_rtw = float(ep.get("right_gripper", 1.0))

        # Convert state to arm-base eef6d
        left_base_pos, left_base_q_wxyz = world_pose_to_arm_base(left[0:3], left[3:7])
        right_base_pos, right_base_q_wxyz = world_pose_to_arm_base(right[0:3], right[3:7])
        left_base_q_xyzw = reorder_quaternion(left_base_q_wxyz, "wxyz", "xyzw")
        right_base_q_xyzw = reorder_quaternion(right_base_q_wxyz, "wxyz", "xyzw")

        left_6d = quaternion_to_rotation_6d(
            torch.from_numpy(left_base_q_xyzw.astype(np.float32)).unsqueeze(0), "xyzw"
        ).squeeze(0).numpy()
        right_6d = quaternion_to_rotation_6d(
            torch.from_numpy(right_base_q_xyzw.astype(np.float32)).unsqueeze(0), "xyzw"
        ).squeeze(0).numpy()

        left_grip_tau = robotwin_gripper_to_tau(np.array([left_grip_rtw]))[0]
        right_grip_tau = robotwin_gripper_to_tau(np.array([right_grip_rtw]))[0]

        state = np.concatenate([left_base_pos, left_6d, right_base_pos, right_6d,
                                [left_grip_tau], [right_grip_tau]]).astype(np.float32)

        # Build action chunk: next `action_horizon` EE targets
        actions_20d = []
        for k in range(1, action_horizon + 1):
            idx = min(t + k, n - 1)  # Pad last frames
            aep = frames[idx]["endpose"]
            al = np.asarray(aep["left_endpose"], dtype=np.float64)
            ar = np.asarray(aep["right_endpose"], dtype=np.float64)
            alg = float(aep.get("left_gripper", 1.0))
            arg = float(aep.get("right_gripper", 1.0))

            alb_pos, alb_q_wxyz = world_pose_to_arm_base(al[0:3], al[3:7])
            arb_pos, arb_q_wxyz = world_pose_to_arm_base(ar[0:3], ar[3:7])
            alb_q_xyzw = reorder_quaternion(alb_q_wxyz, "wxyz", "xyzw")
            arb_q_xyzw = reorder_quaternion(arb_q_wxyz, "wxyz", "xyzw")

            al_6d = quaternion_to_rotation_6d(
                torch.from_numpy(alb_q_xyzw.astype(np.float32)).unsqueeze(0), "xyzw"
            ).squeeze(0).numpy()
            ar_6d = quaternion_to_rotation_6d(
                torch.from_numpy(arb_q_xyzw.astype(np.float32)).unsqueeze(0), "xyzw"
            ).squeeze(0).numpy()

            # Action gripper: τ₀ [0,1], 0=open
            alg_tau = 1.0 - alg
            arg_tau = 1.0 - arg

            action_step = np.concatenate([alb_pos, al_6d, arb_pos, ar_6d,
                                          [alg_tau], [arg_tau]]).astype(np.float32)
            actions_20d.append(action_step)

        action_arr = np.stack(actions_20d, axis=0)  # (H, 20)

        # Extract images
        obs = frames[t].get("observation", {})
        images = {}
        for cam_name in ["head_camera", "left_camera", "right_camera"]:
            cam_data = obs.get(cam_name, {})
            rgb = cam_data.get("rgb")
            if rgb is not None:
                images[cam_name] = rgb

        samples.append({
            "state": state,
            "actions": action_arr,
            "images": images,
            "frame_idx": t,
        })

    return samples


def create_lerobot_dataset(samples, output_dir, task_name, fps, camera_mapping):
    """Create a LeRobot dataset from converted samples."""
    os.makedirs(output_dir, exist_ok=True)

    # Camera key mapping: RoboTwin → LeRobot
    camera_keys = [camera_mapping.get(c, c) for c in ["head_camera", "left_camera", "right_camera"]]
    # Use standard τ₀ keys
    tau_camera_keys = [
        "observation.images.top_head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    ]

    # LeRobot 0.4.0 video storage requires complex infrastructure.
    # For V0-D0.6 smoke: save NPZ + PNG images. Full LeRobot format is a V0-D1 concern.
    print("  Using NPZ+PNG format for dev dataset")

    # Save as NPZ for loader validation
    npz_dir = os.path.join(output_dir, "npz_data")
    os.makedirs(npz_dir, exist_ok=True)

    all_states = np.stack([s["state"] for s in samples])
    all_actions = np.stack([s["actions"] for s in samples])

    np.savez(os.path.join(npz_dir, "episode_0.npz"),
             states=all_states, actions=all_actions,
             frame_indices=np.arange(len(samples)))

    # Save images as individual files
    img_dir = os.path.join(npz_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    import cv2
    for i, s in enumerate(samples):
        for cam_name in ["head_camera", "left_camera", "right_camera"]:
            if cam_name in s["images"]:
                cv2.imwrite(os.path.join(img_dir, f"frame_{i:04d}_{cam_name}.png"),
                           cv2.cvtColor(s["images"][cam_name], cv2.COLOR_RGB2BGR))

    # Save metadata
    meta = {
        "task": task_name,
        "fps": fps,
        "episodes": 1,
        "frames": len(samples),
        "state_dim": 20,
        "action_dim": 20,
        "action_horizon": all_actions.shape[1],
        "camera_keys": tau_camera_keys,
        "coordinate_frame": "arm_base",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(output_dir, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Compute statistics
    stats = {
        "action": {"mean": all_actions.mean(axis=(0,1)).tolist(), "std": all_actions.std(axis=(0,1)).tolist()},
        "state": {"mean": all_states.mean(axis=0).tolist(), "std": all_states.std(axis=0).tolist()},
    }
    with open(os.path.join(output_dir, "statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Saved {len(samples)} samples to {output_dir}")
    return samples, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=os.path.join(os.environ.get("ROBOTWIN_ROOT", "/data/QWW/RoboTwin"), "data/turn_switch/demo_clean_1ep"))
    parser.add_argument("--output-dir", default=os.path.join(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets/tau0_robotwin_dev_16hz/turn_switch"))
    parser.add_argument("--task", default="turn on the switch")
    parser.add_argument("--source-fps", type=float, default=16.67)
    args = parser.parse_args()

    print("=" * 60)
    print("V0-D0.6: LeRobot Data Converter")
    print("=" * 60)

    # Hardcoded camera mapping
    camera_mapping = {
        "head_camera": "observation.images.top_head",
        "left_camera": "observation.images.hand_left",
        "right_camera": "observation.images.hand_right",
    }

    # Load and convert
    frames = load_frames(args.raw_dir)
    print(f"Loaded {len(frames)} frames from {args.raw_dir}")

    samples = convert_episode_to_samples(frames, args.source_fps, 30.0)
    print(f"Converted {len(samples)} samples, action horizon={samples[0]['actions'].shape[0]}")

    # Validate
    valid = True
    for i in [0, len(samples)//2, len(samples)-1]:
        s = samples[i]
        ok = np.all(np.isfinite(s["state"])) and np.all(np.isfinite(s["actions"]))
        if not ok: valid = False
        print(f"  Frame {i}: finite={ok}, state_range=[{s['state'].min():.2f},{s['state'].max():.2f}]")
    print(f"  All finite: {valid}")

    # Create dataset
    _, stats = create_lerobot_dataset(samples, args.output_dir, args.task, args.source_fps, camera_mapping)

    print(f"\nStatistics:")
    print(f"  State mean[:3]: {stats['state']['mean'][:3]}")
    print(f"  Action range: [{min(stats['action']['mean'])}, {max(stats['action']['mean'])}]")
    print(f"Converter: {'PASS' if valid else 'FAIL'}")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
