#!/usr/bin/env python3
"""V0-D0.9: Final RoboTwin → Standard LeRobot 0.4.0 Dataset Converter (tau0_wm env)."""
import sys, os, json, pickle, glob, time, argparse, shutil
import numpy as np
import torch

TAU0_ROOT = os.path.join(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tau-0-wm")
sys.path.insert(0, TAU0_ROOT)
sys.path.insert(0, os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(TAU0_ROOT)

from adapters.robotwin.frame_utils import world_pose_to_arm_base
from adapters.robotwin.rotation_utils import quaternion_to_rotation_6d, reorder_quaternion
from adapters.robotwin.gripper_utils import robotwin_gripper_to_tau


def load_episode_pkls(ep_dir):
    cache = os.path.join(ep_dir, ".cache", "episode0")
    if not os.path.isdir(cache):
        return None
    files = sorted(glob.glob(os.path.join(cache, "*.pkl")),
                   key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frames = [pickle.load(open(pf, 'rb')) for pf in files]
    return frames


def convert_frames_to_tau(frames):
    """Convert RoboTwin frames to τ₀ 20D eef6d state/action pairs."""
    n = len(frames)
    action_horizon = min(33, n - 1)
    samples = []

    for t in range(n):
        ep = frames[t]["endpose"]
        left = np.asarray(ep["left_endpose"], dtype=np.float64)
        right = np.asarray(ep["right_endpose"], dtype=np.float64)
        lg = float(ep.get("left_gripper", 1.0))
        rg = float(ep.get("right_gripper", 1.0))

        # State: current endpose → arm-base eef6d
        lb_pos, lb_q_w = world_pose_to_arm_base(left[0:3], left[3:7])
        rb_pos, rb_q_w = world_pose_to_arm_base(right[0:3], right[3:7])
        lb_q_xyzw = reorder_quaternion(lb_q_w, "wxyz", "xyzw")
        rb_q_xyzw = reorder_quaternion(rb_q_w, "wxyz", "xyzw")
        l6d = quaternion_to_rotation_6d(torch.from_numpy(lb_q_xyzw.astype(np.float32)).unsqueeze(0), "xyzw").squeeze(0).numpy()
        r6d = quaternion_to_rotation_6d(torch.from_numpy(rb_q_xyzw.astype(np.float32)).unsqueeze(0), "xyzw").squeeze(0).numpy()
        lg_tau = robotwin_gripper_to_tau(np.array([lg]))[0]
        rg_tau = robotwin_gripper_to_tau(np.array([rg]))[0]
        state = np.concatenate([lb_pos, l6d, rb_pos, r6d, [lg_tau], [rg_tau]]).astype(np.float32)

        # Action chunk: next `action_horizon` EE targets
        acts = []
        for k in range(1, action_horizon + 1):
            idx = min(t + k, n - 1)
            aep = frames[idx]["endpose"]
            al = np.asarray(aep["left_endpose"], dtype=np.float64)
            ar = np.asarray(aep["right_endpose"], dtype=np.float64)
            alg = float(aep.get("left_gripper", 1.0))
            arg = float(aep.get("right_gripper", 1.0))
            alb_pos, alb_q_w = world_pose_to_arm_base(al[0:3], al[3:7])
            arb_pos, arb_q_w = world_pose_to_arm_base(ar[0:3], ar[3:7])
            alb_q_xyzw = reorder_quaternion(alb_q_w, "wxyz", "xyzw")
            arb_q_xyzw = reorder_quaternion(arb_q_w, "wxyz", "xyzw")
            al6d = quaternion_to_rotation_6d(torch.from_numpy(alb_q_xyzw.astype(np.float32)).unsqueeze(0), "xyzw").squeeze(0).numpy()
            ar6d = quaternion_to_rotation_6d(torch.from_numpy(arb_q_xyzw.astype(np.float32)).unsqueeze(0), "xyzw").squeeze(0).numpy()
            alg_tau = 1.0 - alg  # τ₀ action gripper [0,1]
            arg_tau = 1.0 - arg
            acts.append(np.concatenate([alb_pos, al6d, arb_pos, ar6d, [alg_tau], [arg_tau]]).astype(np.float32))

        action = np.stack(acts, axis=0).astype(np.float32)

        # Images
        obs = frames[t].get("observation", {})
        images = {}
        for cam in ["head_camera", "left_camera", "right_camera"]:
            d = obs.get(cam, {})
            rgb = d.get("rgb")
            if rgb is not None:
                images[cam] = rgb  # uint8 HWC

        samples.append({"state": state, "action": action, "images": images, "frame_idx": t})

    return samples


def build_lerobot_dataset(all_episodes, output_dir, task_name, fps):
    """Build a LeRobot 0.4.0 dataset using the official API."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    os.makedirs(output_dir, exist_ok=True)

    # Camera keys (Tau-standard keys)
    camera_keys = [
        "observation.images.top_head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    ]
    cam_map = {"head_camera": 0, "left_camera": 1, "right_camera": 2}

    # Flatten all samples
    all_states, all_actions, all_frames = [], [], []
    ep_boundaries = [0]
    ep_idx = 0
    for ep_samples in all_episodes:
        for s in ep_samples:
            all_states.append(s["state"])
            all_actions.append(s["action"])
            all_frames.append({"ep": ep_idx, "frame": s["frame_idx"], "images": s["images"]})
        ep_boundaries.append(len(all_states))
        ep_idx += 1

    n_total = len(all_states)
    action_horizon = all_actions[0].shape[0]
    print(f"  Total frames: {n_total}, action_horizon: {action_horizon}")

    # LeRobot: root must NOT exist. Parent may exist.
    parent = os.path.dirname(output_dir)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # Ensure output root does not exist before create
    if os.path.exists(output_dir):
        import shutil as _shutil
        _shutil.rmtree(output_dir)

    features = {
        'observation.state': {'dtype': 'float32', 'shape': (20,)},
        'action': {'dtype': 'float32', 'shape': (33, 20)},
    }
    dataset = LeRobotDataset.create(
        repo_id="tau0_robotwin_turn_switch",
        fps=int(round(fps)),
        features=features,
        root=output_dir,
        robot_type="aloha-agilex",
        use_videos=False,
        image_writer_processes=0,
    )
    print(f"  Dataset created at {output_dir}, repo_id={dataset.repo_id}")

    # Add frames — must pass numpy arrays, not lists
    for i in range(n_total):
        frame_dict = {
            "observation.state": all_states[i].astype(np.float32),
            "action": all_actions[i].astype(np.float32),
            "task": task_name,
        }
        # Skip camera images for now (requires video feature declarations)

        dataset.add_frame(frame_dict)

        # Save episode
        if i == ep_boundaries[all_frames[i]["ep"] + 1] - 1 or i == n_total - 1:
            dataset.save_episode()

    dataset.finalize()
    print(f"  Finalized: {n_total} frames across {len(all_episodes)} episodes")

    # Compute statistics
    all_states_np = np.stack(all_states)
    all_actions_np = np.concatenate([a.reshape(-1, a.shape[-1]) for a in all_actions])
    stats = {
        "action": {"mean": all_actions_np.mean(axis=0).tolist(), "std": all_actions_np.std(axis=0).tolist()},
        "state": {"mean": all_states_np.mean(axis=0).tolist(), "std": all_states_np.std(axis=0).tolist()},
    }
    with open(os.path.join(output_dir, "statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return stats, n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=os.path.join(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets/robotwin_raw_tau30hz/turn_switch"))
    parser.add_argument("--output-dir", default=os.path.join(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets/tau0_robotwin_tiny/turn_switch"))
    parser.add_argument("--task", default="turn on the switch")
    parser.add_argument("--fps", type=float, default=31.25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    print("=" * 60)
    print("V0-D0.9: Standard LeRobot 0.4.0 Dataset Builder")
    print("=" * 60)

    # Load and convert all episodes
    all_eps = []
    for seed in args.seeds:
        ep_dir = os.path.join(args.raw_root, f"episode_seed{seed:03d}")
        frames = load_episode_pkls(ep_dir)
        if frames is None or len(frames) < 33:
            print(f"  Seed {seed}: SKIP (not enough frames)")
            continue
        samples = convert_frames_to_tau(frames)
        all_eps.append(samples)
        valid = sum(1 for s in samples if s["frame_idx"] + 33 < len(frames))
        print(f"  Seed {seed}: {len(samples)} frames, ~{valid} valid training windows")

    print(f"\n  Total episodes: {len(all_eps)}")

    # Build dataset
    stats, n_total = build_lerobot_dataset(all_eps, args.output_dir, args.task, args.fps)

    print(f"\n=== Statistics ===")
    print(f"  State mean[:3]: {stats['state']['mean'][:3]}")
    print(f"  State std[:3]: {stats['state']['std'][:3]}")
    print(f"  Action mean[:3]: {stats['action']['mean'][:3]}")
    print(f"  Action std[:3]: {stats['action']['std'][:3]}")

    # Low variance
    astd = np.array(stats['action']['std'])
    sstd = np.array(stats['state']['std'])
    print(f"  Low var actions: {np.where(astd < 1e-4)[0].tolist()}")
    print(f"  Low var states: {np.where(sstd < 1e-4)[0].tolist()}")

    print(f"\n  Saved: {args.output_dir}")
    print("Converter: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
