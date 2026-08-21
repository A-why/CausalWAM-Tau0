#!/usr/bin/env python3
"""Convert one task's successful official PKLs into canonical LeRobot data."""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TAU = ROOT / "tau-0-wm"
sys.path.insert(0, str(TAU))
sys.path.insert(0, str(ROOT))
os.chdir(TAU)

from adapters.robotwin.frame_utils import world_pose_to_arm_base
from adapters.robotwin.gripper_utils import robotwin_gripper_to_tau
from adapters.robotwin.rotation_utils import quaternion_to_rotation_6d, reorder_quaternion


CAMERA_MAP = {
    "head_camera": "observation.images.top_head",
    "left_camera": "observation.images.hand_left",
    "right_camera": "observation.images.hand_right",
}


def load_frames(episode_dir: Path) -> list[dict]:
    paths = sorted(
        glob.glob(str(episode_dir / ".cache/episode0/*.pkl")),
        key=lambda p: int(Path(p).stem),
    )
    return [pickle.loads(Path(path).read_bytes()) for path in paths]


def absolute_pose20(endpose: dict, *, action: bool) -> np.ndarray:
    values = []
    for side in ("left", "right"):
        pose = np.asarray(endpose[f"{side}_endpose"], dtype=np.float64)
        pos, quat_wxyz = world_pose_to_arm_base(pose[:3], pose[3:7])
        quat_xyzw = reorder_quaternion(quat_wxyz, "wxyz", "xyzw")
        rot6d = (
            quaternion_to_rotation_6d(
                torch.from_numpy(quat_xyzw.astype(np.float32)).unsqueeze(0), "xyzw"
            )
            .squeeze(0)
            .numpy()
        )
        values.extend(np.asarray(pos, dtype=np.float32).tolist())
        values.extend(np.asarray(rot6d, dtype=np.float32).tolist())
    left = float(endpose.get("left_gripper", 1.0))
    right = float(endpose.get("right_gripper", 1.0))
    if action:
        # Tau physical action gripper: 0=open, 1=close.
        values.extend([1.0 - left, 1.0 - right])
    else:
        # Tau proprio gripper: 0=open, 120=close.
        values.extend(robotwin_gripper_to_tau(np.asarray([left, right])).tolist())
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (20,) or not np.isfinite(result).all():
        raise ValueError(f"invalid canonical physical vector: shape={result.shape}")
    return result


def frame_to_sample(frames: list[dict], index: int) -> dict:
    current = frames[index]
    target = frames[min(index + 1, len(frames) - 1)]
    images = {}
    for source, destination in CAMERA_MAP.items():
        rgb = current.get("observation", {}).get(source, {}).get("rgb")
        if rgb is None:
            raise ValueError(f"missing RGB camera {source} at frame {index}")
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"invalid RGB {source} shape {rgb.shape}")
        images[destination] = rgb
    return {
        "state": absolute_pose20(current["endpose"], action=False),
        "action": absolute_pose20(target["endpose"], action=True),
        "images": images,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output = Path(args.output)
    records = []
    for record_path in sorted(raw_root.glob("episode_seed*/collection_record.json")):
        rec = json.loads(record_path.read_text())
        if rec.get("official_success") and rec.get("expert_plan_success") and rec.get("frame_count", 0) >= 33:
            records.append((record_path.parent, rec))
    # MAINLINE-R4 Section 5: the preferred budget is 3 demos/task, but expert-
    # generation-limited tasks may retain 1 genuine official-success trajectory
    # (task-balanced training samples windows with replacement). Zero demos still
    # blocks — never synthesize/duplicate files to fake the count.
    min_successful_demos = 1
    if len(records) < min_successful_demos:
        raise RuntimeError(
            f"{args.task}: expected >= {min_successful_demos} successful demos, got {len(records)}"
        )
    if output.exists() and (output / "meta/info.json").exists():
        print(json.dumps({"task": args.task, "status": "ALREADY_BUILT", "output": str(output)}))
        return 0

    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{args.task}.building_pid{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite incomplete output {output}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    first_frames = load_frames(records[0][0])
    first = frame_to_sample(first_frames, 0)
    h, w, _ = first["images"]["observation.images.top_head"].shape
    features = {
        "observation.state": {"dtype": "float32", "shape": (20,)},
        "action": {"dtype": "float32", "shape": (20,)},
        "observation.images.top_head": {"dtype": "image", "shape": (h, w, 3)},
        "observation.images.hand_left": {"dtype": "image", "shape": (h, w, 3)},
        "observation.images.hand_right": {"dtype": "image", "shape": (h, w, 3)},
    }
    dataset = LeRobotDataset.create(
        repo_id=f"tau0_robotwin_multitask_v0_{args.task}",
        fps=args.fps,
        features=features,
        root=temporary,
        robot_type="aloha-agilex",
        use_videos=False,
        image_writer_processes=0,
    )

    started = time.monotonic()
    episode_summaries = []
    for episode_dir, rec in records:
        frames = load_frames(episode_dir)
        for index in range(len(frames)):
            sample = frame_to_sample(frames, index)
            dataset.add_frame(
                {
                    "observation.state": sample["state"],
                    "action": sample["action"],
                    "task": args.instruction,
                    **sample["images"],
                }
            )
        dataset.save_episode()
        episode_summaries.append({"seed": rec["seed"], "frames": len(frames)})
    dataset.finalize()
    contract = {
        "task": args.task,
        "instruction": args.instruction,
        "demo_count": len(episode_summaries),
        "episodes": episode_summaries,
        "total_frames": sum(ep["frames"] for ep in episode_summaries),
        "fps": args.fps,
        "disk_state": "single-step 20D absolute physical arm-base eef6d; grouped grippers 18/19",
        "disk_action": "single-step 20D absolute physical arm-base eef6d; grouped grippers 18/19",
        "training_transform": "generic loader action_type=relative; model grippers 9/19",
        "native_action_chunk": 33,
        "elapsed_sec": round(time.monotonic() - started, 6),
    }
    (temporary / "canonical_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n"
    )
    temporary.replace(output)
    contract["output"] = str(output)
    contract["status"] = "PASS"
    print(json.dumps(contract, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
