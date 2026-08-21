#!/usr/bin/env python3
"""Fast exact per-task relative statistics using Tau's canonical transform."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TAU = ROOT / "tau-0-wm"
sys.path.insert(0, str(TAU))
sys.path.insert(0, str(ROOT))

from build_robotwin_multitask_dataset_one import absolute_pose20, load_frames
from utils.action_space_utils import abs_eef_to_rela


def interleave(pose18: np.ndarray, grippers2: np.ndarray) -> np.ndarray:
    return np.concatenate([pose18[..., :9], grippers2[..., :1], pose18[..., 9:], grippers2[..., 1:]], axis=-1)


class Running:
    def __init__(self):
        self.count = 0
        self.sum = np.zeros(20, dtype=np.float64)
        self.sq = np.zeros(20, dtype=np.float64)

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, 20)
        self.count += len(values)
        self.sum += values.sum(axis=0)
        self.sq += np.square(values).sum(axis=0)

    def finish(self, floor: float = 1e-6) -> tuple[dict, list[int]]:
        mean = self.sum / self.count
        std = np.sqrt(np.maximum(self.sq / self.count - np.square(mean), 0.0))
        floored = np.flatnonzero(std < floor).tolist()
        std = np.maximum(std, floor)
        return {"mean": mean.tolist(), "std": std.tolist()}, floored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw_root = Path(args.raw_root)
    started = time.monotonic()
    action_stats, state_stats = Running(), Running()
    episode_arrays = []

    records = sorted(raw_root.glob("episode_seed*/collection_record.json"))
    # MAINLINE-R4 Section 5: MIN_SUCCESSFUL_DEMOS_PER_TASK=1. Expert-generation-
    # limited tasks (e.g. put_object_cabinet) may retain 1-2 genuine official-
    # success trajectories; statistics are computed over however many valid
    # episodes are present (>=1). Zero still blocks.
    if len(records) < 1:
        raise RuntimeError(f"expected >= 1 collected episodes, got {len(records)}")
    for record_path in records:
        record = json.loads(record_path.read_text())
        if not (record.get("official_success") and record.get("expert_plan_success")):
            raise RuntimeError(f"non-success record in dataset: {record_path}")
        frames = load_frames(record_path.parent)
        states = np.stack([absolute_pose20(frame["endpose"], action=False) for frame in frames])
        actions = np.stack(
            [
                absolute_pose20(frames[min(index + 1, len(frames) - 1)]["endpose"], action=True)
                for index in range(len(frames))
            ]
        )
        episode_arrays.append((states, actions))
        for index in range(len(frames)):
            query = np.minimum(np.arange(index, index + 33), len(frames) - 1)
            absolute_chunk = actions[query]
            relative_pose = abs_eef_to_rela(
                torch.from_numpy(absolute_chunk[:, :18]),
                torch.from_numpy(states[index : index + 1, :18]),
            ).numpy()
            relative_action = interleave(relative_pose, absolute_chunk[:, 18:20])
            model_state = interleave(states[index, :18], states[index, 18:20])
            action_stats.add(relative_action)
            state_stats.add(model_state)

    action_result, action_floored = action_stats.finish()
    state_result, state_floored = state_stats.finish()
    result = {"action": action_result, "state": state_result}
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2) + "\n")

    # Numeric equivalence gate against actual loader transform and episode
    # boundary padding on the first/middle/final dataset indices.
    from data.example_dataset import CustomLeRobotDataset

    config = yaml.safe_load(Path(args.config).read_text())
    config["data"].update(norm_action=False, filter_action=False, return_video=False)
    dataset = CustomLeRobotDataset(**config["data"])
    direct_pairs = []
    for states, actions in episode_arrays:
        for index in range(len(states)):
            query = np.minimum(np.arange(index, index + 33), len(states) - 1)
            relative_pose = abs_eef_to_rela(
                torch.from_numpy(actions[query, :18]),
                torch.from_numpy(states[index : index + 1, :18]),
            ).numpy()
            direct_pairs.append(
                (
                    interleave(relative_pose, actions[query, 18:20]),
                    interleave(states[index, :18], states[index, 18:20])[None],
                )
            )
    probe_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    max_action_diff = 0.0
    max_state_diff = 0.0
    for index in probe_indices:
        sample = dataset[index]
        direct_action, direct_state = direct_pairs[index]
        max_action_diff = max(max_action_diff, float(np.max(np.abs(sample["actions"].numpy() - direct_action))))
        max_state_diff = max(max_state_diff, float(np.max(np.abs(sample["state"].numpy() - direct_state))))
    if max_action_diff > 1e-6 or max_state_diff > 1e-6:
        raise RuntimeError(
            f"direct statistics transform differs from loader: action={max_action_diff}, state={max_state_diff}"
        )
    audit = {
        "task": args.task,
        "status": "PASS",
        "state_sample_count": state_stats.count,
        "action_sample_count": action_stats.count,
        "statistics_floor": 1e-6,
        "floored_action_dimensions": action_floored,
        "floored_state_dimensions": state_floored,
        "loader_equivalence_probe_indices": probe_indices,
        "loader_max_abs_action_diff": max_action_diff,
        "loader_max_abs_state_diff": max_state_diff,
        "elapsed_sec": round(time.monotonic() - started, 6),
    }
    print(json.dumps(audit), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
