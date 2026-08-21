#!/usr/bin/env python3
"""Build all per-task LeRobot roots, relative statistics, and data configs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import yaml


ROOT = Path(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SUITE = ROOT / "outputs/multitask_init/final_ready_tasks.json"
OUT = ROOT / "outputs/multitask_init/dataset_build_manifest.json"
CONFIG_DIR = ROOT / "configs/data/robotwin_multitask_v0"
BUILDER = ROOT / "scripts/build_robotwin_multitask_dataset_one.py"
PYTHON = "/opt/conda/envs/tau0_wm/bin/python"
TAU = ROOT / "tau-0-wm"
STAT_BUILDER = ROOT / "scripts/compute_multitask_statistics_one.py"


def save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(OUT)


def data_config(task: dict, stats: Path) -> dict:
    pose_indices = list(range(18))
    return {
        "data_class": "CustomLeRobotDataset",
        "data_class_path": "data/example_dataset.py",
        "data": {
            "data_roots": [task["dataset_lerobot_root"]],
            "sample_size": [192, 256],
            "n_view": 3,
            "valid_cam": [
                "observation.images.top_head",
                "observation.images.hand_left",
                "observation.images.hand_right",
            ],
            "fps": 16,
            "chunk": 9,
            "action_chunk": 33,
            "action_type": "relative",
            "action_space": "eef6d",
            "ignore_seek": False,
            "action_key": "action",
            "state_key": "observation.state",
            "return_video": True,
            "return_action": True,
            "norm_action": True,
            "filter_action": False,
            "dual_arm": True,
            "state_index": pose_indices,
            "state_gripper_index": [18, 19],
            "action_index": pose_indices,
            "action_gripper_index": [18, 19],
            "statistic_files": [str(stats)],
        },
    }


def parse_last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return {"status": "FAIL", "error": "no JSON result", "output_tail": text[-3000:]}


def floor_statistics(path: Path, floor: float = 1e-6) -> dict:
    data = json.loads(path.read_text())
    floored = {}
    for key in ("action", "state"):
        indices = [i for i, value in enumerate(data[key]["std"]) if value < floor]
        for index in indices:
            data[key]["std"][index] = floor
        floored[key] = indices
    path.write_text(json.dumps(data, indent=2) + "\n")
    return floored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args()
    suite = json.loads(SUITE.read_text())
    tasks = suite["tasks"]
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [task for task in tasks if task["task"] in wanted]
    payload = json.loads(OUT.read_text()) if OUT.exists() else {"schema": "multitask-dataset-build-v1", "tasks": {}}

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for index, task in enumerate(tasks, 1):
        name = task["task"]
        print(f"DATASET {index}/{len(tasks)} {name}", flush=True)
        started = time.monotonic()
        proc = subprocess.run(
            [
                PYTHON,
                "-u",
                str(BUILDER),
                "--task",
                name,
                "--instruction",
                task["instruction"],
                "--raw-root",
                task["dataset_raw_root"],
                "--output",
                task["dataset_lerobot_root"],
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        rec = parse_last_json(proc.stdout)
        rec["builder_returncode"] = proc.returncode
        if proc.returncode != 0:
            rec["builder_output_tail"] = proc.stdout[-5000:]
            payload["tasks"][name] = rec
            save(payload)
            print(f"BUILD_FAIL {name}: {rec.get('error')}", flush=True)
            continue

        stats = Path(task["dataset_lerobot_root"]) / "statistics_relative_v2.json"
        config_path = CONFIG_DIR / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(data_config(task, stats), sort_keys=False))
        stat_proc = subprocess.run(
            [
                PYTHON,
                "-u",
                str(STAT_BUILDER),
                "--task",
                name,
                "--raw-root",
                task["dataset_raw_root"],
                "--dataset-root",
                task["dataset_lerobot_root"],
                "--config",
                str(config_path),
                "--output",
                str(stats),
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if stat_proc.returncode != 0:
            rec.update({"status": "FAIL", "statistics_error": stat_proc.stdout[-5000:]})
        else:
            stat_audit = parse_last_json(stat_proc.stdout)
            rec.update(
                {
                    "status": "PASS",
                    "statistics_path": str(stats),
                    "statistics_floor_1e-6": {
                        "action": stat_audit.get("floored_action_dimensions", []),
                        "state": stat_audit.get("floored_state_dimensions", []),
                    },
                    "statistics_loader_equivalence": stat_audit,
                    "data_config": str(config_path),
                    "instruction_conditioning": task["instruction"],
                    "elapsed_total_sec": round(time.monotonic() - started, 6),
                }
            )
        payload["tasks"][name] = rec
        save(payload)
        print(f"BUILD {name} status={rec['status']} sec={rec.get('elapsed_total_sec')}", flush=True)

    payload["pass_count"] = sum(
        payload["tasks"].get(task["task"], {}).get("status") == "PASS"
        for task in suite["tasks"]
    )
    payload["N_ready"] = suite["N_ready"]
    save(payload)
    print(json.dumps({"pass_count": payload["pass_count"], "N_ready": payload["N_ready"]}))
    return 0 if payload["pass_count"] == suite["N_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
