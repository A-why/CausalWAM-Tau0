#!/usr/bin/env python3
"""Resumable 49-task closed-loop initialization evaluation orchestrator."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SUITE = ROOT / "outputs/multitask_init/final_ready_tasks.json"
OUT = ROOT / "outputs/multitask_init/closed_loop_eval.json"
CHILD = ROOT / "scripts/robotwin_theta_init_eval_one.py"
PYTHON = "/opt/conda/envs/robotwin/bin/python"


def save(payload: dict) -> None:
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(OUT)


def final_json(text: str, task: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict) and value.get("task") == task and "episodes" in value:
                return value
        except json.JSONDecodeError:
            continue
    return {"task": task, "episodes": [], "error": "no final child JSON", "output_tail": text[-5000:]}


def _expand(value):
    """Recursively expand ${CAUSALWAM_ROOT} / ${ROBOTWIN_ROOT} in manifest strings."""
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--execution-step", type=int, default=33)
    parser.add_argument("--inference-steps", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    suite = _expand(json.loads(SUITE.read_text()))
    selected = args.tasks or suite["ready_tasks"]
    task_meta = {task["task"]: task for task in suite["tasks"]}
    payload = json.loads(OUT.read_text()) if OUT.exists() else {
        "schema": "theta-init-multi-closed-loop-v1",
        "episodes_per_task": args.episodes,
        "records": {},
    }
    for index, name in enumerate(selected, 1):
        previous = payload["records"].get(name, {})
        if previous.get("episodes_complete") == args.episodes:
            print(f"EVAL {index}/{len(selected)} {name}: already complete", flush=True)
            continue
        task = task_meta[name]
        statistics = Path(task["dataset_lerobot_root"]) / "statistics_relative_v2.json"
        proc = subprocess.run(
            [
                PYTHON,
                "-u",
                str(CHILD),
                "--task",
                name,
                "--instruction",
                task["instruction"],
                "--statistics",
                str(statistics),
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--execution-step",
                str(args.execution_step),
                "--inference-steps",
                str(args.inference_steps),
                "--episodes",
                str(args.episodes),
            ],
            cwd=os.environ.get("ROBOTWIN_ROOT", "/data/QWW/RoboTwin"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        rec = final_json(proc.stdout, name)
        rec["returncode"] = proc.returncode
        payload["records"][name] = rec
        save(payload)
        print(
            f"TASK_EVAL {index}/{len(selected)} {name}: episodes={rec.get('episodes_complete')} "
            f"success={rec.get('successes')} rate={rec.get('success_rate')}",
            flush=True,
        )
    records = [payload["records"].get(name, {}) for name in suite["ready_tasks"]]
    payload.update(
        {
            "N_ready": suite["N_ready"],
            "tasks_complete": sum(rec.get("episodes_complete") == args.episodes for rec in records),
            "tasks_success_gt_0": sum(rec.get("successes", 0) > 0 for rec in records),
            "tasks_success_eq_0": [
                name
                for name in suite["ready_tasks"]
                if payload["records"].get(name, {}).get("episodes_complete") == args.episodes
                and payload["records"][name].get("successes", 0) == 0
            ],
            "total_successes": sum(rec.get("successes", 0) for rec in records),
            "total_episodes": sum(rec.get("episodes_complete", 0) for rec in records),
        }
    )
    payload["overall_success"] = (
        payload["total_successes"] / payload["total_episodes"]
        if payload["total_episodes"]
        else None
    )
    save(payload)
    return 0 if payload["tasks_complete"] == suite["N_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
