#!/usr/bin/env python3
"""Resumable one-task-one-process collection of 3 successes per READY task."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SUITE = ROOT / "outputs/multitask_init/final_ready_tasks.json"
MANIFEST = ROOT / "outputs/multitask_init/demo_collection_manifest.json"
CHILD = ROOT / "scripts/robotwin_multitask_collect_one.py"
PYTHON = "/opt/conda/envs/robotwin/bin/python"
RAW_ROOT = ROOT / "datasets/robotwin_multitask_raw_v0"


def save(payload: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(MANIFEST)


def parse_record(stdout: str, task: str, seed: int, returncode: int) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict) and value.get("task") == task:
                value["child_returncode"] = returncode
                return value
        except json.JSONDecodeError:
            continue
    return {
        "task": task,
        "seed": seed,
        "status": "FAIL",
        "child_returncode": returncode,
        "error": "collector emitted no parseable final record",
        "stdout_tail": stdout[-4000:],
    }


def existing_successes(task: str) -> list[dict]:
    records = []
    task_root = RAW_ROOT / task
    for path in sorted(task_root.glob("episode_seed*/collection_record.json")):
        try:
            rec = json.loads(path.read_text())
        except Exception:
            continue
        if rec.get("official_success") and rec.get("expert_plan_success") and rec.get("frame_count", 0) >= 33:
            records.append(rec)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=3)
    parser.add_argument("--max-distinct-seeds", type=int, default=10)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=900, help="per-attempt subprocess timeout (seconds)")
    parser.add_argument("--keep-all-cameras", action="store_true", help="do NOT filter front camera (matches historical scene/object placement)")
    args = parser.parse_args()

    suite = json.loads(SUITE.read_text())
    task_meta = {rec["task"]: rec for rec in suite["tasks"]}
    selected = args.tasks or suite["ready_tasks"]
    if args.max_tasks is not None:
        selected = selected[: args.max_tasks]

    if MANIFEST.exists():
        payload = json.loads(MANIFEST.read_text())
    else:
        payload = {
            "schema": "mainline-r4-demo-collection-v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "target_successes_per_task": args.target,
            "max_distinct_seeds_per_task": args.max_distinct_seeds,
            "process_isolation": "one-attempt-one-process; no multi-task Vulkan process",
            "tasks": {},
        }
    wall_start = time.monotonic()

    for task_index, task in enumerate(selected, 1):
        if task not in task_meta:
            raise ValueError(f"task not in frozen READY suite: {task}")
        task_rec = payload["tasks"].setdefault(task, {"attempts": []})
        successes = existing_successes(task)
        attempted = {int(rec["seed"]) for rec in task_rec["attempts"]}
        attempted.update(int(rec["seed"]) for rec in successes)
        preferred = task_meta[task].get("runtime_seed_verified")
        seed_order = []
        if isinstance(preferred, int) and 0 <= preferred < args.max_distinct_seeds:
            seed_order.append(preferred)
        seed_order.extend(seed for seed in range(args.max_distinct_seeds) if seed not in seed_order)

        print(
            f"TASK {task_index}/{len(selected)} {task}: existing={len(successes)}/{args.target} "
            f"attempted={sorted(attempted)}",
            flush=True,
        )
        for seed in seed_order:
            if len(successes) >= args.target:
                break
            if seed in attempted:
                continue
            cmd = [
                PYTHON,
                "-u",
                str(CHILD),
                "--task",
                task,
                "--seed",
                str(seed),
                "--raw-root",
                str(RAW_ROOT),
            ]
            if args.keep_all_cameras:
                cmd.append("--keep-all-cameras")
            started = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=os.environ.get("ROBOTWIN_ROOT", "/data/QWW/RoboTwin"),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
                rec = parse_record(proc.stdout, task, seed, proc.returncode)
            except subprocess.TimeoutExpired as exc:
                captured = exc.stdout or ""
                if isinstance(captured, bytes):
                    captured = captured.decode(errors="replace")
                rec = {
                    "task": task,
                    "seed": seed,
                    "status": "FAIL",
                    "child_returncode": None,
                    "error": f"TimeoutExpired: official expert attempt exceeded {args.timeout} seconds",
                    "stdout_tail": captured[-4000:],
                }
            rec["orchestrator_elapsed_sec"] = round(time.monotonic() - started, 6)
            task_rec["attempts"].append(rec)
            attempted.add(seed)
            successes = existing_successes(task)
            task_rec.update(
                {
                    "status": "COMPLETE" if len(successes) >= args.target else "IN_PROGRESS",
                    "successful_demos": len(successes),
                    "successful_seeds": [int(r["seed"]) for r in successes],
                    "target": args.target,
                    "distinct_seeds_attempted": sorted(attempted),
                }
            )
            payload["last_update_utc"] = datetime.now(timezone.utc).isoformat()
            save(payload)
            print(
                f"ATTEMPT {task} seed={seed} status={rec.get('status')} "
                f"successes={len(successes)}/{args.target} frames={rec.get('frame_count',0)} "
                f"sec={rec.get('elapsed_sec', rec['orchestrator_elapsed_sec'])}",
                flush=True,
            )

        task_rec["status"] = "COMPLETE" if len(successes) >= args.target else "DEMO_COLLECTION_BLOCKED"
        task_rec["successful_demos"] = len(successes)
        task_rec["successful_seeds"] = [int(r["seed"]) for r in successes]
        task_rec["target"] = args.target
        task_rec["distinct_seeds_attempted"] = sorted(attempted)
        save(payload)

    complete = sum(
        payload["tasks"].get(task, {}).get("status") == "COMPLETE"
        for task in suite["ready_tasks"]
    )
    successful = sum(
        payload["tasks"].get(task, {}).get("successful_demos", 0)
        for task in suite["ready_tasks"]
    )
    payload.update(
        {
            "last_update_utc": datetime.now(timezone.utc).isoformat(),
            "N_ready": suite["N_ready"],
            "target_successful_demos": suite["N_ready"] * args.target,
            "successful_demos": successful,
            "tasks_complete": complete,
            "invocation_walltime_sec": round(time.monotonic() - wall_start, 6),
        }
    )
    save(payload)
    print(
        json.dumps(
            {
                "successful_demos": successful,
                "target": suite["N_ready"] * args.target,
                "tasks_complete": complete,
                "N_ready": suite["N_ready"],
            }
        ),
        flush=True,
    )
    return 0 if complete == suite["N_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
