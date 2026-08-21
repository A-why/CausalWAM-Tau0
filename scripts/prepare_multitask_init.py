#!/usr/bin/env python3
"""Freeze the MAINLINE-R3.1 READY suite from the code-derived R3 manifest."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RTW = Path(os.environ.get("ROBOTWIN_ROOT", "/data/QWW/RoboTwin"))
R3 = ROOT / "outputs/all_task_audit/all_tasks_manifest.json"
OUT = ROOT / "outputs/multitask_init/final_ready_tasks.json"
RECOVERED = {"open_laptop", "place_object_scale", "put_object_cabinet"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_instruction(task: str) -> tuple[str, str]:
    path = RTW / "description/task_instruction" / f"{task}.json"
    payload = json.loads(path.read_text())
    instruction = str(payload["full_description"]).strip()
    # The sole unresolved placeholder in official full descriptions is an arm
    # placeholder, not an object identity. Ground it without adding task state.
    instruction = instruction.replace("{a}", "one arm")
    return instruction, str(path)


def main() -> int:
    manifest = json.loads(R3.read_text())
    ready = sorted(
        rec["task"]
        for rec in manifest["tasks"]
        if rec["formal_pipeline_status"] == "READY_COMPATIBLE"
    )
    if len(ready) != 49:
        raise RuntimeError(f"expected post-recovery N_READY=49, got {len(ready)}")
    if not RECOVERED.issubset(ready):
        raise RuntimeError(f"recovered tasks absent from READY suite: {sorted(RECOVERED-set(ready))}")

    task_records = []
    by_task = {rec["task"]: rec for rec in manifest["tasks"]}
    for task in ready:
        instruction, instruction_resource = official_instruction(task)
        rec = by_task[task]
        task_records.append(
            {
                "task": task,
                "family": rec["family"],
                "instruction": instruction,
                "instruction_resource": instruction_resource,
                "formal_pipeline_status": rec["formal_pipeline_status"],
                "runtime_seed_verified": rec["runtime"].get("seed"),
                "lifecycle_recovery": "RECOVERED_READY" if task in RECOVERED else "NOT_REQUIRED",
                "dataset_raw_root": str(ROOT / "datasets/robotwin_multitask_raw_v0" / task),
                "dataset_lerobot_root": str(ROOT / "datasets/tau0_robotwin_multitask_v0" / task),
            }
        )

    lifecycle_files = {
        task: {
            "path": str(RTW / "envs" / f"{task}.py"),
            "sha256": sha256(RTW / "envs" / f"{task}.py"),
        }
        for task in sorted(RECOVERED)
    }
    payload = {
        "schema": "mainline-r3.1-final-ready-suite-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(R3),
        "source_manifest_sha256": sha256(R3),
        "N_all": manifest["official_task_count"],
        "N_ready": len(ready),
        "ready_tasks": ready,
        "tasks": task_records,
        "official_runtime_blocked": ["dump_bin_bigbin"],
        "runtime_blocker": "official reset raises UnStableError for seeds 0-4; frozen R3 verdict",
        "recovered_tasks": sorted(RECOVERED),
        "lifecycle_benchmark_files": lifecycle_files,
        "method_contract": {
            "disk_state": "20D absolute physical arm-base eef6d (grouped grippers 18/19)",
            "disk_action": "single-step 20D absolute physical arm-base eef6d (grouped grippers 18/19)",
            "training_action_type": "relative",
            "model_action": "canonical interleaved 20D; grippers 9/19",
            "native_action_chunk": 33,
            "task_conditioning": "official natural-language instruction only",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(OUT)
    print(json.dumps({"N_all": payload["N_all"], "N_ready": payload["N_ready"], "output": str(OUT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
