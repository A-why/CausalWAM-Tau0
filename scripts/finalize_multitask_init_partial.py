#!/usr/bin/env python3
"""Materialize truthful MAINLINE-R4 partial reports after an infra stop."""
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = ROOT / "outputs/multitask_init"
STATUS = ROOT / "outputs/status/MULTITASK_TAU_INITIALIZATION.md"


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def gib(value: int) -> float:
    return value / 2**30


def main() -> int:
    suite = load("final_ready_tasks.json")
    demos = load("demo_collection_manifest.json")
    loader = load("loader_audit.json")
    builds = load("dataset_build_manifest.json")
    now = datetime.now(timezone.utc).isoformat()

    complete_tasks = [
        name
        for name in suite["ready_tasks"]
        if demos.get("tasks", {}).get(name, {}).get("status") == "COMPLETE"
    ]
    success_records = []
    for task in suite["tasks"]:
        raw = Path(task["dataset_raw_root"])
        for record_path in sorted(raw.glob("episode_seed*/collection_record.json")):
            rec = json.loads(record_path.read_text())
            if (
                rec.get("official_success")
                and rec.get("expert_plan_success")
                and rec.get("frame_count", 0) >= 33
            ):
                success_records.append(rec)
    attempts = [
        rec
        for task_rec in demos.get("tasks", {}).values()
        for rec in task_rec.get("attempts", [])
    ]
    errors = Counter(
        str(rec.get("error", "")).splitlines()[0]
        for rec in attempts
        if rec.get("status") != "SUCCESS"
    )

    demo_lines = [
        "# Unified Demo Collection — Infrastructure-Blocked Partial",
        "",
        f"Status: **{demos['collection_status']}**.",
        "",
        f"- Successful official demos: **{len(success_records)}/{suite['N_ready'] * 3}**.",
        f"- Tasks complete at 3 demos: **{len(complete_tasks)}/{suite['N_ready']}**.",
        f"- Measured invocation walltime: **{demos['walltime_sec'] / 60:.2f} min**.",
        f"- Raw data currently on disk: **{gib(demos['raw_bytes_on_disk']):.3f} GiB**.",
        "- Isolation: one official attempt per fresh process; no multi-task Vulkan process.",
        "- Only trajectories with `expert_plan_success && check_success() && frames>=33` were committed.",
        "",
        "| Task | Successful demos | Seeds | Collection status | Dataset build |",
        "|---|---:|---|---|---|",
    ]
    for name in suite["ready_tasks"]:
        rec = demos.get("tasks", {}).get(name, {})
        build = builds.get("tasks", {}).get(name, {})
        if rec or build:
            demo_lines.append(
                f"| `{name}` | {rec.get('successful_demos', 0)}/3 | "
                f"{rec.get('successful_seeds', [])} | {rec.get('status', 'NOT_STARTED')} | "
                f"{build.get('status', 'NOT_BUILT')} |"
            )
    demo_lines.extend(
        [
            "",
            "## Failure evidence",
            "",
            *[f"- {count} × `{error}`" for error, count in errors.items()],
            "",
            "`blocks_ranking_size` obtained one valid seed-0 demo, then exhausted the literal "
            "10-distinct-seed gate under GPU0 buffer failures; the GPU1 routing probe failed in "
            "official CuRobo warmup with the known driver-level misaligned-address fault. The "
            "next task reproduced that warmup fault before any expert action, so collection was "
            "paused instead of consuming its remaining seeds.",
            "",
            f"Direct blocker: **{demos['direct_blocker']}**.",
        ]
    )
    (OUT / "demo_collection_summary.md").write_text("\n".join(demo_lines) + "\n")

    loader_lines = [
        "# Unified Multi-Task Loader Audit",
        "",
        "Overall 49-task verdict: **NOT READY — DATA COLLECTION INCOMPLETE**.",
        "",
        f"The real loader audit covers the **{loader['tasks_audited']}** fully collected task roots "
        f"out of {loader['tasks_expected']}. Within that audited subset:",
        "",
        f"- task-balanced sampler: **{'PASS' if loader['task_balanced'] else 'FAIL'}**; "
        f"counts min/max={loader['task_sampling_min']}/{loader['task_sampling_max']};",
        f"- per-dataset statistics selection: **{'PASS' if loader['per_dataset_statistics'] else 'FAIL'}**;",
        f"- instruction conditioning: **{'PASS' if loader['instruction_conditioning'] else 'FAIL'}**;",
        f"- invalid windows observed: **{loader['invalid_windows']}**;",
        "- model input contract: action `[33,20]`, state `[1,20]`, video `[3,3,9,192,256]`, all finite.",
        "",
        "| Task | Windows | Balanced samples | Statistics | Instruction | Finite contract |",
        "|---|---:|---:|---|---|---|",
    ]
    for name, rec in loader["task_records"].items():
        loader_lines.append(
            f"| `{name}` | {rec['length']} | {loader['task_sampling_counts'][name]} | "
            f"{'PASS' if rec['per_dataset_statistics_selected'] else 'FAIL'} | "
            f"{'PASS' if rec['instruction_match'] else 'FAIL'} | {'PASS' if rec['pass'] else 'FAIL'} |"
        )
    loader_lines.extend(
        [
            "",
            "The diagnostic-only source label used for per-task logging is never passed to Tau. "
            "Model task identity remains natural-language instruction plus observation/history only. "
            "Training preparation has an explicit `coverage_complete` gate and refuses to create a "
            "49-task recipe from this partial audit.",
        ]
    )
    (OUT / "multitask_loader_audit.md").write_text("\n".join(loader_lines) + "\n")

    training = {
        "schema": "tau-multitask-sft-training-health-v1",
        "status": "NOT_RUN_INFRASTRUCTURE_BLOCKED",
        "created_utc": now,
        "checkpoint_directory": None,
        "training_steps": 0,
        "walltime_sec": 0.0,
        "gpu_hours": 0.0,
        "tasks_expected": suite["N_ready"],
        "tasks_with_complete_datasets": len(complete_tasks),
        "global_loss": None,
        "per_task_loss": {},
        "task_sampling_counts_by_task": {},
        "invalid_windows": None,
        "nonfinite_loss_count": None,
        "direct_blocker": demos["direct_blocker"],
    }
    (OUT / "training_log.json").write_text(json.dumps(training, indent=2) + "\n")

    with (OUT / "initialization_eval.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Task",
                "Family",
                "Dataset Complete",
                "Teacher Forced",
                "Finite Loss",
                "Finite Actions",
                "Closed Loop Episodes",
                "Official Successes",
                "Success Rate",
                "Status",
            ],
        )
        writer.writeheader()
        for task in suite["tasks"]:
            is_complete = task["task"] in complete_tasks
            writer.writerow(
                {
                    "Task": task["task"],
                    "Family": task["family"],
                    "Dataset Complete": "YES" if is_complete else "NO",
                    "Teacher Forced": "NOT_RUN",
                    "Finite Loss": "NOT_RUN",
                    "Finite Actions": "NOT_RUN",
                    "Closed Loop Episodes": 0,
                    "Official Successes": 0,
                    "Success Rate": "",
                    "Status": "NOT_RUN_INFRASTRUCTURE_BLOCKED",
                }
            )

    measured_vanilla_sec = 702.686615544837
    measured_vanilla_gpu_h = measured_vanilla_sec * 2 / 3600
    compute_lines = [
        "# Correct Shared Multi-Task Compute Accounting",
        "",
        "This replaces the obsolete task-wise `147/441 training runs` interpretation.",
        "",
        "## Actual MAINLINE-R4 consumption before the infrastructure stop",
        "",
        f"- demo collection walltime: **{demos['walltime_sec'] / 3600:.4f} h**;",
        f"- successful demos: **{len(success_records)}**; committed raw disk: **{gib(demos['raw_bytes_on_disk']):.3f} GiB**;",
        "- LeRobot roots built/audited: **3** (2,168 windows; 0 invalid in the bounded loader audit);",
        "- SFT: **0 steps, 0 GPU-hours**; teacher-forced/closed-loop: **not run**.",
        "",
        "At the current measured raw-byte rate, 147 demos project to "
        f"**{gib(demos['raw_bytes_on_disk']) / len(success_records) * 147:.2f} GiB**. "
        "This is a measured-rate projection, not a quota; the first tasks include an extra unused "
        "front-camera stream that later collection config removes.",
        "",
        "## Shared-model experiment topology",
        "",
        "| Training seeds | Multi-task initializations | Vanilla shared RL runs | ER-CAG shared RL runs | Total shared model runs |",
        "|---:|---:|---:|---:|---:|",
        "| 1 | 1 | 1 | 1 | 3 |",
        "| 3 | 3 | 3 | 3 | 9 |",
        "",
        "Let `U` be the later-locked updates per RL run and retain frozen `K=4`. Per training seed: "
        "policy updates are `2U`; real task-balanced candidate rollout chunks are `4U` for Vanilla "
        "plus `4U` candidates (and, if executed for a target, one Hold chunk) for ER-CAG; native WAM "
        "future scoring is `5U` for ER-CAG (`K` candidates + one shared Hold reference). These counts "
        "scale with total interactions, not with the number of checkpoint files, so shared training "
        "does not divide compute by 49.",
        "",
        "For a transparent 20-update screening equivalent per RL method and one seed: **40 policy "
        "updates**, **160 candidate rollout chunks** (up to 180 including executed ER Hold chunks), "
        "and **100 ER native-WAM future evaluations**. Three seeds multiply each by three.",
        "",
        "The only measured RL timing basis remains FG-C Vanilla: 20 updates in "
        f"**{measured_vanilla_sec:.3f} s on 2×H100 = {measured_vanilla_gpu_h:.4f} GPU-h**. "
        "A full ER-CAG per-update runtime and the final `U` are not measured/locked, so no fabricated "
        "total GPU-hour number is reported. With the existing 10-episode main-evaluation convention, "
        "evaluation is 1,470 episodes for one seed and 4,410 for three seeds, despite only 3/9 shared "
        "training runs.",
    ]
    (OUT / "compute_accounting.md").write_text("\n".join(compute_lines) + "\n")

    status_lines = [
        "# [Partial] Unified Multi-Task Tau Initialization",
        "",
        "## 1. Lifecycle recovery",
        "",
        "- `open_laptop`: **RECOVERED_READY**",
        "- `place_object_scale`: **RECOVERED_READY**",
        "- `put_object_cabinet`: **RECOVERED_READY**",
        "",
        "## 2. Final suite",
        "",
        "- N_all: **50**",
        "- N_ready: **49**",
        "- official_runtime_blocked: **dump_bin_bigbin**",
        "",
        "## 3. Demo collection",
        "",
        f"- successful demos: **{len(success_records)}/147**",
        f"- tasks complete: **{len(complete_tasks)}/49**",
        f"- walltime: **{demos['walltime_sec'] / 60:.2f} min**",
        "",
        "## 4. Loader",
        "",
        "- task-balanced: **PASS on 3 collected roots; full-suite NOT READY**",
        "- per-dataset statistics: **PASS 3/3**",
        "- instruction conditioning: **PASS 3/3**",
        "",
        "## 5. theta_init_multi_v0",
        "",
        "- checkpoint: **NOT CREATED**",
        "- training steps: **0**",
        "- walltime/GPU-hours: **0 / 0**",
        "",
        "## 6. Training health",
        "",
        "**NOT RUN — the coverage gate correctly rejected 3/49 datasets.**",
        "",
        "## 7. Closed-loop initialization coverage",
        "",
        "**NOT RUN — no theta_init_multi_v0 exists.**",
        "",
        "## 8. Task-specific contamination",
        "",
        "Active frozen formal path: **NONE**. Lifecycle changes are benchmark-side only; collection "
        "GPU/camera/resource routing is generic infrastructure and never enters Tau/ValueHead/RL.",
        "",
        "## 9. Need 5-demo expansion?",
        "",
        "**NO DECISION** — first complete the frozen 3-demo budget and evaluate it.",
        "",
        "## 10. Exact next",
        "",
        "**Host GPU reset/reboot.**",
        "",
        f"Unique direct blocker: `{demos['direct_blocker']}`.",
        "",
        "No Vanilla Flow-GRPO or ER-CAG run was started.",
    ]
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text("\n".join(status_lines) + "\n")
    print(json.dumps({"status": "PARTIAL", "demos": len(success_records), "tasks_complete": len(complete_tasks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
