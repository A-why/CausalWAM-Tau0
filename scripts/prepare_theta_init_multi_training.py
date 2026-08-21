#!/usr/bin/env python3
"""Materialize the one-pass theta_init_multi_v0 Tau SFT recipe."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import yaml


ROOT = Path(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BASE = ROOT / "configs/archive/training/pbb2_canonical_turn_switch.yaml"
SUITE = ROOT / "outputs/multitask_init/final_ready_tasks.json"
LOADER = ROOT / "outputs/multitask_init/loader_audit.json"
OUTPUT = ROOT / "configs/training/theta_init_multi_v0.yaml"


def main() -> int:
    suite = json.loads(SUITE.read_text())
    loader = json.loads(LOADER.read_text())
    if (
        not loader["task_balanced"]
        or not loader["all_tasks_forward_pass"]
        or not loader.get("coverage_complete", False)
    ):
        raise RuntimeError("loader audit is not fully PASS")
    config = yaml.safe_load(BASE.read_text())
    train_configs = [
        str(ROOT / "configs/data/robotwin_multitask_v0" / f"{task}.yaml")
        for task in suite["ready_tasks"]
    ]
    epoch_samples = int(loader["balanced_epoch_samples"])
    # GPU0 is RAS-quarantined (SRAM Threshold Exceeded = Yes); only GPU1 is
    # healthy, so the shared initialization trains on a single GPU. This is a
    # hardware-parallelism change only; the recipe (lr / batch_size / objective
    # / statistics / data) is unchanged. world_size is kept env-overridable.
    world_size = int(os.environ.get("TRAIN_WORLD_SIZE", "1"))
    global_steps = math.ceil(epoch_samples / world_size / int(config["batch_size"]))
    config.update(
        {
            "model_name": "theta_init_multi_v0",
            "output_dir": str(ROOT / "outputs/theta_init_multi_v0"),
            "sub_folder": "one_balanced_pass",
            "train_epochs": 1,
            "train_steps": global_steps,
            "task_balanced_sampling": True,
            "task_balanced_epoch_samples": epoch_samples,
            "multitask_training_log": str(ROOT / "outputs/multitask_init/training_log.json"),
            "latest_epoch": 0,
            "latest_global_step": 0,
            "latest_log_dir": None,
            "optimizer_path": None,
            "steps_to_save": global_steps + 1,
            "steps_to_val": global_steps + 1,
            "report_to": "none",
        }
    )
    # Start from the frozen official Tau pretrained initialization. PB-B2 is
    # deliberately absent from both model and optimizer inputs.
    config["diffusion_model"]["model_path"] = str(ROOT / "checkpoints/tau0_wm/vam")
    config["data"] = {
        "public_args": {
            "action_chunk": 33,
            "action_space": "eef6d",
            "action_type": "relative",
            "chunk": 9,
            "ignore_seek": False,
        },
        "train": train_configs,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(yaml.safe_dump(config, sort_keys=False))
    plan = {
        "config": str(OUTPUT),
        "checkpoint_base": config["diffusion_model"]["model_path"],
        "PB_B2_used": False,
        "N_tasks": suite["N_ready"],
        "balanced_epoch_samples": epoch_samples,
        "world_size": world_size,
        "batch_size_per_gpu": config["batch_size"],
        "global_training_steps": global_steps,
        "train_epochs": 1,
    }
    plan_path = ROOT / "outputs/multitask_init/training_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
