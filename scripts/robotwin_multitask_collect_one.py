#!/usr/bin/env python3
"""Collect exactly one official RoboTwin expert attempt in one fresh process."""
from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RTW = Path(os.environ.get("ROBOTWIN_ROOT", "/data/QWW/RoboTwin"))
sys.path.insert(0, str(RTW))
# Keep CuRobo and Vulkan/CUDA interop on one explicitly selected physical GPU.
# This must be set before importing RoboTwin (and therefore torch/curobo).
os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES",
    os.environ.get("ROBOTTWIN_COLLECTION_GPU", "0"),
)
os.chdir(RTW)

from script.debug_expert_precheck_single_seed import build_task_args


def warmup_hold_action(env) -> None:
    """One bounded canonical-20D Hold before the expert (matches the MAINLINE-R3.1
    lifecycle reproduction). Executes a single take_action with the current
    endpose + a 0.001 nudge on the active arm; does NOT save frames (single step
    < save_freq) and does not change the scene, only primes CuRobo/physics."""
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from adapters.robotwin.action_adapter import adapt_tau_action_single
    from adapters.robotwin.frame_utils import world_pose_to_arm_base
    from adapters.robotwin.rotation_utils import robotwin_quat_to_tau_6d

    endpose = env.get_obs()["endpose"]
    action = np.zeros(20, dtype=np.float32)
    left = np.asarray(endpose["left_endpose"], dtype=np.float32)
    right = np.asarray(endpose["right_endpose"], dtype=np.float32)
    left_pos, left_quat = world_pose_to_arm_base(left[:3], left[3:7])
    right_pos, right_quat = world_pose_to_arm_base(right[:3], right[3:7])
    action[0:3] = left_pos
    action[3:9] = robotwin_quat_to_tau_6d(left_quat)
    action[9] = 1.0 - float(endpose["left_gripper"])
    action[10:13] = right_pos
    action[13:19] = robotwin_quat_to_tau_6d(right_quat)
    action[19] = 1.0 - float(endpose["right_gripper"])
    active_arm = str(env.arm_tag)
    action[0 if active_arm == "left" else 10] += 0.001
    robotwin_action = adapt_tau_action_single(action[None, :])
    env.take_action(robotwin_action, action_type="ee")


def directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def bound_renderer_capacity() -> None:
    """Avoid RoboTwin's 50k/50k renderer reservation in a one-scene process."""
    import sapien.core as sapien_core
    import sapien.render as sapien_render

    original = sapien_render.set_global_config

    def bounded(*args, **kwargs):
        kwargs["max_num_materials"] = min(int(kwargs.get("max_num_materials", 4096)), 4096)
        kwargs["max_num_textures"] = min(int(kwargs.get("max_num_textures", 4096)), 4096)
        return original(*args, **kwargs)

    sapien_render.set_global_config = bounded
    original_renderer = sapien_core.SapienRenderer
    physical_gpu = os.environ.get("ROBOTTWIN_COLLECTION_GPU", "0")
    renderer_pci = {
        "0": "pci:0000:9c:00.0",
        "1": "pci:0000:a0:00.0",
    }[physical_gpu]

    def selected_renderer(*args, **kwargs):
        # Use the PCI address because SAPIEN's ``cuda:0`` alias can still
        # resolve to the first Vulkan device even under CUDA_VISIBLE_DEVICES.
        kwargs.setdefault("device", renderer_pci)
        return original_renderer(*args, **kwargs)

    sapien_core.SapienRenderer = selected_renderer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--raw-root", default=str(ROOT / "datasets/robotwin_multitask_raw_v0"))
    parser.add_argument("--task-config", default="demo_clean_1ep")
    parser.add_argument("--warmup-action", action="store_true", help="execute one hold action before the expert (historical lifecycle path)")
    parser.add_argument("--eval-mode", action="store_true", help="override eval_mode=True (historical lifecycle path)")
    parser.add_argument("--no-save", action="store_true", help="override save_data=False (no rendering during expert)")
    parser.add_argument("--keep-all-cameras", action="store_true", help="do NOT filter front camera (matches historical scene/object placement)")
    args_cli = parser.parse_args()

    task_root = Path(args_cli.raw_root) / args_cli.task
    destination = task_root / f"episode_seed{args_cli.seed:03d}"
    existing_record = destination / "collection_record.json"
    if existing_record.exists():
        rec = json.loads(existing_record.read_text())
        if rec.get("official_success") and rec.get("expert_plan_success"):
            rec["status"] = "ALREADY_COLLECTED"
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            return 0

    attempt_root = task_root / f".attempt_seed{args_cli.seed:03d}_pid{os.getpid()}"
    if attempt_root.exists():
        shutil.rmtree(attempt_root)
    attempt_root.mkdir(parents=True, exist_ok=False)
    env = None
    started = time.monotonic()
    rec = {
        "task": args_cli.task,
        "seed": args_cli.seed,
        "status": "FAIL",
        "task_config": args_cli.task_config,
        "process_isolation": "one-attempt-one-process",
        "save_freq_physics_steps": 15,
        "nominal_fps": 16,
    }
    try:
        bound_renderer_capacity()
        module = importlib.import_module(f"envs.{args_cli.task}")
        env = getattr(module, args_cli.task)()
        task_args = build_task_args(args_cli.task, args_cli.task_config)
        task_args = copy.deepcopy(task_args)
        # Tau's frozen observation contract consumes head + two wrist RGB
        # views. RoboTwin also instantiates an unused front camera by default;
        # retaining it during long expert recording can exhaust Vulkan render
        # buffers without contributing a dataset field. Filter only that
        # unused static view at the collection-config layer.
        #
        # NOTE: this filter is NOT a pure rendering change. Dropping the front
        # camera consumes fewer np.random draws inside load_camera() (which runs
        # BEFORE load_actors()), so it silently shifts the per-seed object
        # placement. For put_object_cabinet seed 9 this flips the object from
        # 113_coffee-box (expert-solvable) to 047_mouse (unsolvable). Use
        # --keep-all-cameras to match the historical MAINLINE-R3.1 scene exactly.
        if not args_cli.keep_all_cameras:
            for side in ("left_embodiment_config", "right_embodiment_config"):
                embodiment = task_args.get(side, {})
                if "static_camera_list" in embodiment:
                    embodiment["static_camera_list"] = [
                        camera
                        for camera in embodiment["static_camera_list"]
                        if camera.get("name") == "head_camera"
                    ]
        task_args.update(
            {
                "save_path": str(attempt_root),
                "save_data": True,
                "save_freq": 15,
                "need_plan": True,
                "collect_data": False,
                "eval_mode": False,
                "eval_video_log": False,
                "render_freq": 0,
                "use_seed": False,
                "data_type": {
                    "rgb": True,
                    "endpose": True,
                    "qpos": False,
                    "depth": False,
                    "third_view": False,
                    "pointcloud": False,
                    "mesh_segmentation": False,
                    "actor_segmentation": False,
                },
            }
        )
        if args_cli.eval_mode:
            task_args["eval_mode"] = True
        if args_cli.no_save:
            task_args["save_data"] = False
        env.setup_demo(now_ep_num=0, seed=args_cli.seed, **task_args)
        if args_cli.warmup_action:
            warmup_hold_action(env)
        env.play_once()
        success = bool(env.check_success())
        plan_success = bool(getattr(env, "plan_success", True))
        frame_dir = attempt_root / ".cache/episode0"
        frames = sorted(frame_dir.glob("*.pkl")) if frame_dir.is_dir() else []
        rec.update(
            {
                "expert_plan_success": plan_success,
                "official_success": success,
                "official_reward": float(success),
                "frame_count": len(frames),
                "finite_minimum_window": len(frames) >= 33,
                "saved_camera_contract": [
                    "head_camera",
                    "left_camera",
                    "right_camera",
                ],
                "unused_front_camera_rendered": False,
                "renderer_capacity": {
                    "max_num_materials": 4096,
                    "max_num_textures": 4096,
                },
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "renderer_device": "explicit PCI device matching ROBOTTWIN_COLLECTION_GPU and CuRobo",
            }
        )
        if not (success and plan_success):
            raise RuntimeError("official expert did not yield plan_success and check_success")
        if len(frames) < 33:
            raise RuntimeError(f"successful trajectory has only {len(frames)} frames (<33)")

        rec.update(
            {
                "status": "SUCCESS",
                "elapsed_sec": round(time.monotonic() - started, 6),
                "raw_bytes": directory_bytes(attempt_root),
                "episode_path": str(destination),
            }
        )
        (attempt_root / "collection_record.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False) + "\n"
        )
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing destination {destination}")
        attempt_root.replace(destination)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        return 0
    except BaseException as exc:
        rec.update(
            {
                "elapsed_sec": round(time.monotonic() - started, 6),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc(limit=6),
            }
        )
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        return 2
    finally:
        if env is not None:
            try:
                # This collector owns a fresh process for exactly one attempt,
                # so the official renderer cache can and must be torn down on
                # exit. Keeping it alive is only useful for multiple episodes
                # in one process and can strand Vulkan allocations here.
                env.close_env(clear_cache=False)
                env = None
                gc.collect()
                from sapien.render import clear_cache as sapien_clear_cache

                sapien_clear_cache()
            except Exception:
                pass
        if attempt_root.exists():
            shutil.rmtree(attempt_root)


if __name__ == "__main__":
    raise SystemExit(main())
