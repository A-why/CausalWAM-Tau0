#!/usr/bin/env python3
"""Evaluate one RoboTwin task for three seeds in one isolated process."""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RTW = Path(os.environ.get("ROBOTWIN_ROOT", "/data/QWW/RoboTwin"))
TAU = ROOT / "tau-0-wm"
sys.path.insert(0, str(RTW))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TAU))
os.chdir(RTW)

from adapters.robotwin.action_adapter import adapt_tau_action_to_robotwin
from adapters.robotwin.observation_adapter import adapt_observation
from ercag.official_reward import official_reward
from script.debug_expert_precheck_single_seed import build_task_args
from web_infer_utils.openpi_client.websocket_client_policy import WebsocketClientPolicy


def make_env(task: str, seed: int):
    module = importlib.import_module(f"envs.{task}")
    env = getattr(module, task)()
    task_args = build_task_args(task, "demo_clean_1ep")
    task_args.update(
        {
            "save_data": False,
            "collect_data": False,
            "eval_mode": True,
            "eval_video_log": False,
            "render_freq": 0,
        }
    )
    env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **task_args)
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--execution-step", type=int, default=33)
    parser.add_argument("--inference-steps", type=int, default=5)
    args = parser.parse_args()
    statistics = json.loads(Path(args.statistics).read_text())
    client = WebsocketClientPolicy(args.host, args.port)
    episodes = []
    task_started = time.monotonic()
    for seed in range(args.episodes):
        env = None
        rec = {"seed": seed, "status": "FAIL", "success": False}
        started = time.monotonic()
        try:
            env = make_env(args.task, seed)
            horizon = int(env.step_lim)
            actions_executed = 0
            policy_calls = 0
            finite_actions = True
            while actions_executed < horizon and official_reward(env) == 0.0:
                observation = env.get_obs()
                adapted = adapt_observation(observation, args.task)
                adapted["prompt"] = args.instruction
                response = client.infer(
                    {
                        **adapted,
                        "statistics": statistics,
                        "reset": policy_calls == 0,
                        "num_inference_steps": args.inference_steps,
                        "execution_step": args.execution_step,
                        "sample_solver": "unipc",
                    }
                )
                canonical = np.asarray(response["actions"], dtype=np.float32)
                finite_actions = finite_actions and bool(np.isfinite(canonical).all())
                if canonical.shape != (args.execution_step, 20) or not finite_actions:
                    raise ValueError(f"invalid policy action chunk {canonical.shape}")
                robotwin = adapt_tau_action_to_robotwin(canonical)
                policy_calls += 1
                for action in robotwin:
                    if actions_executed >= horizon or official_reward(env) == 1.0:
                        break
                    env.take_action(action, action_type="ee")
                    actions_executed += 1
            reward = official_reward(env)
            rec.update(
                {
                    "status": "PASS",
                    "success": reward == 1.0,
                    "official_reward": float(reward),
                    "official_reward_binary": reward in (0.0, 1.0),
                    "actions_executed": actions_executed,
                    "policy_calls": policy_calls,
                    "horizon": horizon,
                    "finite_actions": finite_actions,
                }
            )
        except BaseException as exc:
            rec.update(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": traceback.format_exc(limit=6),
                }
            )
        finally:
            if env is not None:
                try:
                    env.close_env(clear_cache=False)
                    env = None
                    gc.collect()
                    if seed == args.episodes - 1:
                        from sapien.render import clear_cache as sapien_clear_cache

                        sapien_clear_cache()
                except Exception:
                    pass
        rec["elapsed_sec"] = round(time.monotonic() - started, 6)
        episodes.append(rec)
        print(
            f"EVAL {args.task} seed={seed} status={rec['status']} success={rec['success']} "
            f"actions={rec.get('actions_executed')} sec={rec['elapsed_sec']}",
            flush=True,
        )
    payload = {
        "task": args.task,
        "instruction": args.instruction,
        "statistics": args.statistics,
        "episodes": episodes,
        "episodes_complete": sum(rec["status"] == "PASS" for rec in episodes),
        "successes": sum(rec["success"] for rec in episodes),
        "success_rate": sum(rec["success"] for rec in episodes) / len(episodes),
        "execution_step": args.execution_step,
        "inference_steps": args.inference_steps,
        "elapsed_sec": round(time.monotonic() - task_started, 6),
        "process_isolation": "one-task-one-process",
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0 if payload["episodes_complete"] == args.episodes else 2


if __name__ == "__main__":
    raise SystemExit(main())
