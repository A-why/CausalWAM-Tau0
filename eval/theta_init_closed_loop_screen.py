#!/usr/bin/env python3
"""PART 4A — Fast closed-loop initialization screening (49 tasks x 1 episode).

Runs in the robotwin env (SAPIEN). Spawns ONE VAM inference worker (tau0_wm env)
that loads the theta_init_multi_v0 step_500 checkpoint and serves per-task
inference with per-task statistics + instruction. The parent driver runs each
READY RoboTwin task for a single fixed-seed episode using official
check_success() only (no RL, no ValueHead, no ER-CAG).

Output: outputs/multitask_init/init_step500_eval_1ep.csv (+ .json)

GPU separation (validated pbb2 pattern):
    parent (SAPIEN Vulkan + Warp) -> GPU selected by CUDA_VISIBLE_DEVICES (GPU0)
    worker (5.5B VAM)             -> GPU1 (VAM_GPU env var, default "1")

Usage:
    DISPLAY=:99 CUDA_VISIBLE_DEVICES=0 \
    /opt/conda/envs/robotwin/bin/python eval/theta_init_closed_loop_screen.py \
        --checkpoint <step_500> --episodes 1 --seed 0 [--tasks turn_switch]
"""
import sys, os, json, time, struct, pickle, argparse, importlib, gc, traceback, hashlib
from pathlib import Path

import numpy as np

ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT", "/data/QWW/RoboTwin")
CAUSAL_ROOT = os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAU0_ROOT = os.path.join(CAUSAL_ROOT, "tau-0-wm")
OUTPUT_DIR = os.path.join(CAUSAL_ROOT, "outputs/multitask_init")

sys.path.insert(0, ROBOTWIN_ROOT)
sys.path.insert(0, CAUSAL_ROOT)
sys.path.insert(0, TAU0_ROOT)
os.chdir(ROBOTWIN_ROOT)

from ercag.official_reward import official_reward
from script.debug_expert_precheck_single_seed import build_task_args

# VAM worker physical GPU (must differ from SAPIEN's GPU). Override via env.
VAM_GPU = os.environ.get("VAM_GPU", "1")

DEFAULT_CHECKPOINT = os.path.join(
    CAUSAL_ROOT,
    "checkpoints/theta_init_multi_v0/step_500",
)


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


def get_obs_dict(task_env):
    """Extract raw cameras + endpose dict (worker adapts to tau format)."""
    obs = task_env.get_obs()
    endpose = obs.get("endpose", {})
    cameras = obs.get("observation", {})

    state = {
        "endpose": {
            "left_endpose": list(endpose.get("left_endpose", [])),
            "right_endpose": list(endpose.get("right_endpose", [])),
            "left_gripper": float(endpose.get("left_gripper", 1.0)),
            "right_gripper": float(endpose.get("right_gripper", 1.0)),
        },
        "cameras": {},
    }
    for cam_name in ["head_camera", "left_camera", "right_camera"]:
        cam_data = cameras.get(cam_name, {})
        rgb = cam_data.get("rgb")
        if rgb is not None:
            state["cameras"][cam_name] = rgb

    return state


class VAMWorker:
    """Manages the theta_init VAM inference subprocess (tau0_wm env)."""

    def __init__(self, checkpoint_path, init_stats):
        self.worker_path = os.path.join(CAUSAL_ROOT, "eval/theta_init_vam_worker.py")
        self.checkpoint_path = checkpoint_path
        self.init_stats = init_stats
        self.proc = None
        self._start()

    def _start(self):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = VAM_GPU
        import subprocess
        cmd = [
            "/opt/conda/envs/tau0_wm/bin/python", self.worker_path,
            "--checkpoint", self.checkpoint_path,
            "--stats", self.init_stats,
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env,
        )
        start_t = time.time()
        while time.time() - start_t < 300:
            line = self.proc.stdout.readline()
            if b"VAM_READY" in line:
                return
            if self.proc.poll() is not None:
                stderr_data = self.proc.stderr.read().decode(errors="replace")
                raise RuntimeError(
                    f"VAM worker died during init. stderr={stderr_data[:8000]}"
                )
        raise RuntimeError("VAM worker did not print VAM_READY within 300s")

    def restart(self):
        self.close()
        self._start()

    def infer(self, req):
        data = pickle.dumps(req)
        try:
            self.proc.stdin.write(struct.pack(">I", len(data)))
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"VAM worker died: {e}")

        len_bytes = self.proc.stdout.read(4)
        if len(len_bytes) < 4:
            raise RuntimeError("VAM worker died")
        msg_len = struct.unpack(">I", len_bytes)[0]
        return pickle.loads(self.proc.stdout.read(msg_len))

    def close(self):
        if self.proc:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
            self.proc = None


def get_trace_state(task_env, task):
    """Read deterministic robot/task state without rendering or stepping physics."""
    state = {
        "left_endpose": np.asarray(task_env.get_arm_pose("left"), dtype=np.float64).tolist(),
        "right_endpose": np.asarray(task_env.get_arm_pose("right"), dtype=np.float64).tolist(),
        "left_gripper": float(task_env.robot.get_left_gripper_val()),
        "right_gripper": float(task_env.robot.get_right_gripper_val()),
    }
    if task == "adjust_bottle":
        state["bottle_functional_point_0"] = np.asarray(
            task_env.bottle.get_functional_point(0), dtype=np.float64
        ).tolist()
    return state


def run_episode(worker, task, instruction, stats_file, seed, execution_step,
                inference_steps, sample_solver, observation_cadence,
                record_trace=False, deterministic_comparison=False,
                max_episode_actions=None):
    """Run one closed-loop episode; returns a result dict."""
    rec = {
        "task": task, "seed": seed, "status": "FAIL", "success": False,
        "official_reward": 0.0, "actions_executed": 0, "policy_calls": 0,
        "observations_acquired": 0, "model_forwards": 0, "success_checks": 0,
        "observation_cadence": observation_cadence,
        "execution_horizon": execution_step, "horizon": None, "error": None,
    }
    env = None
    try:
        env = make_env(task, seed)
        rec["horizon"] = int(env.step_lim)
        horizon = rec["horizon"]
        if max_episode_actions is not None:
            horizon = min(horizon, int(max_episode_actions))
        actions_executed = 0
        observations_acquired = 0
        model_forwards = 0
        success_checks = 0
        finite_actions = True
        reward = 0.0
        executed_actions = []
        success_sequence = []
        while actions_executed < horizon and reward == 0.0:
            chunk_boundary = actions_executed % execution_step == 0
            acquire_observation = (
                observation_cadence == "every-step" or chunk_boundary
            )
            if acquire_observation:
                obs_dict = get_obs_dict(env)
                observations_acquired += 1
                request = {
                    "cameras": obs_dict["cameras"],
                    "endpose": obs_dict["endpose"],
                    "statistics": stats_file,
                    "instruction": instruction,
                    "task_name": task,
                    "num_inference_steps": inference_steps,
                    "execution_step": execution_step,
                    "sample_solver": sample_solver,
                    "new_episode": actions_executed == 0,
                    "episode_seed": seed,
                    "deterministic_comparison": deterministic_comparison,
                }
            else:
                # ActionChunkBroker does not inspect the observation/request on
                # cache hits; avoid rendering and serializing unused cameras.
                request = {"execution_step": execution_step}
            resp = worker.infer(request)
            if resp.get("error"):
                raise RuntimeError(f"worker error: {resp['error']}")
            action = np.asarray(resp["robotwin_action"], dtype=np.float32)
            finite_actions = finite_actions and bool(np.isfinite(action).all())
            if action.shape != (16,) or not finite_actions:
                raise ValueError(f"invalid brokered action shape {action.shape}")
            model_forwards += int(bool(resp.get("model_forward")))
            if record_trace:
                executed_actions.append(action.copy())
            env.take_action(action, action_type="ee")
            actions_executed += 1
            # Explicit wrapper-boundary check immediately after every action.
            reward = official_reward(env)
            success_checks += 1
            if record_trace:
                success_sequence.append(bool(reward))
        rec.update({
            "status": "PASS",
            "success": reward == 1.0,
            "official_reward": float(reward),
            "actions_executed": actions_executed,
            "observations_acquired": observations_acquired,
            "policy_calls": model_forwards,
            "model_forwards": model_forwards,
            "success_checks": success_checks,
            "finite_actions": finite_actions,
        })
        if record_trace:
            action_array = np.asarray(executed_actions, dtype=np.float32)
            rec.update({
                "executed_actions": action_array.tolist(),
                "executed_actions_sha256": hashlib.sha256(
                    np.ascontiguousarray(action_array).tobytes()
                ).hexdigest(),
                "success_sequence": success_sequence,
                "final_state": get_trace_state(env, task),
            })
    except BaseException as exc:
        rec.update({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": traceback.format_exc(limit=5),
        })
    finally:
        if env is not None:
            try:
                env.close_env(clear_cache=False)
                env = None
                gc.collect()
            except Exception:
                pass
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execution-step", type=int, default=33)
    parser.add_argument("--inference-steps", type=int, default=5)
    parser.add_argument("--sample-solver", default="unipc")
    parser.add_argument(
        "--observation-cadence", choices=("chunk", "every-step"), default="chunk"
    )
    parser.add_argument("--record-trace", action="store_true")
    parser.add_argument(
        "--force-seed", action="store_true",
        help="use --seed exactly instead of each task's verified manifest seed",
    )
    parser.add_argument(
        "--compare-cadence-task", default=None,
        help="for this one selected task, run every-step then chunk cadence",
    )
    parser.add_argument(
        "--comparison-max-actions", type=int, default=None,
        help="optional short horizon used only by --compare-cadence-task",
    )
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="subset of tasks to run (default: all 49)")
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "init_step500_eval_1ep.csv"))
    args = parser.parse_args()

    manifest_path = os.path.join(OUTPUT_DIR, "final_ready_tasks.json")
    manifest = json.load(open(manifest_path))
    tasks = []
    for t in manifest["tasks"]:
        if args.tasks and t["task"] not in args.tasks:
            continue
        tasks.append({
            "task": t["task"],
            "instruction": t["instruction"],
            "stats_file": os.path.join(t["dataset_lerobot_root"], "statistics_relative_v2.json"),
            # Per-task stable eval seed (manifest runtime_seed_verified). Some
            # tasks reset unstably at seed 0 (e.g. shake_bottle=3), so a uniform
            # seed would raise UnStableError before any action is taken.
            "seed": int(args.seed if args.force_seed else t.get("runtime_seed_verified", args.seed)),
        })
    if not tasks:
        raise SystemExit("no tasks selected")

    print(f"PART4A screening: {len(tasks)} tasks x {args.episodes} episode(s), "
          f"seed={args.seed}, checkpoint={args.checkpoint}", flush=True)

    # Validate all stats files exist before spawning the worker.
    for t in tasks:
        if not os.path.exists(t["stats_file"]):
            raise SystemExit(f"missing statistics: {t['stats_file']}")

    init_stats = tasks[0]["stats_file"]
    worker = VAMWorker(args.checkpoint, init_stats)

    rows = []
    csv_path = args.output
    json_path = args.output.replace(".csv", ".json")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # CSV header
    header = ["task", "seed", "observation_cadence", "status", "success", "official_reward",
              "execution_horizon", "actions_executed", "observations_acquired", "success_checks",
              "model_forwards", "policy_calls", "horizon", "elapsed_sec", "error"]
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")

    n_success = 0
    n_pass = 0
    t_start = time.monotonic()
    try:
        for i, t in enumerate(tasks):
            task = t["task"]
            base_seed = t.get("seed", args.seed)
            for seed in range(base_seed, base_seed + args.episodes):
                cadences = (
                    ("every-step", "chunk")
                    if task == args.compare_cadence_task
                    else (args.observation_cadence,)
                )
                for cadence in cadences:
                    started = time.monotonic()
                    rec = run_episode(
                        worker, task, t["instruction"], t["stats_file"], seed,
                        args.execution_step, args.inference_steps, args.sample_solver,
                        cadence, args.record_trace,
                        deterministic_comparison=(task == args.compare_cadence_task),
                        max_episode_actions=(
                            args.comparison_max_actions
                            if task == args.compare_cadence_task else None
                        ),
                    )
                    # Restart the worker once on a comm/death OR CUDA-context fault.
                    if rec.get("error") and (
                        "worker died" in rec["error"]
                        or "CUDA error" in rec["error"]
                        or "misaligned address" in rec["error"]
                    ):
                        print(f"  ⚠ worker death on {task}; restarting worker and retrying", flush=True)
                        try:
                            worker.restart()
                        except Exception as e:
                            print(f"  ⚠ worker restart failed: {e}", flush=True)
                        started = time.monotonic()
                        rec = run_episode(
                            worker, task, t["instruction"], t["stats_file"], seed,
                            args.execution_step, args.inference_steps, args.sample_solver,
                            cadence, args.record_trace,
                            deterministic_comparison=(task == args.compare_cadence_task),
                            max_episode_actions=(
                                args.comparison_max_actions
                                if task == args.compare_cadence_task else None
                            ),
                        )
                    rec["elapsed_sec"] = round(time.monotonic() - started, 2)
                    rows.append(rec)
                    if rec["status"] == "PASS":
                        n_pass += 1
                    if rec["success"]:
                        n_success += 1
                    with open(csv_path, "a") as f:
                        f.write(",".join(
                            str(rec.get(k, "")) for k in header
                        ) + "\n")
                    print(
                        f"[{i+1}/{len(tasks)}] {task} seed={seed} cadence={cadence} "
                        f"status={rec['status']} success={rec['success']} "
                        f"actions={rec.get('actions_executed')} "
                        f"obs={rec.get('observations_acquired')} "
                        f"sec={rec.get('elapsed_sec')}",
                        flush=True,
                    )
    finally:
        worker.close()

    # Summary
    payload = {
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "episodes": args.episodes,
        "n_tasks": len(tasks),
        "overall_success": n_success,
        "n_pass": n_pass,
        "success_rate": round(n_success / max(1, len(rows)), 6),
        "execution_horizon": args.execution_step,
        "observation_cadence": args.observation_cadence,
        "success_check_cadence": "after_every_env_action",
        "action_broker": "tau-0-wm ActionChunkBroker",
        "tasks_with_success": sorted(r["task"] for r in rows if r["success"]),
        "elapsed_sec": round(time.monotonic() - t_start, 2),
        "rows": rows,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"OVERALL SUCCESS: {n_success}/{len(rows)} "
          f"({payload['success_rate']:.4f})")
    if payload["tasks_with_success"]:
        print(f"TASKS WITH SUCCESS ({len(payload['tasks_with_success'])}): "
              f"{', '.join(payload['tasks_with_success'])}")
    else:
        print("TASKS WITH SUCCESS: (none)")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
