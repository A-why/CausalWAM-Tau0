#!/usr/bin/env python3
"""PB-B Phase E1: Closed-loop development evaluation.

Runs in robotwin env (SAPIEN). Spawns VAM inference subprocess in tau0_wm env.
Evaluates pretrained, step_100, step_300, step_800 on seeds 100-104.

Usage:
    cd ${ROBOTWIN_ROOT} && DISPLAY=:99 \
    /opt/conda/envs/robotwin/bin/python ${CAUSALWAM_ROOT}/eval/pbb_closed_loop.py
"""
import sys, os, json, time, struct, pickle, argparse, yaml
import numpy as np

# ── Paths ──────────────────────────────────────────────────
ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT", "/data/QWW/RoboTwin")
CAUSAL_ROOT = os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAU0_ROOT = os.path.join(CAUSAL_ROOT, "tau-0-wm")
OUTPUT_DIR = os.path.join(CAUSAL_ROOT, "outputs/pbb_productive_policy/evaluation")

sys.path.insert(0, ROBOTWIN_ROOT)
sys.path.insert(0, os.path.join(ROBOTWIN_ROOT, "description/utils"))

# ── Constants ──────────────────────────────────────────────
TASK = "turn_switch"
MAX_POLICY_STEPS = 200
CONTROL_FREQ = 15.625  # Hz

# From PB-A contract — full precision
SUCCESS_THRESHOLD = 0.14198621809482576
GAIN_TOLERANCE = 1e-7  # numerical tolerance for float32 qpos comparison

SEEDS = [100, 101, 102, 103, 104]

CHECKPOINTS = {
    "pretrained": {
        "path": os.path.join(CAUSAL_ROOT, "checkpoints/tau0_wm/vam"),
        "action_type": "absolute",  # pretrained τ₀ uses absolute actions
    },
    "step100": {
        "path": os.path.join(CAUSAL_ROOT, "outputs/pbb/turn_switch/2026_08_12_02_29_06/step_100"),
        "action_type": "relative",  # PB-B fine-tuned with relative actions
    },
    "step300": {
        "path": os.path.join(CAUSAL_ROOT, "outputs/pbb/turn_switch/2026_08_12_02_29_06/step_300"),
        "action_type": "relative",
    },
    "step800": {
        "path": os.path.join(CAUSAL_ROOT, "outputs/pbb/turn_switch/2026_08_12_02_29_06/step_800"),
        "action_type": "relative",
    },
}

STATS_FILE = os.path.join(CAUSAL_ROOT, "datasets/tau0_robotwin_success_v3_lerobot/turn_switch/statistics.json")
ADAPTER_DIR = os.path.join(CAUSAL_ROOT, "adapters/robotwin")

# ── Environment ────────────────────────────────────────────

def init_task_env(seed):
    """Initialize turn_switch RoboTwin environment."""
    from envs.turn_switch import turn_switch
    from envs import CONFIGS_PATH
    task_env = turn_switch()

    with open(os.path.join(ROBOTWIN_ROOT, "task_config/demo_clean.yml")) as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml")) as f:
        emb_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    embodiment_type = args.get("embodiment", ["aloha-agilex"])
    robot_file = emb_types[embodiment_type[0]]["file_path"]

    def get_emb(rf):
        with open(os.path.join(rf, "config.yml")) as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = TASK
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = get_emb(robot_file)
    args["right_embodiment_config"] = get_emb(robot_file)
    head_cam = args["camera"]["head_camera_type"]
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml")) as f:
        cam_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["head_camera_h"] = cam_cfg[head_cam]["h"]
    args["head_camera_w"] = cam_cfg[head_cam]["w"]
    args["seed"] = seed
    args["save_data"] = False
    args["eval_mode"] = False
    args["data_type"] = {"rgb": True, "endpose": True, "qpos": False, "depth": False}
    args["render_freq"] = 0
    args["need_plan"] = True
    args["collect_data"] = False

    task_env.setup_demo(**args)
    return task_env


def get_obs_dict(task_env):
    """Extract observation dict from task_env."""
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


def get_switch_qpos(task_env):
    """Get switch joint qpos via env's switch articulation actor."""
    try:
        if hasattr(task_env, 'switch'):
            qpos = task_env.switch.get_qpos()
            if qpos is not None and len(qpos) > 0:
                return float(qpos[0])
    except Exception:
        pass
    return None


def check_success(qpos):
    """Check if switch qpos passes the success threshold."""
    if qpos is None:
        return False
    return qpos >= SUCCESS_THRESHOLD


# ── VAM Inference Subprocess ───────────────────────────────

class VAMClient:
    """Manages VAM inference subprocess in tau0_wm env."""

    def __init__(self, checkpoint_path, action_type=None):
        vam_server_path = os.path.join(CAUSAL_ROOT, "eval/vam_server.py")
        env = os.environ.copy()

        self.proc = None
        self._start_proc(vam_server_path, checkpoint_path, action_type, env)

    def _start_proc(self, server_path, checkpoint_path, action_type, env):
        import subprocess
        cmd = [
            "/opt/conda/envs/tau0_wm/bin/python", server_path,
            "--checkpoint", checkpoint_path,
            "--stats", STATS_FILE,
            "--adapter-dir", ADAPTER_DIR,
        ]
        if action_type:
            cmd += ["--action-type", action_type]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        # Wait for VAM_READY on stdout (vam_server prints it to stdout)
        start_t = time.time()
        while time.time() - start_t < 180:
            line = self.proc.stdout.readline()
            if b"VAM_READY" in line:
                break
            if self.proc.poll() is not None:
                stderr_data = self.proc.stderr.read().decode()
                stdout_data = self.proc.stdout.read().decode()
                raise RuntimeError(f"VAM process died during init. stderr={stderr_data[:500]} stdout={stdout_data[:500]}")

    def infer(self, obs_dict):
        """Send observation, receive action. Returns action dict."""
        obs_bytes = pickle.dumps(obs_dict)
        self.proc.stdin.write(struct.pack(">I", len(obs_bytes)))
        self.proc.stdin.write(obs_bytes)
        self.proc.stdin.flush()

        try:
            action_len_bytes = self.proc.stdout.read(4)
            if len(action_len_bytes) < 4:
                raise RuntimeError("VAM process died")
            action_len = struct.unpack(">I", action_len_bytes)[0]
            action_bytes = self.proc.stdout.read(action_len)
            return pickle.loads(action_bytes)
        except Exception as e:
            raise RuntimeError(f"VAM comm error: {e}")

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.stdout.close()
            try:
                self.proc.wait(timeout=10)
            except:
                self.proc.kill()
            self.proc = None


# ── Episode Runner ─────────────────────────────────────────

def run_episode(checkpoint_name, checkpoint_info, seed, output_dir):
    """Run one closed-loop episode."""
    checkpoint_path = checkpoint_info["path"]
    action_type = checkpoint_info.get("action_type", "relative")
    print(f"\n{'='*60}")
    print(f"[{checkpoint_name}] seed={seed} action_type={action_type}")
    print(f"{'='*60}")

    result = {
        "checkpoint": checkpoint_name,
        "seed": seed,
        "action_type": action_type,
        "success": False,
        "Ymax": 0.0,
        "threshold": SUCCESS_THRESHOLD,
        "success_margin": float(-SUCCESS_THRESHOLD),
        "success_timestep": -1,
        "episode_length": 0,
        "controller_errors": 0,
        "action_nan_inf": 0,
        "termination": "timeout",
        "switch_qpos_trace": [],
        "left_dxyz_trace": [],
        "right_dxyz_trace": [],
    }

    # Start VAM
    try:
        vam = VAMClient(checkpoint_path, action_type=action_type)
    except Exception as e:
        print(f"  ❌ VAM init failed: {e}")
        result["termination"] = "vam_init_error"
        return result

    # Init env
    try:
        task_env = init_task_env(seed)
    except Exception as e:
        print(f"  ❌ Env init failed: {e}")
        result["termination"] = "env_init_error"
        vam.close()
        return result

    try:
        for step in range(MAX_POLICY_STEPS):
            obs = get_obs_dict(task_env)

            # Get action from VAM
            try:
                action_data = vam.infer(obs)
            except Exception as e:
                print(f"  Step {step}: VAM error: {e}")
                result["termination"] = "vam_error"
                break

            if action_data.get("error"):
                print(f"  Step {step}: VAM error: {action_data['error']}")
                result["termination"] = "vam_error"
                break

            rtw_action = np.array(action_data["robotwin_action"], dtype=np.float32)
            if rtw_action.shape != (16,):
                print(f"  Step {step}: Bad action shape {rtw_action.shape}")
                result["termination"] = "bad_action_shape"
                break

            # Check for NaN/Inf
            if np.isnan(rtw_action).any() or np.isinf(rtw_action).any():
                print(f"  Step {step}: NaN/Inf in action!")
                result["action_nan_inf"] += 1
                result["termination"] = "nan_inf_action"
                break

            # Execute
            try:
                task_env.take_action(rtw_action, action_type='ee')
            except Exception as e:
                print(f"  Step {step}: Controller error: {e}")
                result["controller_errors"] += 1
                result["termination"] = "controller_error"
                break

            # Track
            left_dxyz = float(action_data.get("left_dxyz_norm", 0))
            right_dxyz = float(action_data.get("right_dxyz_norm", 0))
            result["left_dxyz_trace"].append(left_dxyz)
            result["right_dxyz_trace"].append(right_dxyz)

            # Get switch qpos
            qpos = get_switch_qpos(task_env)
            if qpos is not None:
                result["switch_qpos_trace"].append(float(qpos))
                if qpos > result["Ymax"]:
                    result["Ymax"] = float(qpos)

            result["episode_length"] = step + 1

            # Check success
            if qpos is not None and check_success(qpos):
                result["success"] = True
                result["success_timestep"] = step
                result["success_margin"] = float(qpos - SUCCESS_THRESHOLD)
                result["termination"] = "success"
                print(f"  ✅ Step {step}: SUCCESS! Ymax={qpos:.6f}, margin={result['success_margin']:.6f}")
                break

            if step % 50 == 0 and step > 0:
                qp_str = f"qpos={qpos:.6f}" if qpos is not None else "qpos=N/A"
                print(f"  Step {step}: Δleft={left_dxyz:.4f}m, Δright={right_dxyz:.4f}m, {qp_str}")

    finally:
        task_env.close()
        vam.close()

    # Final summary
    result["success_margin"] = float(result["Ymax"] - SUCCESS_THRESHOLD)
    status = "✅ SUCCESS" if result["success"] else "❌ FAIL"
    print(f"  Result: {status} | Ymax={result['Ymax']:.6f} | margin={result['success_margin']:.6f} | "
          f"steps={result['episode_length']} | term={result['termination']}")

    # Save individual result
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{checkpoint_name}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ── Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.path.join(OUTPUT_DIR, "closed_loop"))
    parser.add_argument("--checkpoints", nargs="+",
                        default=["pretrained", "step100", "step300", "step800"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--single", type=str, default=None,
                        help="Run single checkpoint for quick test")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ckpts_to_run = [args.single] if args.single else args.checkpoints

    print("=" * 70)
    print("PB-B Phase E1: Closed-Loop Development Evaluation")
    print(f"Task: {TASK}")
    print(f"Seeds: {args.seeds}")
    print(f"Checkpoints: {ckpts_to_run}")
    print(f"Success threshold: {SUCCESS_THRESHOLD}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    all_results = []
    results_by_ckpt = {}

    for ckpt_name in ckpts_to_run:
        ckpt_info = CHECKPOINTS.get(ckpt_name)
        if not ckpt_info:
            print(f"  ❌ Unknown checkpoint: {ckpt_name}")
            continue

        ckpt_results = []
        for seed in args.seeds:
            result = run_episode(ckpt_name, ckpt_info, seed, args.output_dir)
            all_results.append(result)
            ckpt_results.append(result)
        results_by_ckpt[ckpt_name] = ckpt_results

    # ── Summary ──
    print(f"\n{'='*70}")
    print("CLOSED-LOOP SUMMARY")
    print(f"{'='*70}")
    print(f"{'Checkpoint':<15} {'Success':>8} {'Ymax_mean':>12} {'Best_Margin':>14}")
    print("-" * 55)

    for ckpt_name in ckpts_to_run:
        cr = results_by_ckpt.get(ckpt_name, [])
        n_success = sum(1 for r in cr if r["success"])
        ym_values = [r["Ymax"] for r in cr if r["Ymax"] > 0]
        ym_mean = np.mean(ym_values) if ym_values else 0
        margins = [r["success_margin"] for r in cr]
        best_margin = max(margins) if margins else -999

        print(f"{ckpt_name:<15} {n_success}/{len(cr):<6} {ym_mean:>12.6f} {best_margin:>14.6f}")

    # Save summary
    summary_path = os.path.join(args.output_dir, "closed_loop_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "threshold": SUCCESS_THRESHOLD,
            "seeds": args.seeds,
            "per_checkpoint": {
                ckpt: {
                    "success_rate": f"{sum(1 for r in results if r['success'])}/{len(results)}",
                    "Ymax_mean": float(np.mean([r['Ymax'] for r in results if r['Ymax'] > 0] or [0])),
                    "best_margin": float(max([r['success_margin'] for r in results] or [-999])),
                    "episodes": results,
                }
                for ckpt, results in results_by_ckpt.items()
            }
        }, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_path}")

    # Save JSONL
    jsonl_path = os.path.join(args.output_dir, "closed_loop_results.jsonl")
    with open(jsonl_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"JSONL saved to: {jsonl_path}")


if __name__ == "__main__":
    main()
