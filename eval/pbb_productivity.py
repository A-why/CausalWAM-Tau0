#!/usr/bin/env python3
"""PB-B Phase E2: Interaction Snapshot Productivity Test.

Selects I0/I1/I2 snapshots from PB-A trajectory replay,
generates K=4 action candidates per checkpoint×snapshot,
executes each in real environment, computes G_true.

Usage:
    cd ${ROBOTWIN_ROOT} && DISPLAY=:99 PYTHONUNBUFFERED=1 \
    /opt/conda/envs/robotwin/bin/python ${CAUSALWAM_ROOT}/eval/pbb_productivity.py
"""
import sys, os, json, time, struct, pickle, argparse, yaml
import numpy as np

# ── Paths ──────────────────────────────────────────────────
ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT", "/data/QWW/RoboTwin")
CAUSAL_ROOT = os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAU0_ROOT = os.path.join(CAUSAL_ROOT, "tau-0-wm")

sys.path.insert(0, ROBOTWIN_ROOT)
sys.path.insert(0, os.path.join(ROBOTWIN_ROOT, "description/utils"))

# ── Constants ──────────────────────────────────────────────
TASK = "turn_switch"
SUCCESS_THRESHOLD = 0.14198621809482576
K = 4  # candidates per snapshot×checkpoint
CANDIDATE_SEEDS = [42, 43, 44, 45]

# Snapshots from seed0 trajectory (downsampled frames at ~15.625Hz)
# I0: pre-contact (ds=12, qpos=0), I1: pre-switch-motion (ds=44, qpos=0),
# I2: post-success (ds=64, qpos=0.192)
SNAPSHOTS = {
    "I0": {"ds_frame": 12, "label": "pre-contact", "qpos_at": 0.0},
    "I1": {"ds_frame": 44, "label": "approach/contact", "qpos_at": 0.0},
    "I2": {"ds_frame": 64, "label": "post-success", "qpos_at": 0.192},
}

CHECKPOINTS = {
    "pretrained": {
        "path": os.path.join(CAUSAL_ROOT, "checkpoints/tau0_wm/vam"),
        "action_type": "absolute",
    },
    "step100": {
        "path": os.path.join(CAUSAL_ROOT, "outputs/pbb/turn_switch/2026_08_12_02_29_06/step_100"),
        "action_type": "relative",
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
EEF_TRAJECTORY = os.path.join(CAUSAL_ROOT, "datasets/robotwin_current_success_raw/turn_switch/seed0_eef_trajectory.npz")


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


def replay_eef_to_frame(task_env, eef_data, target_ds_frame):
    """Replay EEF trajectory up to target downsampled frame to reach snapshot."""
    DOWN = 16  # 250Hz -> 15.625Hz
    left_ee = eef_data["left_ee"]
    right_ee = eef_data["right_ee"]
    left_gripper = eef_data["left_gripper"]
    right_gripper = eef_data["right_gripper"]
    arm_used = str(eef_data["arm_used"])

    for ds in range(target_ds_frame + 1):
        raw_idx = ds * DOWN
        if raw_idx >= len(left_ee):
            break

        # Build EE action [left_xyz(3), left_quat_wxyz(4), left_grip(1),
        #                right_xyz(3), right_quat_wxyz(4), right_grip(1)]
        lee = left_ee[raw_idx]
        ree = right_ee[raw_idx]
        lg = left_gripper[raw_idx]
        rg = right_gripper[raw_idx]

        ee_action = np.array([
            lee[0], lee[1], lee[2],  # left xyz
            lee[6], lee[3], lee[4], lee[5],  # left quat xyzw -> wxyz
            lg,
            ree[0], ree[1], ree[2],  # right xyz
            ree[6], ree[3], ree[4], ree[5],  # right quat xyzw -> wxyz
            rg,
        ], dtype=np.float32)

        try:
            task_env.take_action(ee_action, action_type='ee')
        except Exception as e:
            print(f"    WARNING: take_action failed at ds={ds}: {e}")
            break

    return get_obs_dict(task_env)


# ── VAM Inference Subprocess ───────────────────────────────

class VAMClient:
    """Manages VAM inference subprocess in tau0_wm env."""

    def __init__(self, checkpoint_path, action_type=None):
        vam_server_path = os.path.join(CAUSAL_ROOT, "eval/vam_server.py")
        env = os.environ.copy()
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
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env,
        )
        start_t = time.time()
        while time.time() - start_t < 180:
            line = self.proc.stdout.readline()
            if b"VAM_READY" in line:
                break
            if self.proc.poll() is not None:
                stderr_data = self.proc.stderr.read().decode()
                raise RuntimeError(f"VAM died. stderr={stderr_data[:500]}")

    def infer(self, obs_dict, seed=None):
        """Send observation, receive action."""
        if seed is not None:
            obs_dict = dict(obs_dict)
            obs_dict["seed"] = seed
        obs_bytes = pickle.dumps(obs_dict)
        self.proc.stdin.write(struct.pack(">I", len(obs_bytes)))
        self.proc.stdin.write(obs_bytes)
        self.proc.stdin.flush()

        action_len_bytes = self.proc.stdout.read(4)
        if len(action_len_bytes) < 4:
            raise RuntimeError("VAM process died")
        action_len = struct.unpack(">I", action_len_bytes)[0]
        action_bytes = self.proc.stdout.read(action_len)
        return pickle.loads(action_bytes)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.stdout.close()
            try:
                self.proc.wait(timeout=10)
            except:
                self.proc.kill()
            self.proc = None


# ── Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",
                        default=os.path.join(CAUSAL_ROOT, "outputs/pbb_productive_policy/evaluation/productivity"))
    parser.add_argument("--checkpoints", nargs="+",
                        default=["pretrained", "step100", "step300", "step800"])
    parser.add_argument("--snapshots", nargs="+", default=["I0", "I1", "I2"])
    parser.add_argument("--k", type=int, default=K)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("PB-B Phase E2: Interaction Snapshot Productivity Test")
    print(f"Task: {TASK}")
    print(f"Snapshots: {list(SNAPSHOTS.keys())}")
    print(f"Checkpoints: {args.checkpoints}")
    print(f"K={args.k} candidates per snapshot×checkpoint")
    print(f"Success threshold: {SUCCESS_THRESHOLD}")
    print("=" * 70)

    # Load EEF trajectory
    eef_data = np.load(EEF_TRAJECTORY, allow_pickle=True)
    print(f"\nLoaded EEF trajectory: {len(eef_data['left_ee'])} frames, arm={eef_data['arm_used']}")

    all_results = []

    for ckpt_name in args.checkpoints:
        ckpt_info = CHECKPOINTS[ckpt_name]
        print(f"\n{'='*70}")
        print(f"CHECKPOINT: {ckpt_name} (action_type={ckpt_info['action_type']})")
        print(f"{'='*70}")

        # Start VAM
        print(f"  Starting VAM subprocess...")
        try:
            vam = VAMClient(ckpt_info["path"], action_type=ckpt_info["action_type"])
        except Exception as e:
            print(f"  ❌ VAM init failed: {e}")
            continue

        try:
            for snap_name in args.snapshots:
                snap_info = SNAPSHOTS[snap_name]
                print(f"\n  --- Snapshot {snap_name} ({snap_info['label']}, "
                      f"ds_frame={snap_info['ds_frame']}) ---")

                # Initialize fresh env and replay to snapshot
                print(f"    Replaying expert to ds_frame={snap_info['ds_frame']}...")
                task_env = init_task_env(0)  # seed 0 matches trajectory
                try:
                    obs = replay_eef_to_frame(task_env, eef_data, snap_info['ds_frame'])
                    initial_qpos = get_switch_qpos(task_env)
                    print(f"    Initial qpos: {initial_qpos}")

                    # Generate K candidates
                    candidates = []
                    for seed_idx, seed in enumerate(CANDIDATE_SEEDS[:args.k]):
                        print(f"    Candidate {seed_idx+1}/{args.k} (seed={seed})...", end=" ", flush=True)
                        try:
                            action_data = vam.infer(obs, seed=seed)
                            if action_data.get("error"):
                                print(f"VAM error: {action_data['error']}")
                                candidates.append({"error": action_data["error"]})
                                continue

                            rtw_action = np.array(action_data["robotwin_action"], dtype=np.float32)
                            left_d = action_data.get("left_dxyz_norm", 0)
                            right_d = action_data.get("right_dxyz_norm", 0)

                            # Execute in fresh env copy (replay to same snapshot, then take VAM action)
                            task_env2 = init_task_env(0)
                            _ = replay_eef_to_frame(task_env2, eef_data, snap_info['ds_frame'])
                            pre_qpos = get_switch_qpos(task_env2)

                            try:
                                task_env2.take_action(rtw_action, action_type='ee')
                            except Exception as e:
                                print(f"exec error: {e}")

                            post_qpos = get_switch_qpos(task_env2)
                            pre_qpos = pre_qpos if pre_qpos is not None else 0.0
                            post_qpos = post_qpos if post_qpos is not None else pre_qpos

                            candidate = {
                                "seed": seed,
                                "left_dxyz": float(left_d),
                                "right_dxyz": float(right_d),
                                "pre_qpos": float(pre_qpos),
                                "post_qpos": float(post_qpos),
                                "delta_qpos": float(post_qpos - pre_qpos),
                                "success": float(post_qpos) >= SUCCESS_THRESHOLD,
                                "action": rtw_action.tolist(),
                            }
                            candidates.append(candidate)
                            print(f"ΔL={left_d:.3f}m ΔR={right_d:.3f}m "
                                  f"qpos: {pre_qpos:.6f}→{post_qpos:.6f} ({'+' if post_qpos>=pre_qpos else ''}{post_qpos-pre_qpos:.6f})")
                        except Exception as e:
                            print(f"error: {e}")
                            candidates.append({"error": str(e)})
                        finally:
                            if 'task_env2' in dir():
                                task_env2.close()

                    # Compute productivity metrics
                    post_qpos_values = [c.get("post_qpos", 0) for c in candidates if "error" not in c]
                    success_values = [c.get("success", False) for c in candidates if "error" not in c]
                    n_success = sum(success_values)
                    n_fail = len(success_values) - n_success

                    if len(post_qpos_values) >= 2:
                        qpos_range = max(post_qpos_values) - min(post_qpos_values)
                        qpos_std = float(np.std(post_qpos_values))
                    else:
                        qpos_range = 0.0
                        qpos_std = 0.0

                    # Classify productivity
                    if n_success > 0 and n_fail > 0:
                        group_type = "mixed-sign"
                    elif n_success == 0 and qpos_range > 1e-6:
                        group_type = "informative-all-bad"
                    elif n_success == len(success_values) and qpos_range > 1e-6:
                        group_type = "informative-all-good"
                    elif qpos_range <= 1e-6:
                        group_type = "degenerate-floor"
                    else:
                        group_type = "non-degenerate"

                    result = {
                        "checkpoint": ckpt_name,
                        "snapshot": snap_name,
                        "snapshot_label": snap_info["label"],
                        "ds_frame": snap_info["ds_frame"],
                        "initial_qpos": float(initial_qpos) if initial_qpos is not None else None,
                        "candidates": candidates,
                        "n_success": n_success,
                        "n_fail": n_fail,
                        "qpos_range": float(qpos_range),
                        "qpos_std": float(qpos_std),
                        "group_type": group_type,
                    }
                    all_results.append(result)

                    print(f"    → {group_type}: {n_success}S/{n_fail}F, "
                          f"qpos_range={qpos_range:.6f}, qpos_std={qpos_std:.6f}")

                finally:
                    task_env.close()

        finally:
            vam.close()

    # ── Summary ──
    print(f"\n{'='*70}")
    print("PHASE E2: PRODUCTIVITY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Snapshot':<10} {'Checkpoint':<15} {'Group_Type':<22} {'S/F':<8} {'Qpos_Range':>12} {'Qpos_Std':>12}")
    print("-" * 75)

    productive_groups = 0
    informative_groups = 0
    degenerate_groups = 0

    for r in all_results:
        print(f"{r['snapshot']:<10} {r['checkpoint']:<15} {r['group_type']:<22} "
              f"{r['n_success']}/{r['n_fail']:<6} {r['qpos_range']:>12.6f} {r['qpos_std']:>12.6f}")
        if r['group_type'] == 'mixed-sign':
            productive_groups += 1
        elif r['group_type'].startswith('informative'):
            informative_groups += 1
        elif r['group_type'] == 'degenerate-floor':
            degenerate_groups += 1

    print(f"\nProductive (mixed-sign): {productive_groups}")
    print(f"Informative: {informative_groups}")
    print(f"Degenerate/Floor: {degenerate_groups}")

    # Select theta_prod
    # Priority: mixed-sign > informative > non-degenerate > floor
    theta_prod = None
    for r in all_results:
        if r['group_type'] == 'mixed-sign':
            theta_prod = r['checkpoint']
            break
    if theta_prod is None:
        for r in all_results:
            if r['group_type'].startswith('informative'):
                theta_prod = r['checkpoint']
                break
    if theta_prod is None:
        for r in all_results:
            if r['group_type'] != 'degenerate-floor':
                theta_prod = r['checkpoint']
                break
    if theta_prod is None:
        theta_prod = "NONE — universal floor"

    print(f"\nθ_prod selection: {theta_prod}")

    # Save results
    out_path = os.path.join(args.output_dir, "productivity_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "K": args.k,
            "candidate_seeds": CANDIDATE_SEEDS[:args.k],
            "snapshots": SNAPSHOTS,
            "success_threshold": SUCCESS_THRESHOLD,
            "theta_prod": theta_prod,
            "productive_groups": productive_groups,
            "informative_groups": informative_groups,
            "degenerate_groups": degenerate_groups,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    return 0 if theta_prod is not None else 1


if __name__ == "__main__":
    sys.exit(main())
