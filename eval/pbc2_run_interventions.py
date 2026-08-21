#!/usr/bin/env python3
"""PB-C2 Phase C: real-environment paired interventions (RoboTwin env, SAPIEN).

For each snapshot I0/I1/I2, runs one Hold reference and K=16 candidate
interventions (33-step action chunks, generated in Phase B). Records the switch
qpos trajectory over the horizon; Y = max qpos, G_true = Y_i - Y0.

No VAM. No ACVS. No training. GPU fault candidates are re-run (Section 28).

Usage:
    cd ${ROBOTWIN_ROOT} && DISPLAY=:99 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
    /opt/conda/envs/robotwin/bin/python ${CAUSALWAM_ROOT}/eval/pbc2_run_interventions.py
"""
import sys, os, json, argparse, yaml
import numpy as np

ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT", "/data/QWW/RoboTwin")
CAUSAL_ROOT = os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROBOTWIN_ROOT)
sys.path.insert(0, os.path.join(ROBOTWIN_ROOT, "description/utils"))

TASK = "turn_switch"
EEF_TRAJECTORY = os.path.join(CAUSAL_ROOT, "datasets/robotwin_current_success_raw/turn_switch/seed0_eef_trajectory.npz")
OUT_ROOT = os.path.join(CAUSAL_ROOT, "outputs/pbc2_sde_productivity")
CAND_JSONL = os.path.join(OUT_ROOT, "sde_candidates", "candidates.jsonl")
ROLLOUT_DIR = os.path.join(OUT_ROOT, "paired_rollouts")
HORIZON = 33  # candidate + reference horizon (must be identical)

SNAPSHOTS = {"I0": 12, "I1": 44, "I2": 64}
MAX_RETRIES = 2  # re-run a candidate whose rollout hit an infra fault


def init_task_env(seed):
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
    try:
        if hasattr(task_env, 'switch'):
            qpos = task_env.switch.get_qpos()
            if qpos is not None and len(qpos) > 0:
                return float(qpos[0])
    except Exception:
        pass
    return None


def replay_eef_to_frame(task_env, eef_data, target_ds_frame):
    DOWN = 16
    left_ee = eef_data["left_ee"]
    right_ee = eef_data["right_ee"]
    left_gripper = eef_data["left_gripper"]
    right_gripper = eef_data["right_gripper"]
    for ds in range(target_ds_frame + 1):
        raw_idx = ds * DOWN
        if raw_idx >= len(left_ee):
            break
        lee = left_ee[raw_idx]
        ree = right_ee[raw_idx]
        lg = left_gripper[raw_idx]
        rg = right_gripper[raw_idx]
        ee_action = np.array([
            lee[0], lee[1], lee[2], lee[6], lee[3], lee[4], lee[5], lg,
            ree[0], ree[1], ree[2], ree[6], ree[3], ree[4], ree[5], rg,
        ], dtype=np.float32)
        task_env.take_action(ee_action, action_type='ee')
    return get_obs_dict(task_env)


def build_hold_action(obs):
    """Hold current pose (16-dim) from snapshot observation endpose."""
    ep = obs["endpose"]
    left = np.asarray(ep["left_endpose"], dtype=np.float32)   # xyz + quat_wxyz
    right = np.asarray(ep["right_endpose"], dtype=np.float32)
    a = np.zeros(16, dtype=np.float32)
    a[0:7] = left
    a[7] = np.float32(ep["left_gripper"])
    a[8:15] = right
    a[15] = np.float32(ep["right_gripper"])
    return a


def run_rollout(task_env, action_chunk_16):
    """Execute a 33-step chunk (33,16) open-loop, record qpos trace. Return Y, trace, error."""
    trace = []
    for t in range(HORIZON):
        try:
            task_env.take_action(action_chunk_16[t], action_type='ee')
        except Exception as e:
            return None, trace, f"take_action step {t}: {e}"
        q = get_switch_qpos(task_env)
        trace.append(q if q is not None else None)
    ys = [q for q in trace if q is not None]
    Y = float(max(ys)) if ys else 0.0
    return Y, trace, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", nargs="+", default=["I0", "I1", "I2"])
    parser.add_argument("--only-reference", action="store_true")
    args = parser.parse_args()
    os.makedirs(ROLLOUT_DIR, exist_ok=True)

    # load candidates
    candidates = []
    with open(CAND_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates")

    # load EEF trajectory + PB-A contract threshold (NOT hardcoded)
    eef_data = np.load(EEF_TRAJECTORY, allow_pickle=True)
    SUCCESS_THRESHOLD = float(eef_data["success_threshold"])
    print(f"success_threshold (PB-A contract): {SUCCESS_THRESHOLD!r}")

    records = []
    references = {}

    for snap in args.snapshots:
        ds = SNAPSHOTS[snap]
        print(f"\n{'='*60}\nSnapshot {snap} (ds_frame={ds})\n{'='*60}")

        # ---- Reference: hold current pose ----
        ref_Y0 = None
        for attempt in range(MAX_RETRIES + 1):
            task_env = init_task_env(0)
            try:
                obs = replay_eef_to_frame(task_env, eef_data, ds)
                pre_qpos = get_switch_qpos(task_env)
                hold = build_hold_action(obs)
                Y0, trace, err = run_rollout(task_env, np.tile(hold[None, :], (HORIZON, 1)))
                if err is not None:
                    raise RuntimeError(err)
                ref_Y0 = Y0
                task_env.close()
                break
            except Exception as e:
                task_env.close()
                print(f"  reference attempt {attempt+1} failed: {e}")
                if attempt == MAX_RETRIES:
                    ref_Y0 = None
        references[snap] = {"Y0": ref_Y0, "pre_qpos": pre_qpos}
        print(f"  Reference (hold): Y0={ref_Y0}")
        with open(os.path.join(ROLLOUT_DIR, f"{snap}_reference.json"), "w") as f:
            json.dump({"snapshot": snap, "ds_frame": ds, "Y0": ref_Y0,
                       "qpos_trace": trace, "success_threshold": SUCCESS_THRESHOLD}, f)

        if args.only_reference:
            continue

        # ---- Candidates ----
        snap_cands = [c for c in candidates if c["snapshot_id"] == snap]
        for cand in snap_cands:
            cid = cand["candidate_id"]
            rtw_chunk = np.asarray(cand["robotwin_absolute_action"], dtype=np.float32)  # (33,16)
            rec = None
            for attempt in range(MAX_RETRIES + 1):
                task_env = init_task_env(0)
                try:
                    obs = replay_eef_to_frame(task_env, eef_data, ds)
                    pre_qpos = get_switch_qpos(task_env)
                    Y, trace, err = run_rollout(task_env, rtw_chunk)
                    if err is not None:
                        raise RuntimeError(err)
                    post = Y
                    rec = {
                        "candidate_id": cid,
                        "snapshot_id": snap,
                        "sde_rng_seed": cand["sde_rng_seed"],
                        "trajectory_hash": cand["trajectory_hash"],
                        "pre_qpos": pre_qpos,
                        "Y": Y,
                        "Y0": ref_Y0,
                        "G_true": float(Y - ref_Y0) if ref_Y0 is not None else None,
                        "success": bool(Y >= SUCCESS_THRESHOLD),
                        "qpos_trace": trace,
                        "attempts": attempt + 1,
                        "error": None,
                    }
                    task_env.close()
                    break
                except Exception as e:
                    task_env.close()
                    print(f"  {cid} attempt {attempt+1} fault: {e}")
                    rec = {"candidate_id": cid, "snapshot_id": snap,
                           "sde_rng_seed": cand["sde_rng_seed"], "error": str(e),
                           "Y": None, "G_true": None}
            records.append(rec)
            if rec.get("error") is None:
                print(f"  {cid}: pre={rec['pre_qpos']:.6f} Y={rec['Y']:.6f} "
                      f"G_true={rec['G_true']:.6f} success={rec['success']}", flush=True)
            else:
                print(f"  {cid}: ERROR {rec['error'][:80]}", flush=True)

            # persist incrementally
            with open(os.path.join(ROLLOUT_DIR, f"{cid}.json"), "w") as f:
                json.dump(rec, f)

    # ---- summary jsonl ----
    out_jsonl = os.path.join(OUT_ROOT, "productivity_results.jsonl")
    with open(out_jsonl, "w") as f:
        json.dump({"type": "reference", "references": references,
                   "success_threshold": SUCCESS_THRESHOLD}, f)
        f.write("\n")
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(records)} candidate rollouts + references -> {out_jsonl}")

    return 0


if __name__ == "__main__":
    main()
