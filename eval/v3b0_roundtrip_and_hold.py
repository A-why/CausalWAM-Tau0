#!/usr/bin/env python3
"""V3-B0 Phase B: simulator-native snapshot round-trip gate + hold reference.

For each native snapshot S0-S3 (full simulator state), verifies the save->restore
contract and measures the hold-reference outcome Y0 (switch qpos max over H=33 under a
fixed hold action).

Round-trip checks per snapshot:
  1. immediate-fidelity: restore into a fresh env, read back switch qpos/qvel, robot
     qpos, and EEF poses, and compare to the saved values (max abs error).
  2. reproducibility: restore twice into two fresh envs, apply the SAME hold action
     for H=33 steps, and compare the switch-qpos traces A vs B (must be ~identical).
  3. qvel-matters: restore but zero all qvel, apply the same hold, and compare to the
     full restore trace — isolates whether capturing qvel is load-bearing.

Hold reference (Y0) = max switch qpos over H=33 under the fixed hold action, measured
on a full restore. Y0 is the baseline for the K=8 SDE sign probe (G_true = Y_i - Y0).

No VAM. No ACVS. No training. optimizer.step = 0.

Usage:
    cd ${ROBOTWIN_ROOT} && DISPLAY=:99 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
    /opt/conda/envs/robotwin/bin/python ${CAUSALWAM_ROOT}/eval/v3b0_roundtrip_and_hold.py
"""
import sys, os, json, pickle, random, argparse, yaml
import numpy as np

ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT", "/data/QWW/RoboTwin")
CAUSAL_ROOT = os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROBOTWIN_ROOT)
sys.path.insert(0, os.path.join(ROBOTWIN_ROOT, "description/utils"))

TASK = "turn_switch"
OUT_ROOT = os.path.join(CAUSAL_ROOT, "outputs/v3b0_outcome_contract")
SNAP_DIR = os.path.join(OUT_ROOT, "native_snapshots")
SEED = 0
HORIZON = 33
SNAPSHOTS = ["S0", "S1", "S2", "S3"]


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


def get_switch_qpos(task_env):
    try:
        if hasattr(task_env, 'switch'):
            qpos = task_env.switch.get_qpos()
            if qpos is not None and len(qpos) > 0:
                return float(qpos[0])
    except Exception:
        pass
    return None


def build_hold_action(obs):
    ep = obs["endpose"]
    left = np.asarray(ep["left_endpose"], dtype=np.float32)
    right = np.asarray(ep["right_endpose"], dtype=np.float32)
    a = np.zeros(16, dtype=np.float32)
    a[0:7] = left
    a[7] = np.float32(ep["left_gripper"])
    a[8:15] = right
    a[15] = np.float32(ep["right_gripper"])
    return a


def run_rollout(task_env, action_chunk_16):
    """Execute a 33-step chunk (33,16) open-loop, record switch qpos trace."""
    trace = []
    for t in range(HORIZON):
        try:
            task_env.take_action(action_chunk_16[t], action_type='ee')
        except Exception as e:
            return None, trace, f"take_action step {t}: {e}"
        q = get_switch_qpos(task_env)
        trace.append(q if q is not None else None)
    return trace, None


def restore_snapshot(task_env, full_state, zero_qvel=False):
    """Restore full simulator state into task_env. Returns a dict of read-back values."""
    fs = full_state

    # switch
    sw = task_env.switch
    try:
        sw.set_qpos(np.asarray(fs["switch"]["qpos"], dtype=np.float64))
    except Exception as e:
        raise RuntimeError(f"set switch qpos: {e}")
    qvel = fs["switch"].get("qvel")
    try:
        if zero_qvel or qvel is None:
            sw.set_qvel(np.zeros_like(np.asarray(fs["switch"]["qpos"])))
        else:
            sw.set_qvel(np.asarray(qvel, dtype=np.float64))
    except Exception as e:
        raise RuntimeError(f"set switch qvel: {e}")

    # robot articulation
    rob = task_env.robot._entity
    try:
        rob.set_qpos(np.asarray(fs["robot"]["qpos"], dtype=np.float64))
    except Exception as e:
        raise RuntimeError(f"set robot qpos: {e}")
    rqvel = fs["robot"].get("qvel")
    try:
        if zero_qvel or rqvel is None:
            rob.set_qvel(np.zeros_like(np.asarray(fs["robot"]["qpos"])))
        else:
            rob.set_qvel(np.asarray(rqvel, dtype=np.float64))
    except Exception as e:
        raise RuntimeError(f"set robot qvel: {e}")

    # other articulations (best-effort; robot+switch already handled above)
    for name, entry in fs.get("articulations", {}).items():
        for a in task_env.scene.get_all_articulations():
            aname = a.get_name() if hasattr(a, "get_name") else ""
            if aname == name and name not in ("", "056_switch"):
                if entry.get("qpos") is not None:
                    a.set_qpos(np.asarray(entry["qpos"], dtype=np.float64))
                if entry.get("qvel") is not None and not zero_qvel:
                    a.set_qvel(np.asarray(entry["qvel"], dtype=np.float64))

    # actors (best-effort)
    for name, entry in fs.get("actors", {}).items():
        if entry.get("pose") is None:
            continue
        for a in task_env.scene.get_all_actors():
            aname = a.get_name() if hasattr(a, "get_name") else ""
            if aname == name:
                try:
                    p = entry["pose"]
                    import sapien
                    a.set_pose(sapien.Pose([p[0], p[1], p[2]], [p[3], p[4], p[5], p[6]]))
                except Exception:
                    pass

    # RNG
    rng = fs.get("rng", {})
    if rng.get("numpy") is not None:
        try:
            np.random.set_state(rng["numpy"])
        except Exception:
            pass
    if rng.get("random") is not None:
        try:
            random.setstate(rng["random"])
        except Exception:
            pass

    # read back
    readback = {
        "switch_qpos": get_switch_qpos(task_env),
        "switch_qvel": float(np.asarray(sw.get_qvel())[0]) if sw.get_qvel() is not None else None,
        "robot_qpos": np.asarray(rob.get_qpos(), dtype=np.float64).tolist(),
        "left_ee": list(task_env.robot.get_left_ee_pose()),
        "right_ee": list(task_env.robot.get_right_ee_pose()),
    }
    return readback


def max_abs_diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("nan")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", nargs="+", default=SNAPSHOTS)
    args = parser.parse_args()

    roundtrip = {}
    hold_ref = {}

    for snap in args.snapshots:
        with open(os.path.join(SNAP_DIR, f"{snap}.pkl"), "rb") as f:
            snap_data = pickle.load(f)
        fs = snap_data["full_state"]
        step_index = snap_data["step_index"]
        saved_switch_qpos = snap_data["switch_qpos"]
        obs = fs["observation"]
        hold_action = build_hold_action(obs)
        hold_chunk = np.tile(hold_action[None, :], (HORIZON, 1))

        print(f"\n{'='*60}\nSnapshot {snap} (step {step_index}, saved switch_qpos {saved_switch_qpos:.6f})\n{'='*60}")

        # ---- 1. immediate fidelity + reproducibility (two restores) ----
        traces = []
        readbacks = []
        for rep in range(2):
            task_env = init_task_env(SEED)
            try:
                rb = restore_snapshot(task_env, fs, zero_qvel=False)
                readbacks.append(rb)
                trace, err = run_rollout(task_env, hold_chunk)
                if err is not None:
                    raise RuntimeError(err)
                traces.append([q for q in trace if q is not None])
            finally:
                task_env.close()

        # immediate fidelity vs saved
        saved_switch_qvel = float(np.asarray(fs["switch"]["qvel"])[0]) if fs["switch"].get("qvel") is not None else None
        fidelity = {
            "switch_qpos_err": abs(readbacks[0]["switch_qpos"] - saved_switch_qpos),
            "switch_qvel_err": abs(readbacks[0]["switch_qvel"] - saved_switch_qvel) if saved_switch_qvel is not None else None,
            "robot_qpos_err": max_abs_diff(readbacks[0]["robot_qpos"], fs["robot"]["qpos"]),
            "left_ee_err": max_abs_diff(readbacks[0]["left_ee"], obs["endpose"]["left_endpose"]),
            "right_ee_err": max_abs_diff(readbacks[0]["right_ee"], obs["endpose"]["right_endpose"]),
        }

        # reproducibility: trace A vs trace B
        ta = np.asarray(traces[0])
        tb = np.asarray(traces[1])
        reprod_err = float(np.max(np.abs(ta - tb))) if len(ta) == len(tb) else float("nan")
        Y0 = float(np.max(ta)) if ta.size else 0.0

        # ---- 3. qvel-matters ----
        task_env = init_task_env(SEED)
        try:
            restore_snapshot(task_env, fs, zero_qvel=True)
            trace_zq, err = run_rollout(task_env, hold_chunk)
            if err is not None:
                raise RuntimeError(err)
            tzq = np.asarray([q for q in trace_zq if q is not None])
        finally:
            task_env.close()
        qvel_err = float(np.max(np.abs(ta - tzq))) if len(ta) == len(tzq) else float("nan")

        roundtrip[snap] = {
            "step_index": step_index,
            "saved_switch_qpos": saved_switch_qpos,
            "fidelity": fidelity,
            "reproducibility_err": reprod_err,
            "qvel_zero_err": qvel_err,
            "trace_full_restore": traces[0],
            "trace_zero_qvel": [float(x) for x in tzq],
            "verdict_fidelity": "PASS" if fidelity["switch_qpos_err"] < 1e-6 and fidelity["robot_qpos_err"] < 1e-6 else "FAIL",
            "verdict_reproducibility": "PASS" if reprod_err < 1e-6 else "FAIL",
        }
        hold_ref[snap] = {
            "step_index": step_index,
            "saved_switch_qpos": saved_switch_qpos,
            "Y0": Y0,
            "qpos_trace": traces[0],
        }
        print(f"  fidelity: switch_qpos_err={fidelity['switch_qpos_err']:.3e} "
              f"switch_qvel_err={fidelity['switch_qvel_err']:.3e} robot_qpos_err={fidelity['robot_qpos_err']:.3e} "
              f"left_ee_err={fidelity['left_ee_err']:.3e}")
        print(f"  reproducibility_err={reprod_err:.3e}  qvel_zero_err={qvel_err:.3e}")
        print(f"  Y0 (hold reference)={Y0:.6f}  trace={[round(x,4) for x in traces[0][:8]]}...")

    with open(os.path.join(OUT_ROOT, "snapshot_roundtrip.json"), "w") as f:
        json.dump({"horizon": HORIZON, "snapshots": roundtrip}, f, indent=2, default=str)
    with open(os.path.join(OUT_ROOT, "hold_reference.json"), "w") as f:
        json.dump({"horizon": HORIZON, "hold_reference": hold_ref}, f, indent=2, default=str)
    print(f"\nSaved snapshot_roundtrip.json + hold_reference.json -> {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
