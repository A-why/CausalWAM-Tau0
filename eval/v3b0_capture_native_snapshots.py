#!/usr/bin/env python3
"""V3-B0 Phase A: capture simulator-NATIVE snapshots S0-S3 from a current-physics
successful native expert episode (NOT EE replay).

Runs the official RoboTwin turn_switch expert (seed 0, same as PB-A) and, at four
switch-progress states, dumps the FULL simulator state: every scene articulation
(qpos/qvel/root pose/root velocity), every actor (pose/velocity), the switch, and the
RNG state. Also saves the observation (endpose + cameras) at each snapshot so the Tau
side can condition the SDE sampler on the exact restored state.

Snapshots (indices derived dynamically from the switch-qpos trace):
    S0 = last step with switch qpos == 0 before the ramp (switch OFF, pre-contact)
    S1 = first step with switch qpos > 0.01 (contact, near floor)
    S2 = first step with switch qpos >= 0.5 * threshold (partial progress)
    S3 = first step with switch qpos >= threshold - 0.005 (near threshold, pre-success)

No VAM. No ACVS. No training. optimizer.step = 0.

Usage:
    cd ${ROBOTWIN_ROOT} && DISPLAY=:99 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
    /opt/conda/envs/robotwin/bin/python ${CAUSALWAM_ROOT}/eval/v3b0_capture_native_snapshots.py
"""
import sys, os, json, pickle, argparse, random, yaml
import numpy as np

ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT", "/data/QWW/RoboTwin")
CAUSAL_ROOT = os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROBOTWIN_ROOT)
sys.path.insert(0, os.path.join(ROBOTWIN_ROOT, "description/utils"))

TASK = "turn_switch"
OUT_ROOT = os.path.join(CAUSAL_ROOT, "outputs/v3b0_outcome_contract")
SNAP_DIR = os.path.join(OUT_ROOT, "native_snapshots")
SEED = 0


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
            state["cameras"][cam_name] = np.asarray(rgb)
    return state


def _pose_list(pose):
    if pose is None:
        return None
    p = pose.p
    q = pose.q
    return [float(p[0]), float(p[1]), float(p[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def capture_full_state(task_env):
    """Dump the complete simulator state. All values are plain Python/numpy (picklable)."""
    state = {}

    # --- switch (ArticulationActor) ---
    sw = task_env.switch
    sw_state = {
        "name": sw.get_name() if hasattr(sw, "get_name") else "056_switch",
        "qpos": np.asarray(sw.get_qpos(), dtype=np.float64).tolist(),
    }
    for attr, key in [("get_qvel", "qvel"), ("get_qlimits", "qlimits")]:
        try:
            sw_state[key] = np.asarray(getattr(sw, attr)(), dtype=np.float64).tolist()
        except Exception:
            sw_state[key] = None
    try:
        sw_state["qlimits_0"] = np.asarray(sw.get_qlimits()[0], dtype=np.float64).tolist()
    except Exception:
        sw_state["qlimits_0"] = None
    try:
        sw_state["pose"] = _pose_list(sw.get_pose())
    except Exception:
        sw_state["pose"] = None
    state["switch"] = sw_state

    # --- robot (dual-arm PhysxArticulation) ---
    rob = task_env.robot._entity
    rob_state = {}
    try:
        rob_state["qpos"] = np.asarray(rob.get_qpos(), dtype=np.float64).tolist()
    except Exception:
        rob_state["qpos"] = None
    try:
        rob_state["qvel"] = np.asarray(rob.get_qvel(), dtype=np.float64).tolist()
    except Exception:
        rob_state["qvel"] = None
    try:
        rob_state["root_pose"] = _pose_list(rob.get_root_pose())
    except Exception:
        rob_state["root_pose"] = None
    try:
        rob_state["root_linear_velocity"] = np.asarray(rob.get_root_linear_velocity(), dtype=np.float64).tolist()
    except Exception:
        rob_state["root_linear_velocity"] = None
    try:
        rob_state["root_angular_velocity"] = np.asarray(rob.get_root_angular_velocity(), dtype=np.float64).tolist()
    except Exception:
        rob_state["root_angular_velocity"] = None
    rob_state["name"] = rob.get_name() if hasattr(rob, "get_name") else "robot"
    state["robot"] = rob_state

    # --- every scene articulation ---
    articulations = {}
    try:
        arts = task_env.scene.get_all_articulations()
    except Exception:
        arts = []
    for a in arts:
        name = a.get_name() if hasattr(a, "get_name") else str(a.get_id())
        entry = {}
        try:
            entry["qpos"] = np.asarray(a.get_qpos(), dtype=np.float64).tolist()
        except Exception:
            entry["qpos"] = None
        try:
            entry["qvel"] = np.asarray(a.get_qvel(), dtype=np.float64).tolist()
        except Exception:
            entry["qvel"] = None
        try:
            entry["root_pose"] = _pose_list(a.get_root_pose())
        except Exception:
            entry["root_pose"] = None
        articulations[name] = entry
    state["articulations"] = articulations

    # --- every scene actor (rigid bodies: table/wall/ground/etc.) ---
    actors = {}
    try:
        acts = task_env.scene.get_all_actors()
    except Exception:
        acts = []
    for a in acts:
        name = a.get_name() if hasattr(a, "get_name") else str(a.get_id())
        entry = {}
        try:
            entry["pose"] = _pose_list(a.get_pose())
        except Exception:
            entry["pose"] = None
        try:
            entry["linear_velocity"] = np.asarray(a.get_linear_velocity(), dtype=np.float64).tolist()
        except Exception:
            entry["linear_velocity"] = None
        try:
            entry["angular_velocity"] = np.asarray(a.get_angular_velocity(), dtype=np.float64).tolist()
        except Exception:
            entry["angular_velocity"] = None
        actors[name] = entry
    state["actors"] = actors

    # --- RNG state ---
    state["rng"] = {
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    return state


def run_episode(record_at=None):
    """Run the native expert once. Return (switch_trace, ee_trace, snapshots dict, limits).

    If record_at is a set of step indices, capture full state at those indices.
    """
    task_env = init_task_env(SEED)
    threshold = None
    try:
        limit = task_env.switch.get_qlimits()[0]
        limit_upper = float(limit[1])
        threshold = limit_upper - 0.05
    except Exception:
        pass

    original_step = task_env.scene.step
    switch_trace = []
    ee_trace = []
    snapshots = {}
    step_counter = [0]

    def recording_step():
        idx = step_counter[0]
        q = float(task_env.switch.get_qpos()[0])
        switch_trace.append(q)
        ee_trace.append({
            "left_ee": list(task_env.robot.get_left_ee_pose()),
            "right_ee": list(task_env.robot.get_right_ee_pose()),
        })
        if record_at is not None and idx in record_at:
            snapshots[idx] = capture_full_state(task_env)
            snapshots[idx]["observation"] = get_obs_dict(task_env)
            snapshots[idx]["switch_qpos"] = q
        step_counter[0] += 1
        return original_step()

    task_env.scene.step = recording_step
    try:
        task_env.play_once()
    finally:
        task_env.scene.step = original_step

    success = task_env.check_success()
    final_qpos = float(task_env.switch.get_qpos()[0])
    model_id = task_env.model_id if hasattr(task_env, "model_id") else None
    model_name = task_env.model_name if hasattr(task_env, "model_name") else None
    task_env.close()
    return {
        "switch_trace": switch_trace,
        "ee_trace": ee_trace,
        "snapshots": snapshots,
        "threshold": threshold,
        "limit_upper": limit_upper,
        "success": success,
        "final_qpos": final_qpos,
        "model_id": model_id,
        "model_name": model_name,
    }


def select_snapshot_indices(switch_trace, threshold):
    """Dynamically pick S0-S3 from the switch qpos trace."""
    q = np.asarray(switch_trace)
    ramp_start = int(np.where(q > 1e-6)[0][0]) if (q > 1e-6).any() else len(q) - 1
    s0 = max(0, ramp_start - 1)

    def first_ge(arr, val):
        idx = np.where(arr >= val)[0]
        return int(idx[0]) if len(idx) else len(arr) - 1

    s1 = first_ge(q, 0.01)
    s2 = first_ge(q, 0.5 * threshold)
    s3 = first_ge(q, threshold - 0.005)
    return {"S0": s0, "S1": s1, "S2": s2, "S3": s3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=SNAP_DIR)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- Pass 1: record switch trace, derive S0-S3 indices ----
    print("Pass 1: recording native expert switch trajectory...", flush=True)
    r1 = run_episode(record_at=None)
    q = np.asarray(r1["switch_trace"])
    print(f"  n_steps={len(q)}  success={r1['success']}  final_qpos={r1['final_qpos']:.6f}  "
          f"limit_upper={r1['limit_upper']:.6f}  threshold={r1['threshold']:.6f}", flush=True)
    if not r1["success"]:
        print("  WARNING: seed0 episode did not reach success; snapshots may be degenerate.", flush=True)

    indices = select_snapshot_indices(q, r1["threshold"])
    print("  S0-S3 step indices:", indices, flush=True)
    for name, idx in indices.items():
        print(f"    {name}: step={idx}  switch_qpos={q[idx]:.6f}", flush=True)

    # ---- Pass 2: re-run, dump full state at S0-S3 ----
    print("Pass 2: re-running and dumping full state at S0-S3...", flush=True)
    r2 = run_episode(record_at=set(indices.values()))
    snapshots = r2["snapshots"]

    manifest = {
        "phase": "V3-B0 native snapshot capture",
        "seed": SEED,
        "task": TASK,
        "model_name": r2["model_name"],
        "model_id": r2["model_id"],
        "limit_upper": r2["limit_upper"],
        "threshold": r2["threshold"],
        "success": r2["success"],
        "final_qpos": r2["final_qpos"],
        "n_steps": len(r2["switch_trace"]),
        "snapshot_indices": {k: int(v) for k, v in indices.items()},
        "snapshot_switch_qpos": {k: float(q[v]) for k, v in indices.items()},
        "snapshot_definitions": {
            "S0": "last step with switch qpos == 0 before the ramp (switch OFF, pre-contact)",
            "S1": "first step with switch qpos > 0.01 (contact, near floor)",
            "S2": "first step with switch qpos >= 0.5*threshold (partial progress)",
            "S3": "first step with switch qpos >= threshold-0.005 (near threshold, pre-success)",
        },
        "optimizer_step": 0,
        "training": False,
    }

    for name, idx in indices.items():
        if idx not in snapshots:
            print(f"  ERROR: snapshot {name} (idx={idx}) missing from pass 2", flush=True)
            continue
        out_path = os.path.join(args.out, f"{name}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump({"snapshot": name, "step_index": int(idx),
                         "switch_qpos": float(q[idx]), "full_state": snapshots[idx]}, f)
        print(f"  saved {name}.pkl (step {idx}, switch_qpos {q[idx]:.6f}, "
              f"n_articulations={len(snapshots[idx].get('articulations', {}))}, "
              f"n_actors={len(snapshots[idx].get('actors', {}))})", flush=True)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  manifest saved -> {args.out}/manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
