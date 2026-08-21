#!/usr/bin/env python3
"""PB-C2 Phase A: capture I0/I1/I2 interaction snapshots (RoboTwin env).

Replays the PB-A successful EEF trajectory to each snapshot frame (ds 12/44/64)
and saves the observation (endpose + cameras) + switch qpos to disk, so the
Tau-side candidate generator (Phase B) can condition on the exact snapshot
without SAPIEN co-residency.

Usage:
    cd ${ROBOTWIN_ROOT} && DISPLAY=:99 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
    /opt/conda/envs/robotwin/bin/python ${CAUSALWAM_ROOT}/eval/pbc2_capture_snapshots.py
"""
import sys, os, json, pickle, argparse, yaml
import numpy as np

ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT", "/data/QWW/RoboTwin")
CAUSAL_ROOT = os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROBOTWIN_ROOT)
sys.path.insert(0, os.path.join(ROBOTWIN_ROOT, "description/utils"))

TASK = "turn_switch"
EEF_TRAJECTORY = os.path.join(CAUSAL_ROOT, "datasets/robotwin_current_success_raw/turn_switch/seed0_eef_trajectory.npz")
OUT_ROOT = os.path.join(CAUSAL_ROOT, "outputs/pbc2_sde_productivity")
SNAP_DIR = os.path.join(OUT_ROOT, "sde_candidates", "snapshots")

# PB-C2 snapshots — MUST match PB-B2 (ds 12/44/64).
SNAPSHOTS = {
    "I0": {"ds_frame": 12},
    "I1": {"ds_frame": 44},
    "I2": {"ds_frame": 64},
}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=SNAP_DIR)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    eef_data = np.load(EEF_TRAJECTORY, allow_pickle=True)
    print(f"Loaded EEF trajectory: {len(eef_data['left_ee'])} frames, arm={eef_data['arm_used']}")

    for snap_name, snap_info in SNAPSHOTS.items():
        ds = snap_info["ds_frame"]
        print(f"Capturing {snap_name} (ds_frame={ds})...")
        task_env = init_task_env(0)
        try:
            obs = replay_eef_to_frame(task_env, eef_data, ds)
            qpos = get_switch_qpos(task_env)
            out_path = os.path.join(args.out, f"{snap_name}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump({"snapshot": snap_name, "ds_frame": ds,
                             "qpos": qpos, "obs": obs}, f)
            print(f"  saved {out_path}  qpos={qpos}")
        finally:
            task_env.close()

    print("Done capturing snapshots.")


if __name__ == "__main__":
    main()
