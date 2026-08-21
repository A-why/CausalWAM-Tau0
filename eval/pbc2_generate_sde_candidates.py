#!/usr/bin/env python3
"""PB-C2 Phase B: generate K=16 Flow-GRPO SDE candidates per snapshot (Tau env).

Loads the PB-B2 canonical step802 checkpoint and, for each interaction snapshot
I0/I1/I2 (captured in Phase A), draws K=16 candidate action chunks through the
FORMAL Flow-GRPO SDE sampler (FG-A/B/C verified TauPipelineWithLogprob, L=5
flow steps, plain sigma-interpolation noise, shift=1.0) — NOT native UniPC.

Saves, per candidate: candidate_id, snapshot_id, SDE RNG seed, full SDE
trajectory hash, normalized relative action (33,20), physical absolute action
(33,20), finite check, physical validity.

optimizer.step = 0. No training. No ACVS. No logprob recomputation.

Usage:
    CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/tau0_wm/bin/python \
        ${CAUSALWAM_ROOT}/eval/pbc2_generate_sde_candidates.py
"""
import sys, os, json, pickle, hashlib, argparse, time
import numpy as np
import torch

CAUSAL_ROOT = os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAU0_ROOT = os.path.join(CAUSAL_ROOT, "tau-0-wm")
FLOW_GRPO_DIR = os.path.join(CAUSAL_ROOT, "flow_grpo")
sys.path.insert(0, TAU0_ROOT)
sys.path.insert(0, CAUSAL_ROOT)
sys.path.insert(0, FLOW_GRPO_DIR)
os.chdir(TAU0_ROOT)

CHECKPOINT = os.path.join(CAUSAL_ROOT, "checkpoints/pbb2_turn_switch/step_802")
STATS_FILE = os.path.join(CAUSAL_ROOT, "datasets/tau0_robotwin_success_v3_lerobot/turn_switch/statistics_relative_v2.json")
DEPLOY_CFG = os.path.join(CAUSAL_ROOT, "configs/runtime/vam_deploy.yaml")
OUT_ROOT = os.path.join(CAUSAL_ROOT, "outputs/pbc2_sde_productivity")
SNAP_DIR = os.path.join(OUT_ROOT, "sde_candidates", "snapshots")
CAND_DIR = os.path.join(OUT_ROOT, "sde_candidates")

K = 16
NUM_INFERENCE_STEPS = 5   # L=5 flow steps (FG-C)
EXECUTION_STEPS = 33
SHIFT = 1.0
BASE_SEED = 42
SDE_SEEDS = [BASE_SEED * 1000 + i for i in range(K)]  # 42000..42015

SNAPSHOTS = ["I0", "I1", "I2"]


def postprocess_relative_to_absolute(action_norm, state_14d, gripper, policy):
    """Replicate TauPolicy.play() relative->absolute post-processing exactly.

    action_norm: (33, 20) normalized relative action (on device)
    state_14d: (14,) robot state [l_xyz, l_quat_xyzw, r_xyz, r_quat_xyzw]
    gripper: (2,) tau gripper
    Returns: (33, 20) physical absolute eef6d action (CPU numpy)
    """
    from utils.action_space_utils import rela_eef_to_abs, quaternion_to_rotation_6d

    arm_dim = (policy.action_dim - 2 * policy.gripper_dim) // 2  # 9
    gripper_dim = policy.gripper_dim                              # 1

    state_t = torch.tensor(np.asarray(state_14d), dtype=torch.float32).unsqueeze(0)  # (1,14)
    grip_t = torch.tensor(np.asarray(gripper), dtype=torch.float32).unsqueeze(0)    # (1,2)
    state_rot_l_6d = quaternion_to_rotation_6d(state_t[:, 3:7])
    state_rot_r_6d = quaternion_to_rotation_6d(state_t[:, 10:14])
    state_6d = torch.cat((
        state_t[:, :3], state_rot_l_6d, grip_t[:, :1],
        state_t[:, 7:10], state_rot_r_6d, grip_t[:, 1:],
    ), dim=-1)  # (1, 20)

    # denormalize (act_std/act_mean are (1,1,20) on CPU; move action to CPU to match play())
    actions_norm = action_norm.detach().cpu().unsqueeze(0).float()  # (1,33,20)
    final = actions_norm * policy.act_std + policy.act_mean          # (1,33,20)

    action_ = torch.cat((
        final[:, :, :arm_dim],
        final[:, :, arm_dim + gripper_dim:2 * arm_dim + gripper_dim]
    ), dim=-1)[0]  # (33,18)

    state_6d = state_6d.unsqueeze(0)  # (1,1,20)
    state_ = torch.cat((
        state_6d[:, :, :arm_dim],
        state_6d[:, :, arm_dim + gripper_dim:2 * arm_dim + gripper_dim]
    ), dim=-1)[0]  # (1,18)

    abs_action = rela_eef_to_abs(action_, state_)  # (33,18)
    final[0, :, :arm_dim] = abs_action[:, :arm_dim]
    final[0, :, arm_dim + gripper_dim:2 * arm_dim + gripper_dim] = abs_action[:, arm_dim:]
    return final[0].numpy()  # (33,20)


def trajectory_hash(all_latents, seed):
    h = hashlib.sha256()
    h.update(str(seed).encode())
    for lat in all_latents:
        h.update(lat.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def check_physical_validity(abs_action):
    """finite + rotation-6d validity (det≈1, orthonormal)."""
    from utils.action_space_utils import rotation_6d_to_matrix
    if not np.isfinite(abs_action).all():
        return {"finite": False, "rotation_valid": False, "position_valid": False}
    left_6d = torch.tensor(abs_action[:, 3:9], dtype=torch.float32)
    right_6d = torch.tensor(abs_action[:, 13:19], dtype=torch.float32)
    rot_valid = True
    for r6d in (left_6d, right_6d):
        R = rotation_6d_to_matrix(r6d)
        det = torch.det(R)
        orth = torch.bmm(R, R.transpose(-1, -2)) - torch.eye(3)
        if (torch.abs(det - 1.0) > 1e-2).any() or (orth.abs() > 1e-2).any():
            rot_valid = False
    pos_valid = bool(np.isfinite(abs_action[:, [0, 1, 2, 10, 11, 12]]).all())
    return {"finite": True, "rotation_valid": rot_valid, "position_valid": pos_valid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", nargs="+", default=SNAPSHOTS)
    parser.add_argument("--k", type=int, default=K)
    args = parser.parse_args()

    os.makedirs(CAND_DIR, exist_ok=True)
    device = torch.device("cuda:0")

    import utils.model_utils
    utils.model_utils.forward_pass = lambda *a, **kw: None
    from models.wan_2_2_models.transformers.attention import set_attention_backend
    set_attention_backend(attention_impl='sdpa')
    try:
        set_attention_backend(attention_impl="sdpa", sdpa_backend="math")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    from yaml import Loader, load, dump, Dumper
    from web_infer_utils.TauPolicy import TauPolicy
    from flow_grpo.tau_pipeline_with_logprob import TauPipelineWithLogprob
    from adapters.robotwin.observation_adapter import adapt_observation
    from adapters.robotwin.action_adapter import adapt_tau_action_to_robotwin

    cfg = load(open(DEPLOY_CFG), Loader=Loader)
    cfg["diffusion_model"]["model_path"] = CHECKPOINT
    cfg["statistics_file"] = STATS_FILE
    cfg["action_type"] = "relative"
    cfg["action_space"] = "eef6d"
    tmp = f"/tmp/pbc2_cfg_{os.getpid()}.yaml"
    with open(tmp, "w") as f:
        dump(cfg, f, Dumper=Dumper)

    print(f"Loading checkpoint: {CHECKPOINT}")
    t0 = time.time()
    policy = TauPolicy(config_file=tmp, device=device, rank=0,
                       compile_model=False, attention_impl='sdpa',
                       enable_self_attn_fused_qkv=True,
                       enable_context_null_cache=True)
    wrapper = TauPipelineWithLogprob(policy)
    print(f"  loaded in {time.time()-t0:.1f}s")

    candidates = []
    sde_config = None

    for snap in args.snapshots:
        snap_path = os.path.join(SNAP_DIR, f"{snap}.pkl")
        with open(snap_path, "rb") as f:
            snap_data = pickle.load(f)
        obs = snap_data["obs"]
        qpos_at = snap_data["qpos"]
        print(f"\n=== Snapshot {snap} (ds={snap_data['ds_frame']}, qpos={qpos_at}) ===")

        # wrap obs for adapt_observation (same as vam_server)
        cameras_wrapped = {k: {"rgb": v} for k, v in obs["cameras"].items()}
        robotwin_obs = {"observation": cameras_wrapped, "endpose": obs["endpose"]}
        tau_input = adapt_observation(robotwin_obs, task_name="turn_switch")
        obs_img = tau_input["obs"]            # (V,3,H,W) float32 [-1,1]
        state_14d = tau_input["state"]        # (14,)
        gripper = tau_input["gripper_states"] # (2,)
        prompt = tau_input["prompt"]

        for i in range(args.k):
            seed = SDE_SEEDS[i]
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            t0 = time.time()
            res = wrapper.sample_with_logprob(
                state_14d=state_14d, gripper_states=gripper, obs_img=obs_img,
                prompt=prompt, num_inference_steps=NUM_INFERENCE_STEPS,
                execution_steps=EXECUTION_STEPS, seed=seed, generator=gen, shift=SHIFT,
            )
            action_norm = res["action"]  # (33,20) normalized
            abs_action = postprocess_relative_to_absolute(action_norm, state_14d, gripper, policy)
            rtw_abs_action = adapt_tau_action_to_robotwin(abs_action)  # (33,16) robotwin EE
            validity = check_physical_validity(abs_action)
            thash = trajectory_hash(res["all_latents"], seed)

            if sde_config is None:
                sde_config = {
                    "sde_type": "plain sigma-interpolation (Gaussian isotropic, Wan2.1)",
                    "L": NUM_INFERENCE_STEPS,
                    "shift": SHIFT,
                    "num_train_timesteps": int(policy.pipeline.num_train_timesteps),
                    "timesteps": res["timesteps"].detach().cpu().tolist(),
                    "sigmas": res["sigmas"].detach().cpu().tolist(),
                    "noise": "std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma",
                }

            cand = {
                "candidate_id": f"{snap}_{i:02d}",
                "snapshot_id": snap,
                "sde_rng_seed": seed,
                "trajectory_hash": thash,
                "normalized_relative_action": action_norm.detach().cpu().float().numpy().tolist(),
                "physical_absolute_action": abs_action.tolist(),
                "robotwin_absolute_action": rtw_abs_action.tolist(),
                "finite": bool(validity["finite"]),
                "rotation_valid": bool(validity["rotation_valid"]),
                "position_valid": bool(validity["position_valid"]),
                "gen_s": round(time.time() - t0, 3),
            }
            candidates.append(cand)
            print(f"  {cand['candidate_id']} seed={seed} finite={validity['finite']} "
                  f"rot={validity['rotation_valid']} pos={validity['position_valid']} "
                  f"hash={thash[:12]} ({cand['gen_s']}s)", flush=True)

        del tau_input, obs_img

    # Save candidates.jsonl
    cand_jsonl = os.path.join(CAND_DIR, "candidates.jsonl")
    with open(cand_jsonl, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    print(f"\nSaved {len(candidates)} candidates -> {cand_jsonl}")

    # manifest.json
    manifest = {
        "phase": "PB-C2 SDE candidate generation",
        "checkpoint": CHECKPOINT,
        "stats": STATS_FILE,
        "stats_sha256_prefix": "0eda42bdbd2d43d7",
        "action_type": "relative",
        "action_space": "eef6d",
        "sde_config": sde_config,
        "K": args.k,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "execution_steps": EXECUTION_STEPS,
        "shift": SHIFT,
        "sde_seeds": SDE_SEEDS[:args.k],
        "snapshots": {s: {"ds_frame": {"I0": 12, "I1": 44, "I2": 64}[s]} for s in args.snapshots},
        "n_candidates": len(candidates),
        "optimizer_step": 0,
        "training": False,
    }
    manifest_path = os.path.join(OUT_ROOT, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest -> {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
