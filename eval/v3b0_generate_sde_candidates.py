#!/usr/bin/env python3
"""V3-B0 Phase C (Tau side): generate K=8 Flow-GRPO SDE candidates per native snapshot.

Loads the PB-B2 canonical step802 checkpoint and, for each simulator-native snapshot
S0-S3 (captured in V3-B0 Phase A), draws K=8 candidate action chunks through the FORMAL
Flow-GRPO SDE sampler (FG-A/B/C verified TauPipelineWithLogprob, L=5 flow steps, plain
sigma-interpolation noise, shift=1.0). Conditions on the SAVED observation captured at
the restored native state.

optimizer.step = 0. No training. No ACVS.

Usage:
    CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/tau0_wm/bin/python \
        ${CAUSALWAM_ROOT}/eval/v3b0_generate_sde_candidates.py
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
OUT_ROOT = os.path.join(CAUSAL_ROOT, "outputs/v3b0_outcome_contract")
SNAP_DIR = os.path.join(OUT_ROOT, "native_snapshots")
CAND_DIR = os.path.join(OUT_ROOT, "sde_candidates")

K = 8
NUM_INFERENCE_STEPS = 5
EXECUTION_STEPS = 33
SHIFT = 1.0
SDE_SEEDS = [7000 + i for i in range(K)]
SNAPSHOTS = ["S0", "S1", "S2", "S3"]


def postprocess_relative_to_absolute(action_norm, state_14d, gripper, policy):
    from utils.action_space_utils import rela_eef_to_abs, quaternion_to_rotation_6d
    arm_dim = (policy.action_dim - 2 * policy.gripper_dim) // 2
    gripper_dim = policy.gripper_dim

    state_t = torch.tensor(np.asarray(state_14d), dtype=torch.float32).unsqueeze(0)
    grip_t = torch.tensor(np.asarray(gripper), dtype=torch.float32).unsqueeze(0)
    state_rot_l_6d = quaternion_to_rotation_6d(state_t[:, 3:7])
    state_rot_r_6d = quaternion_to_rotation_6d(state_t[:, 10:14])
    state_6d = torch.cat((
        state_t[:, :3], state_rot_l_6d, grip_t[:, :1],
        state_t[:, 7:10], state_rot_r_6d, grip_t[:, 1:],
    ), dim=-1)

    actions_norm = action_norm.detach().cpu().unsqueeze(0).float()
    final = actions_norm * policy.act_std + policy.act_mean

    action_ = torch.cat((
        final[:, :, :arm_dim],
        final[:, :, arm_dim + gripper_dim:2 * arm_dim + gripper_dim]
    ), dim=-1)[0]

    state_6d = state_6d.unsqueeze(0)
    state_ = torch.cat((
        state_6d[:, :, :arm_dim],
        state_6d[:, :, arm_dim + gripper_dim:2 * arm_dim + gripper_dim]
    ), dim=-1)[0]

    abs_action = rela_eef_to_abs(action_, state_)
    final[0, :, :arm_dim] = abs_action[:, :arm_dim]
    final[0, :, arm_dim + gripper_dim:2 * arm_dim + gripper_dim] = abs_action[:, arm_dim:]
    return final[0].numpy()


def trajectory_hash(all_latents, seed):
    h = hashlib.sha256()
    h.update(str(seed).encode())
    for lat in all_latents:
        h.update(lat.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


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
    tmp = f"/tmp/v3b0_cfg_{os.getpid()}.yaml"
    with open(tmp, "w") as f:
        dump(cfg, f, Dumper=Dumper)

    print(f"Loading checkpoint: {CHECKPOINT}", flush=True)
    t0 = time.time()
    policy = TauPolicy(config_file=tmp, device=device, rank=0,
                       compile_model=False, attention_impl='sdpa',
                       enable_self_attn_fused_qkv=True,
                       enable_context_null_cache=True)
    wrapper = TauPipelineWithLogprob(policy)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    candidates = []
    sde_config = None

    for snap in args.snapshots:
        with open(os.path.join(SNAP_DIR, f"{snap}.pkl"), "rb") as f:
            snap_data = pickle.load(f)
        obs = snap_data["full_state"]["observation"]
        print(f"\n=== Snapshot {snap} (step {snap_data['step_index']}, switch_qpos {snap_data['switch_qpos']:.6f}) ===")

        cameras_wrapped = {k: {"rgb": v} for k, v in obs["cameras"].items()}
        robotwin_obs = {"observation": cameras_wrapped, "endpose": obs["endpose"]}
        tau_input = adapt_observation(robotwin_obs, task_name="turn_switch")
        obs_img = tau_input["obs"]
        state_14d = tau_input["state"]
        gripper = tau_input["gripper_states"]
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
            action_norm = res["action"]
            abs_action = postprocess_relative_to_absolute(action_norm, state_14d, gripper, policy)
            rtw_abs_action = adapt_tau_action_to_robotwin(abs_action)
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

            finite = bool(np.isfinite(abs_action).all())
            cand = {
                "candidate_id": f"{snap}_{i:02d}",
                "snapshot_id": snap,
                "sde_rng_seed": seed,
                "trajectory_hash": thash,
                "robotwin_absolute_action": rtw_abs_action.tolist(),
                "finite": finite,
                "gen_s": round(time.time() - t0, 3),
            }
            candidates.append(cand)
            print(f"  {cand['candidate_id']} seed={seed} finite={finite} hash={thash[:12]} ({cand['gen_s']}s)", flush=True)

        del tau_input, obs_img

    cand_jsonl = os.path.join(CAND_DIR, "candidates.jsonl")
    with open(cand_jsonl, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    print(f"\nSaved {len(candidates)} candidates -> {cand_jsonl}", flush=True)

    manifest = {
        "phase": "V3-B0 SDE candidate generation",
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
        "snapshots": args.snapshots,
        "n_candidates": len(candidates),
        "optimizer_step": 0,
        "training": False,
    }
    with open(os.path.join(CAND_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest -> {CAND_DIR}/manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
