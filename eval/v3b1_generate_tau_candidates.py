#!/usr/bin/env python3
"""V3-B1 Phase A (Tau side): regenerate step802 SDE candidates in tau-RELATIVE format (33,20).

V3-B0 stored only robotwin_absolute_action (33,16). The recovered ACVS (TauSimulator /
model_sim.py reward head) consumes RELATIVE eef6d actions (statistics_sim.json action mean:
position ~0, rotation ~identity, gripper absolute [0,1]) — the SAME relative convention the
step802 policy natively emits (statistics_relative_v2.json). Relative eef6d deltas are
frame-invariant, so no world<->arm-base conversion is needed at the ACVS boundary.

This script regenerates the SAME K=8 candidates per snapshot (deterministic SDE sampler,
seeds 7000-7007) and stores:
  - tau_relative_action  (33,20)  denormalized relative eef6d  == ACVS input
  - tau_absolute_action_world (33,20)  world-frame absolute (reference, == V3-B0 semantics)
  - robotwin_absolute_action (33,16)   for hash cross-check

Determinism is verified by matching trajectory_hash against V3-B0. It also builds the
RELATIVE hold action per snapshot (Q0 reference): zero position delta + identity rotation
delta + current gripper.

optimizer.step = 0. No training. No ACVS (that is a separate cuda:1 script).

Usage:
    CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/tau0_wm/bin/python \
        ${CAUSALWAM_ROOT}/eval/v3b1_generate_tau_candidates.py
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
V3B0_ROOT = os.path.join(CAUSAL_ROOT, "outputs/v3b0_outcome_contract")
V3B1_ROOT = os.path.join(CAUSAL_ROOT, "outputs/v3b1_acvs_positive_neutral")
SNAP_DIR = os.path.join(V3B0_ROOT, "native_snapshots")
V3B0_CAND = os.path.join(V3B0_ROOT, "sde_candidates", "candidates.jsonl")
OUT_DIR = os.path.join(V3B1_ROOT, "tau_candidates")

K = 8
NUM_INFERENCE_STEPS = 5
EXECUTION_STEPS = 33
SHIFT = 1.0
SDE_SEEDS = [7000 + i for i in range(K)]
SNAPSHOTS = ["S0", "S1", "S2", "S3"]

IDENTITY_6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def postprocess_relative_to_absolute(action_norm, state_14d, gripper, policy):
    """Return (abs_action (33,20) world, rel_action (33,20) denormalized relative).

    rel_action is `final = action_norm * act_std + act_mean` BEFORE rela_eef_to_abs.
    This is the RELATIVE eef6d action in physical units, the ACVS input format.
    """
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
    final = actions_norm * policy.act_std + policy.act_mean       # (1,33,20) RELATIVE
    rel_action = final[0].clone().numpy()                          # (33,20) RELATIVE

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
    return final[0].numpy(), rel_action


def trajectory_hash(all_latents, seed):
    h = hashlib.sha256()
    h.update(str(seed).encode())
    for lat in all_latents:
        h.update(lat.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def build_relative_hold_action(obs_endpose):
    """Build (33,20) RELATIVE hold action: zero pos/rot delta + current gripper.

    Relative eef6d layout: [L_xyz(3) L_rot6d(6) L_grip(1) R_xyz(3) R_rot6d(6) R_grip(1)].
    Gripper is absolute tau-action [0,1] (0=open,1=close) = 1 - rtw_gripper.
    """
    rtw_l = float(obs_endpose["left_gripper"])
    rtw_r = float(obs_endpose["right_gripper"])
    grip_l = 1.0 - rtw_l   # tau action convention
    grip_r = 1.0 - rtw_r
    one = np.zeros(20, dtype=np.float32)
    one[0:3] = 0.0
    one[3:9] = IDENTITY_6D
    one[9] = grip_l
    one[10:13] = 0.0
    one[13:19] = IDENTITY_6D
    one[19] = grip_r
    return np.tile(one[None, :], (EXECUTION_STEPS, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", nargs="+", default=SNAPSHOTS)
    parser.add_argument("--k", type=int, default=K)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
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
    tmp = f"/tmp/v3b1_cfg_{os.getpid()}.yaml"
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

    v3b0_hash = {}
    with open(V3B0_CAND) as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                v3b0_hash[c["candidate_id"]] = c["trajectory_hash"]

    candidates = []
    holds = {}
    sde_config = None

    for snap in args.snapshots:
        with open(os.path.join(SNAP_DIR, f"{snap}.pkl"), "rb") as f:
            snap_data = pickle.load(f)
        obs = snap_data["full_state"]["observation"]
        print(f"\n=== Snapshot {snap} (step {snap_data['step_index']}, switch_qpos {snap_data['switch_qpos']:.6f}) ===", flush=True)

        cameras_wrapped = {k: {"rgb": v} for k, v in obs["cameras"].items()}
        robotwin_obs = {"observation": cameras_wrapped, "endpose": obs["endpose"]}
        tau_input = adapt_observation(robotwin_obs, task_name="turn_switch")
        obs_img = tau_input["obs"]
        state_14d = tau_input["state"]
        gripper = tau_input["gripper_states"]
        prompt = tau_input["prompt"]

        hold_rel = build_relative_hold_action(obs["endpose"])
        holds[snap] = {"tau_relative_hold_action": hold_rel.tolist()}

        for i in range(args.k):
            seed = SDE_SEEDS[i]
            cid = f"{snap}_{i:02d}"
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            t0 = time.time()
            res = wrapper.sample_with_logprob(
                state_14d=state_14d, gripper_states=gripper, obs_img=obs_img,
                prompt=prompt, num_inference_steps=NUM_INFERENCE_STEPS,
                execution_steps=EXECUTION_STEPS, seed=seed, generator=gen, shift=SHIFT,
            )
            action_norm = res["action"]
            abs_action, rel_action = postprocess_relative_to_absolute(action_norm, state_14d, gripper, policy)
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
                }

            v3b0_t = v3b0_hash.get(cid)
            hash_match = (v3b0_t is not None) and (v3b0_t == thash)
            finite = bool(np.isfinite(rel_action).all())

            cand = {
                "candidate_id": cid,
                "snapshot_id": snap,
                "sde_rng_seed": seed,
                "trajectory_hash": thash,
                "v3b0_hash_match": hash_match,
                "tau_relative_action": rel_action.tolist(),
                "tau_absolute_action_world": abs_action.tolist(),
                "robotwin_absolute_action": rtw_abs_action.tolist(),
                "finite": finite,
                "gen_s": round(time.time() - t0, 3),
            }
            candidates.append(cand)
            print(f"  {cid} seed={seed} hash_match={hash_match} finite={finite} "
                  f"rel_Lpos[0]={rel_action[0,0:3].round(4)} hash={thash[:12]} ({cand['gen_s']}s)", flush=True)

        del tau_input, obs_img

    cand_jsonl = os.path.join(OUT_DIR, "candidates.jsonl")
    with open(cand_jsonl, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    with open(os.path.join(OUT_DIR, "hold_actions.json"), "w") as f:
        json.dump(holds, f, indent=2)
    print(f"\nSaved {len(candidates)} candidates -> {cand_jsonl}", flush=True)

    n_match = sum(1 for c in candidates if c["v3b0_hash_match"])
    manifest = {
        "phase": "V3-B1 SDE candidate regeneration (tau-RELATIVE format)",
        "checkpoint": CHECKPOINT,
        "stats": STATS_FILE,
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
        "n_hash_match": n_match,
        "hash_match_all": n_match == len(candidates),
        "acvs_input": "tau_relative_action (33,20) denormalized relative eef6d; frame-invariant; normalized by statistics_sim.json inside TauSimulator.play()",
        "optimizer_step": 0,
        "training": False,
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest -> {OUT_DIR}/manifest.json (hash_match_all={n_match==len(candidates)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
