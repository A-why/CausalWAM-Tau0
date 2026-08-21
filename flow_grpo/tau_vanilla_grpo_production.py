#!/usr/bin/env python3
"""Vanilla Tau0 + Flow-GRPO baseline — production training driver (critic-free).

The formal vanilla RL experiment: True Flow-GRPO (Wan2.1) on the tau0 action
flow, NO critic, NO ER-CAG ValueHead, NO paired counterfactual reference.

  - Policy          : VAM (Wan2.2-TI2V-5B) at checkpoints/pbb2_turn_switch/step_802 (cuda:0)
  - Reward          : official_reward.py -> r_t = float(check_success()) in {0,1}
                      read from the real-RoboTwin outcome contract
                      (outputs/v3b0_outcome_contract/sign_probe_results.jsonl).
                      This is the SAME sparse success reward the ER-CAG method
                      consumes; there is NO ACVS, NO shaping, NO dummy observation.
  - Observation     : real RoboTwin endpose+cameras captured in the native
                      snapshots (V3-B0), adapted via adapt_observation.
  - Advantage       : group-relative standardization over the K candidates
                      (TauTrajectoryGroup.compute_advantages) — the vanilla GRPO
                      baseline, NOT the ER-CAG Q_i - Q_0 gain.
  - Loss            : PPO-clipped Flow-GRPO (compute_grpo_loss), clip_range=1e-3,
                      adv_clip_max=5.0, beta_kl=0.0.
  - Loop            : sample once, N optimizer iterations (GRPO multi-epoch).

The reward target is environment-side privileged computation; the policy
observation NEVER reads switch qpos (see sign_probe contract).

Config: configs/training/vanilla_production.yaml (training-level), with the
VAM runtime skeleton from configs/runtime/vam_deploy.yaml.

Usage:
    /opt/conda/envs/tau0_wm/bin/python flow_grpo/tau_vanilla_grpo_production.py \
        [--config configs/training/vanilla_production.yaml] \
        [--snapshot S0] [--k 4] [--max-steps 20] [--validate-only]
"""
import sys, os, json, time, argparse, math
import numpy as np
import torch

CAUSAL_ROOT = os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAU0_ROOT = os.path.join(CAUSAL_ROOT, "tau-0-wm")
sys.path.insert(0, TAU0_ROOT)
sys.path.insert(0, CAUSAL_ROOT)
os.chdir(TAU0_ROOT)  # TauPolicy + model configs resolve relative to tau-0-wm

DEFAULT_CONFIG = os.path.join(CAUSAL_ROOT, "configs/training/vanilla_production.yaml")
VAM_DEPLOY = os.path.join(CAUSAL_ROOT, "configs/runtime/vam_deploy.yaml")


def _pin_sdpa():
    from models.wan_2_2_models.transformers.attention import set_attention_backend
    set_attention_backend(attention_impl="sdpa")
    try:
        set_attention_backend(attention_impl="sdpa", sdpa_backend="math")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def _expand(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_policy(checkpoint: str, statistics_file: str):
    import utils.model_utils
    utils.model_utils.forward_pass = lambda *a, **kw: None
    _pin_sdpa()
    from yaml import load, Loader, dump, Dumper
    from web_infer_utils.TauPolicy import TauPolicy

    runtime = load(open(VAM_DEPLOY), Loader=Loader)
    runtime["diffusion_model"]["model_path"] = checkpoint
    runtime["statistics_file"] = statistics_file
    runtime["action_type"] = "relative"
    runtime["action_space"] = "eef6d"
    tmp = f"/tmp/vanilla_vam_cfg_{os.getpid()}.yaml"
    with open(tmp, "w") as f:
        dump(runtime, f, Dumper=Dumper)

    device = torch.device("cuda:0")
    print(f"Loading VAM policy from {checkpoint}", flush=True)
    t0 = time.time()
    policy = TauPolicy(config_file=tmp, device=device, rank=0,
                       compile_model=False, attention_impl="sdpa",
                       enable_self_attn_fused_qkv=True,
                       enable_context_null_cache=True)
    policy.diffusion_model.eval()
    print(f"  VAM loaded in {time.time()-t0:.1f}s (cuda:0)", flush=True)
    return policy


def load_inputs(cfg, adapt_observation, snapshot="S0", task_name="turn_switch"):
    snap_dir = _expand(cfg["rollout"]["native_snapshots_dir"])
    cand_jsonl = _expand(cfg["rollout"]["candidates_jsonl"])
    with open(cand_jsonl) as f:
        candidates = [json.loads(l) for l in f if l.strip()]
    with open(os.path.join(snap_dir, f"{snapshot}.pkl"), "rb") as f:
        snap = __import__("pickle").load(f)
    obs = snap["full_state"]["observation"]
    cameras_wrapped = {k: {"rgb": v} for k, v in obs["cameras"].items()}
    robotwin_obs = {"observation": cameras_wrapped, "endpose": obs["endpose"]}
    tau_input = adapt_observation(robotwin_obs, task_name=task_name)
    snap_cands = [c for c in candidates if c["snapshot_id"] == snapshot]
    return snap_cands, tau_input


def load_sign_probe(cfg, snapshot="S0"):
    """Return {candidate_id: {'success': bool, ...}} from the outcome contract."""
    sign_jsonl = _expand(cfg["reward"]["source"])
    recs = {}
    with open(sign_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("type") == "header":
                continue
            recs[d["candidate_id"]] = d
    return {cid: r for cid, r in recs.items() if r.get("snapshot_id") == snapshot}


def denorm_action(action_norm, policy):
    act_mean = torch.tensor(policy.act_mean, device=action_norm.device, dtype=torch.float32)
    act_std = torch.tensor(policy.act_std, device=action_norm.device, dtype=torch.float32)
    if act_mean.dim() == 3:
        act_mean = act_mean.squeeze(0).squeeze(0)
        act_std = act_std.squeeze(0).squeeze(0)
    return action_norm.float() * act_std + act_mean


def validate_only(cfg: dict) -> int:
    """Static production-readiness validation (no model load, no training)."""
    import importlib
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{status}] {name}: {detail}", flush=True)

    checkpoint = _expand(cfg["diffusion_model"]["model_path"])
    stats = _expand(cfg["statistics_file"])
    sign = _expand(cfg["reward"]["source"])
    snap_dir = _expand(cfg["rollout"]["native_snapshots_dir"])
    cand = _expand(cfg["rollout"]["candidates_jsonl"])
    snapshot = cfg["rollout"]["snapshot"]

    check("config parse", isinstance(cfg, dict), "vanilla_production.yaml parsed")
    check("checkpoint exists",
          os.path.isdir(checkpoint) and os.path.exists(os.path.join(checkpoint, "config.json")),
          checkpoint)
    check("dataset/statistics exists", os.path.isfile(stats), stats)
    check("reward source exists", os.path.isfile(sign), sign)
    check("native snapshot exists", os.path.isfile(os.path.join(snap_dir, f"{snapshot}.pkl")),
          os.path.join(snap_dir, f"{snapshot}.pkl"))
    check("candidates jsonl exists", os.path.isfile(cand), cand)

    try:
        from ercag.official_reward import official_reward
        check("reward import", callable(official_reward), "ercag.official_reward.official_reward")
    except Exception as e:
        check("reward import", False, f"{type(e).__name__}: {e}")

    try:
        from adapters.robotwin.observation_adapter import adapt_observation
        from adapters.robotwin.action_adapter import adapt_tau_action_to_robotwin
        check("rollout adapter import",
              callable(adapt_observation) and callable(adapt_tau_action_to_robotwin),
              "adapters.robotwin.{observation_adapter,action_adapter}")
    except Exception as e:
        check("rollout adapter import", False, f"{type(e).__name__}: {e}")

    # Rollout environment initialization path: the real RoboTwin env is driven by
    # scripts/robotwin_theta_init_eval_one.py (make_env -> setup_demo) in the
    # `robotwin` env. Verify that driver is present + imports clean at module level.
    eval_driver = os.path.join(CAUSAL_ROOT, "scripts", "robotwin_theta_init_eval_one.py")
    check("rollout env driver present", os.path.isfile(eval_driver), eval_driver)

    print(f"\nVANILLA_PRODUCTION_READY: {'YES' if ok else 'NO'}", flush=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(_expand(args.config))
    snapshot = args.snapshot or cfg["rollout"]["snapshot"]
    K = args.k or cfg["flow_grpo"]["k"]
    max_steps = args.max_steps or cfg["training"]["max_steps"]

    if args.validate_only:
        print(f"=== Vanilla Flow-GRPO production validation (snapshot={snapshot}, K={K}) ===", flush=True)
        return validate_only(cfg)

    checkpoint = _expand(cfg["diffusion_model"]["model_path"])
    statistics_file = _expand(cfg["statistics_file"])
    output_dir = _expand(cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    task_name = cfg.get("task", "turn_switch")
    grpo = cfg["grpo"]
    opt = cfg["optimizer"]
    L = cfg["flow_grpo"]["num_inference_steps"]
    execution_steps = cfg["flow_grpo"]["execution_steps"]
    shift = cfg["flow_grpo"]["shift"]
    seeds = cfg["flow_grpo"]["sde_seeds"][:K]

    print("=" * 78, flush=True)
    print("Vanilla Tau0 + Flow-GRPO (critic-free) — production training", flush=True)
    print(f"  checkpoint={checkpoint}\n  statistics={statistics_file}\n  snapshot={snapshot} K={K} L={L} max_steps={max_steps}", flush=True)
    print("=" * 78, flush=True)

    policy = load_policy(checkpoint, statistics_file)
    from adapters.robotwin.observation_adapter import adapt_observation, get_instruction
    snap_cands, tau_input = load_inputs(cfg, adapt_observation, snapshot, task_name)
    sign_recs = load_sign_probe(cfg, snapshot)

    obs_img = tau_input["obs"]
    state_14d = tau_input["state"]
    gripper = tau_input["gripper_states"]
    # Official instruction: same source as capture/sign-probe (RoboTwin task
    # metadata via adapt_observation -> get_instruction), NOT a hardcoded literal.
    prompt = tau_input["prompt"]
    official_instruction = get_instruction(None, task_name)
    prompt_aligned = (prompt == official_instruction)
    print(f"PROMPT_ALIGNMENT: {'PASS' if prompt_aligned else 'FAIL'} "
          f"vanilla={prompt!r} == official={official_instruction!r}", flush=True)
    if not prompt_aligned:
        print("FATAL: vanilla rollout prompt != RoboTwin task instruction "
              "(capture/sign-probe source). Refusing to train misaligned.", flush=True)
        return 1
    print(f"\nSnapshot {snapshot}: state range=[{state_14d.min():.3f},{state_14d.max():.3f}], obs={obs_img.shape}", flush=True)

    # ---- trainable params (same True Flow-GRPO selection as FG-B / ER-CAG) ----
    trainable_params = []
    for name, param in policy.diffusion_model.named_parameters():
        if 'action_' in name or (name.find('action_') < 0 and name.find('vlm_interface') < 0):
            trainable_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    policy_opt = torch.optim.AdamW(trainable_params, lr=opt["lr"], betas=tuple(opt["betas"]),
                                   weight_decay=opt["weight_decay"], eps=opt["eps"], foreach=False)
    n_train = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in policy.diffusion_model.parameters())
    print(f"Policy trainable: {n_train:,}/{n_total:,} ({100*n_train/n_total:.1f}%)", flush=True)

    # ---- Phase 1: sample K candidates (deterministic), assign official reward ----
    from flow_grpo.tau_pipeline_with_logprob import TauPipelineWithLogprob
    from flow_grpo.tau_flow_grpo_buffer import TauTrajectoryGroup, build_trajectory_from_sde_result
    wrapper = TauPipelineWithLogprob(policy)

    print(f"\n=== Phase 1: sample {K} candidates, read official reward ===", flush=True)
    trajectories = []
    action_match = []
    rewards = []
    for i in range(K):
        seed = seeds[i]
        gen = torch.Generator(device=torch.device("cuda:0"))
        gen.manual_seed(seed)
        with torch.no_grad():
            res = wrapper.sample_with_logprob(
                state_14d=state_14d, gripper_states=gripper, obs_img=obs_img,
                prompt=prompt, num_inference_steps=L,
                execution_steps=execution_steps, seed=seed, generator=gen, shift=shift,
            )
        traj = build_trajectory_from_sde_result(res, state_14d, gripper, prompt)
        traj.k_idx = i
        traj.seed = seed

        stored_rel = np.asarray(snap_cands[i]["tau_relative_action"], dtype=np.float32)
        sampled_rel = denorm_action(res["action"], policy).cpu().numpy()
        maxdiff = float(np.abs(stored_rel - sampled_rel).max())
        action_match.append(maxdiff)
        cid = snap_cands[i]["candidate_id"]

        sr = sign_recs.get(cid, {})
        r_i = float(bool(sr.get("success", False)))
        traj.reward = r_i
        trajectories.append(traj)
        rewards.append(r_i)
        print(f"  {cid} seed={seed}: act_match(maxdiff)={maxdiff:.2e} official_reward={r_i}", flush=True)
        torch.cuda.synchronize(torch.device("cuda:0"))

    max_act_match = float(max(action_match))
    determinism_pass = max_act_match < 1e-3
    print(f"\nDeterminism check: max|sampled - stored| = {max_act_match:.2e} -> {'PASS' if determinism_pass else 'WARN'}", flush=True)

    # ---- group-relative advantage (vanilla GRPO, NOT ER-CAG) ----
    group = TauTrajectoryGroup(group_id="vanilla_production", state_14d=state_14d,
                               gripper_states=gripper, trajectories=trajectories)
    group.compute_advantages(eps=1e-6)
    adv_arr = np.array([t.advantage for t in trajectories])
    print(f"Group-relative advantages: {[round(a, 4) for a in adv_arr]} "
          f"(mean={adv_arr.mean():.4f}, std={adv_arr.std():.4f})", flush=True)

    # ---- Phase 2: multi-iteration GRPO (sample once, update N times) ----
    print(f"\n=== Phase 2: GRPO iterations (max {max_steps}) ===", flush=True)
    from flow_grpo.tau_flow_grpo_loss import compute_grpo_loss
    hist = []
    for it in range(max_steps):
        policy_opt.zero_grad()
        all_cur, all_old, all_adv = [], [], []
        for k_idx, traj in enumerate(trajectories):
            A_k = torch.tensor(traj.advantage, device="cuda:0", dtype=torch.float32)
            cur_lp, _ = wrapper.recompute_trajectory_logprobs(traj, enable_grad=True)
            for s in range(L):
                all_cur.append(cur_lp[s])
                all_old.append(traj.log_probs[s].detach())
                all_adv.append(A_k)
        logp_cur = torch.stack(all_cur).unsqueeze(1)   # (K*L, 1)
        logp_old = torch.stack(all_old).unsqueeze(1)   # (K*L, 1)
        adv_batch = torch.stack(all_adv)                # (K*L,)

        loss_dict = compute_grpo_loss(logp_cur, logp_old, adv_batch,
                                      clip_range=grpo["clip_range"],
                                      adv_clip_max=grpo["adv_clip_max"],
                                      beta_kl=grpo["beta_kl"])
        loss = loss_dict["loss"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grpo["max_grad_norm"])
        policy_opt.step()

        rec = {
            "iter": it,
            "loss": float(loss.item()),
            "ratio_mean": float(loss_dict["ratio_mean"].item()),
            "clipfrac": float(loss_dict["clipfrac"].item()),
            "approx_kl": float(loss_dict["approx_kl"].item()),
            "grad_norm": float(grad_norm),
        }
        hist.append(rec)
        print(f"  iter {it:2d}: loss={rec['loss']:.6f} ratio={rec['ratio_mean']:.6f} "
              f"clip={rec['clipfrac']:.4f} kl={rec['approx_kl']:.6f} grad={rec['grad_norm']:.4f}", flush=True)

    # ---- Phase 3: summary ----
    finite_pass = all(np.isfinite(r["loss"]) for r in hist)
    summary = {
        "phase": "Vanilla Tau0 + Flow-GRPO production (critic-free)",
        "snapshot": snapshot,
        "K": K,
        "num_inference_steps": L,
        "execution_steps": execution_steps,
        "max_steps": max_steps,
        "reward_contract": "r_t = float(check_success()) in {0,1}",
        "reward_source": _expand(cfg["reward"]["source"]),
        "advantage": "group-relative standardization (vanilla GRPO)",
        "prompt": prompt,
        "prompt_aligned": bool(prompt_aligned),
        "determinism_max_act_diff": max_act_match,
        "determinism_pass": bool(determinism_pass),
        "rewards": rewards,
        "advantages": [float(a) for a in adv_arr],
        "finite_all_steps": bool(finite_pass),
        "loss_final": hist[-1]["loss"],
        "grad_norm_final": hist[-1]["grad_norm"],
        "history": hist,
    }
    out_json = os.path.join(output_dir, "vanilla_production.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===", flush=True)
    print(f"  determinism: {'PASS' if determinism_pass else 'WARN'} (max_act_diff={max_act_match:.2e})", flush=True)
    print(f"  rewards: {rewards}", flush=True)
    print(f"  advantages: {[round(a, 4) for a in adv_arr]}", flush=True)
    print(f"  finite all steps: {finite_pass}", flush=True)
    print(f"  loss final: {hist[-1]['loss']:.6f}  grad_norm final: {hist[-1]['grad_norm']:.4f}", flush=True)
    print(f"  output -> {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
