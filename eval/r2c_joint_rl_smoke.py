#!/usr/bin/env python3
"""MAINLINE-R2C §20 — Official-Success Reward + Native Shared Value Joint RL Smoke.

One-process joint RL smoke (tau0_wm env, both GPUs):
  - VAM policy   (step802) on cuda:0  (True Flow-GRPO, critic-free, SDE sampler)
  - Simulator    (model_sim) on cuda:1 (frozen; native future hook -> Zhat)
  - Shared ValueHead          on cuda:1 (ercag/value_head.py, 100% shared params)

Reward contract (§1/§6): r_t = float(check_success()) in {0,1}. The reward TARGET
here is the real RoboTwin outcome already recorded in
outputs/v3b0_outcome_contract/sign_probe_results.jsonl — each candidate's `success`
bool == official check_success() == (switch.qpos >= limit[1]-0.05). This is
environment-side privileged computation; the policy observation / ValueHead input /
Tau WAM input NEVER read switch qpos (§3).

Joint loop (sample once, N update iterations — standard GRPO multi-epoch):
  For each candidate i (SDE seed 7000+i) + Hold reference:
    Zhat_i = native future hidden  [B, 864, 3072]  (simulator, frozen, no grad)
    Q_i    = V(Zhat_i)              [B, 3]           (shared ValueHead)
    R_i    = real success flag      {0,1}            (broadcast to 3 horizons)
  Value loss:   L_value = MSE(Q, R)  -> ValueHead only (AdamW lr=5e-5)
  ER advantage: G_i = Q_i - Q_0 ; A_i_ER = G_i / (sqrt(mean_j G_j^2) + eps)  [detach]
  Policy loss:  True Flow-GRPO PPO-clip on recomputed logprobs -> policy only

Gradient gates (§18, 4/4):
  L_value  -> ValueHead > 0 ; L_value  -> policy     = 0
  L_policy -> policy     > 0 ; L_policy -> ValueHead = 0

Smoke scale (§20): 1 -> 5 -> 20 update iterations. No long training.

Usage:
    /opt/conda/envs/tau0_wm/bin/python ${CAUSALWAM_ROOT}/eval/r2c_joint_rl_smoke.py \
        [--k 4] [--max-steps 20] [--snapshot S0]
"""
import sys, os, json, pickle, time, argparse, math
import numpy as np
import torch
import torch.nn.functional as F

CAUSAL_ROOT = os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAU0_ROOT = os.path.join(CAUSAL_ROOT, "tau-0-wm")
FLOW_GRPO_DIR = os.path.join(CAUSAL_ROOT, "flow_grpo")
sys.path.insert(0, TAU0_ROOT)
sys.path.insert(0, CAUSAL_ROOT)
sys.path.insert(0, FLOW_GRPO_DIR)
os.chdir(TAU0_ROOT)

# ---- model / data paths ----
CHECKPOINT = os.path.join(CAUSAL_ROOT, "checkpoints/pbb2_turn_switch/step_802")
STATS_FILE = os.path.join(CAUSAL_ROOT, "datasets/tau0_robotwin_success_v3_lerobot/turn_switch/statistics_relative_v2.json")
DEPLOY_CFG = os.path.join(CAUSAL_ROOT, "configs/runtime/vam_deploy.yaml")
ACVS_CKPT = os.path.join(CAUSAL_ROOT, "checkpoints/tau0_wm/simulator/diffusion_pytorch_model.bin")
ACVS_CFG = os.path.join(CAUSAL_ROOT, "configs/runtime/acvs_deploy.yaml")

V3B0_ROOT = os.path.join(CAUSAL_ROOT, "outputs/v3b0_outcome_contract")
V3B1_ROOT = os.path.join(CAUSAL_ROOT, "outputs/v3b1_acvs_positive_neutral")
SNAP_DIR = os.path.join(V3B0_ROOT, "native_snapshots")
CAND_JSONL = os.path.join(V3B1_ROOT, "tau_candidates", "candidates.jsonl")
HOLD_JSON = os.path.join(V3B1_ROOT, "tau_candidates", "hold_actions.json")
SIGN_JSONL = os.path.join(V3B0_ROOT, "sign_probe_results.jsonl")
OUT_ROOT = os.path.join(CAUSAL_ROOT, "outputs/r2c_joint_rl_smoke")

# ---- smoke hyperparameters ----
NUM_INFERENCE_STEPS = 5      # SDE flow steps (L)
EXECUTION_STEPS = 33         # action chunk length
ACVS_INFERENCE_STEPS = 10    # simulator denoising steps
SHIFT = 1.0
SDE_SEEDS = [7000 + i for i in range(8)]   # candidate seeds (V3-B1 scheme)
TASK = "turn_switch"
OBS_PROMPT = "turn on the switch"

# value optimizer (§17)
VALUE_LR = 5e-5
VALUE_BETAS = (0.9, 0.95)
VALUE_WD = 1e-5

# policy optimizer (§17 — current True Flow-GRPO / FG-B)
POLICY_LR = 1e-6
POLICY_BETAS = (0.9, 0.999)
POLICY_WD = 1e-4
POLICY_EPS = 1e-8

# GRPO (Wan2.1 / FG-B)
CLIP_RANGE = 1e-3
ADV_CLIP_MAX = 5.0
BETA_KL = 0.0
MAX_GRAD_NORM = 1.0
ADV_EPS = 1e-6


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


def load_policy():
    import utils.model_utils
    utils.model_utils.forward_pass = lambda *a, **kw: None
    _pin_sdpa()
    from yaml import load, Loader, dump, Dumper
    from web_infer_utils.TauPolicy import TauPolicy
    cfg = load(open(DEPLOY_CFG), Loader=Loader)
    cfg["diffusion_model"]["model_path"] = CHECKPOINT
    cfg["statistics_file"] = STATS_FILE
    cfg["action_type"] = "relative"
    cfg["action_space"] = "eef6d"
    tmp = f"/tmp/r2c_vam_cfg_{os.getpid()}.yaml"
    with open(tmp, "w") as f:
        dump(cfg, f, Dumper=Dumper)
    device = torch.device("cuda:0")
    print(f"Loading VAM policy from {CHECKPOINT}", flush=True)
    t0 = time.time()
    policy = TauPolicy(config_file=tmp, device=device, rank=0,
                       compile_model=False, attention_impl="sdpa",
                       enable_self_attn_fused_qkv=True,
                       enable_context_null_cache=True)
    policy.diffusion_model.eval()
    print(f"  VAM loaded in {time.time()-t0:.1f}s (cuda:0)", flush=True)
    return policy


def load_simulator():
    from yaml import load, Loader, dump, Dumper
    from web_infer_utils.simulator.TauSimulator import TauSimulator
    acvs_cfg = load(open(ACVS_CFG), Loader=Loader)
    acvs_cfg["diffusion_model"]["model_path"] = ACVS_CKPT
    tmp = f"/tmp/r2c_acvs_cfg_{os.getpid()}.yaml"
    with open(tmp, "w") as f:
        dump(acvs_cfg, f, Dumper=Dumper)
    device = torch.device("cuda:1")
    print(f"Loading simulator from {ACVS_CKPT}", flush=True)
    t0 = time.time()
    sim = TauSimulator(config_file=tmp, device=device, rank=1)
    print(f"  simulator loaded in {time.time()-t0:.1f}s (cuda:1)", flush=True)
    return sim


def load_inputs(adapt_observation, snapshot="S0"):
    candidates = [json.loads(l) for l in open(CAND_JSONL) if l.strip()]
    holds = json.load(open(HOLD_JSON))
    with open(os.path.join(SNAP_DIR, f"{snapshot}.pkl"), "rb") as f:
        snap = pickle.load(f)
    obs = snap["full_state"]["observation"]
    cameras_wrapped = {k: {"rgb": v} for k, v in obs["cameras"].items()}
    robotwin_obs = {"observation": cameras_wrapped, "endpose": obs["endpose"]}
    tau_input = adapt_observation(robotwin_obs, task_name=TASK)
    snap_cands = [c for c in candidates if c["snapshot_id"] == snapshot]
    hold_ab = np.asarray(holds[snapshot]["tau_relative_hold_action"], dtype=np.float32)
    return snap_cands, hold_ab, tau_input


def load_sign_probe(snapshot="S0"):
    """Return {candidate_id: {'success': bool, 'Y': float, 'Y0': float}}, success_threshold, hold_success."""
    success_threshold = None
    recs = {}
    hold_ref = {}
    with open(SIGN_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("type") == "header":
                success_threshold = d.get("success_threshold")
                hold_ref = d.get("hold_reference", {})
                continue
            recs[d["candidate_id"]] = d
    snap_recs = {cid: r for cid, r in recs.items() if r.get("snapshot_id") == snapshot}
    hold_success = None
    hr = hold_ref.get(snapshot)
    if hr is not None and success_threshold is not None:
        hold_success = bool(float(hr["Y0"]) >= success_threshold)
    return snap_recs, success_threshold, hold_success


def denorm_action(action_norm, policy):
    act_mean = torch.tensor(policy.act_mean, device=action_norm.device, dtype=torch.float32)
    act_std = torch.tensor(policy.act_std, device=action_norm.device, dtype=torch.float32)
    if act_mean.dim() == 3:
        act_mean = act_mean.squeeze(0).squeeze(0)
        act_std = act_std.squeeze(0).squeeze(0)
    return action_norm.float() * act_std + act_mean


def _apply_config_overrides(cfg: dict) -> None:
    """Override module globals from ercag_production.yaml (re-point only, no algorithm change)."""
    g = globals()
    _ev = lambda p: os.path.expandvars(os.path.expanduser(str(p)))

    dm = cfg.get("diffusion_model", {})
    if dm.get("model_path"):
        g["CHECKPOINT"] = _ev(dm["model_path"])
    if cfg.get("statistics_file"):
        g["STATS_FILE"] = _ev(cfg["statistics_file"])
    sim = cfg.get("simulator", {})
    if sim.get("model_path"):
        g["ACVS_CKPT"] = _ev(sim["model_path"])
    if cfg.get("task"):
        g["TASK"] = cfg["task"]
    if cfg.get("prompt"):
        g["OBS_PROMPT"] = cfg["prompt"]
    snap = cfg.get("snapshot", {})
    if snap.get("dir"):
        g["SNAP_DIR"] = _ev(snap["dir"])
    if snap.get("candidates_jsonl"):
        g["CAND_JSONL"] = _ev(snap["candidates_jsonl"])
    if snap.get("hold_actions_json"):
        g["HOLD_JSON"] = _ev(snap["hold_actions_json"])
    reward = cfg.get("reward", {})
    if reward.get("source"):
        g["SIGN_JSONL"] = _ev(reward["source"])
    if cfg.get("output_dir"):
        g["OUT_ROOT"] = _ev(cfg["output_dir"])
    fg = cfg.get("flow_grpo", {})
    for key, gkey in (("num_inference_steps", "NUM_INFERENCE_STEPS"),
                      ("execution_steps", "EXECUTION_STEPS"),
                      ("acvs_inference_steps", "ACVS_INFERENCE_STEPS"),
                      ("shift", "SHIFT")):
        if fg.get(key) is not None:
            g[gkey] = fg[key]
    vo = cfg.get("value_optimizer", {})
    if vo.get("lr") is not None:
        g["VALUE_LR"] = vo["lr"]
    if vo.get("betas"):
        g["VALUE_BETAS"] = tuple(vo["betas"])
    if vo.get("weight_decay") is not None:
        g["VALUE_WD"] = vo["weight_decay"]
    po = cfg.get("policy_optimizer", {})
    if po.get("lr") is not None:
        g["POLICY_LR"] = po["lr"]
    if po.get("betas"):
        g["POLICY_BETAS"] = tuple(po["betas"])
    if po.get("weight_decay") is not None:
        g["POLICY_WD"] = po["weight_decay"]
    if po.get("eps") is not None:
        g["POLICY_EPS"] = po["eps"]
    grpo = cfg.get("grpo", {})
    for key, gkey in (("clip_range", "CLIP_RANGE"),
                      ("adv_clip_max", "ADV_CLIP_MAX"),
                      ("beta_kl", "BETA_KL"),
                      ("max_grad_norm", "MAX_GRAD_NORM"),
                      ("adv_eps", "ADV_EPS")):
        if grpo.get(key) is not None:
            g[gkey] = grpo[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="ercag_production.yaml (overrides hardcoded smoke constants)")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--snapshot", default=None)
    args = ap.parse_args()

    cfg = None
    if args.config:
        import yaml as _yaml
        cfg = _yaml.safe_load(open(os.path.expandvars(args.config)))
        _apply_config_overrides(cfg)

    # Resolve runtime settings: explicit CLI > config > smoke fallback (K only).
    K = args.k if args.k is not None else (cfg.get("training", {}).get("k") if cfg else None)
    SNAP = args.snapshot if args.snapshot is not None else (cfg.get("rollout", {}).get("snapshot") if cfg else None)
    max_steps = args.max_steps if args.max_steps is not None else (cfg.get("training", {}).get("max_steps") if cfg else None)
    if K is None:
        K = 4
    if SNAP is None:
        SNAP = "S0"
    if max_steps is None:
        raise SystemExit("--max-steps required (or set training.max_steps in the config)")

    os.makedirs(OUT_ROOT, exist_ok=True)
    seeds = SDE_SEEDS[:K]

    # ---- load everything ----
    policy = load_policy()
    sim = load_simulator()
    from ercag.native_hook import enable_native_future_hook, get_native_future_hidden
    enable_native_future_hook(sim)
    from ercag.value_head import ValueHead

    from adapters.robotwin.observation_adapter import adapt_observation
    snap_cands, hold_ab, tau_input = load_inputs(adapt_observation, SNAP)
    sign_recs, success_threshold, hold_success = load_sign_probe(SNAP)

    obs_img = tau_input["obs"]
    state_14d = tau_input["state"]
    gripper = tau_input["gripper_states"]
    prompt = tau_input["prompt"]
    print(f"\nSnapshot {SNAP}: state range=[{state_14d.min():.3f},{state_14d.max():.3f}], "
          f"obs={obs_img.shape}, success_threshold={success_threshold}", flush=True)

    # ---- ValueHead (fresh init, shared candidate/reference) ----
    value_head = ValueHead(dim=3072).to(torch.device("cuda:1"))
    value_head.train()
    value_opt = torch.optim.AdamW(value_head.parameters(), lr=VALUE_LR,
                                  betas=VALUE_BETAS, weight_decay=VALUE_WD, foreach=False)
    print(f"\nValueHead: fresh init, dim=3072, params={sum(p.numel() for p in value_head.parameters())}", flush=True)

    # ---- policy trainable params (§17: current True Flow-GRPO selection) ----
    trainable_params = []
    for name, param in policy.diffusion_model.named_parameters():
        if 'action_' in name or (name.find('action_') < 0 and name.find('vlm_interface') < 0):
            trainable_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    policy_opt = torch.optim.AdamW(trainable_params, lr=POLICY_LR, betas=POLICY_BETAS,
                                   weight_decay=POLICY_WD, eps=POLICY_EPS, foreach=False)
    n_train = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in policy.diffusion_model.parameters())
    print(f"Policy trainable: {n_train:,}/{n_total:,} ({100*n_train/n_total:.1f}%)", flush=True)

    # ---- Phase 1: sample trajectories + run simulator -> Zhat (frozen, once) ----
    from flow_grpo.tau_pipeline_with_logprob import TauPipelineWithLogprob
    from flow_grpo.tau_flow_grpo_buffer import build_trajectory_from_sde_result
    wrapper = TauPipelineWithLogprob(policy)

    print(f"\n=== Phase 1: sample {K} candidates + hold, run simulator -> Zhat ===", flush=True)
    trajectories = []          # TauTrajectory per candidate
    zhat_candidates = []       # [B,864,3072] per candidate (cuda:1, detached)
    rewards = []               # real success flag per candidate
    action_match = []          # determinism check

    # candidate sampling + simulator
    for i in range(K):
        seed = seeds[i]
        gen = torch.Generator(device=torch.device("cuda:0"))
        gen.manual_seed(seed)
        with torch.no_grad():
            res = wrapper.sample_with_logprob(
                state_14d=state_14d, gripper_states=gripper, obs_img=obs_img,
                prompt=prompt, num_inference_steps=NUM_INFERENCE_STEPS,
                execution_steps=EXECUTION_STEPS, seed=seed, generator=gen, shift=SHIFT,
            )
        traj = build_trajectory_from_sde_result(res, state_14d, gripper, prompt)
        traj.k_idx = i
        traj.seed = seed
        trajectories.append(traj)

        # determinism check vs stored candidate action
        stored_rel = np.asarray(snap_cands[i]["tau_relative_action"], dtype=np.float32)
        sampled_rel = denorm_action(res["action"], policy).cpu().numpy()
        maxdiff = float(np.abs(stored_rel - sampled_rel).max())
        action_match.append(maxdiff)
        cid = snap_cands[i]["candidate_id"]

        # simulator on STORED action (validated in V3-B0/B1) -> Zhat_i
        sim.reset()
        with torch.inference_mode():
            sim.play(obs=obs_img, prompt=prompt, actions=stored_rel.astype(np.float32),
                     num_inference_steps=ACVS_INFERENCE_STEPS,
                     execution_step=EXECUTION_STEPS, n_mem=3)
        zhat = get_native_future_hidden(sim)   # [1, 864, 3072] on cuda:1, detached
        zhat_candidates.append(zhat)

        # reward target from real RoboTwin outcome
        sr = sign_recs.get(cid, {})
        r_i = float(bool(sr.get("success", False)))
        rewards.append(r_i)
        print(f"  {cid} seed={seed}: act_match(maxdiff)={maxdiff:.2e}, "
              f"Zhat={tuple(zhat.shape)}, success={bool(r_i)}", flush=True)
        # flush both devices' async work before next sample (reduces intermittent
        # driver fault surface under concurrent 2-GPU load)
        torch.cuda.synchronize(torch.device("cuda:0"))
        torch.cuda.synchronize(torch.device("cuda:1"))

    # hold reference -> Zhat_0 + reward_0
    sim.reset()
    with torch.inference_mode():
        sim.play(obs=obs_img, prompt=prompt, actions=hold_ab.astype(np.float32),
                 num_inference_steps=ACVS_INFERENCE_STEPS,
                 execution_step=EXECUTION_STEPS, n_mem=3)
    zhat_hold = get_native_future_hidden(sim)
    r_0 = float(bool(hold_success)) if hold_success is not None else 0.0
    print(f"  HOLD: Zhat_0={tuple(zhat_hold.shape)}, success={bool(r_0)}", flush=True)

    max_act_match = float(max(action_match))
    determinism_pass = max_act_match < 1e-3
    print(f"\nDeterminism check: max|sampled - stored| = {max_act_match:.2e} -> "
          f"{'PASS' if determinism_pass else 'WARN'}", flush=True)

    # ---- prepare value targets (broadcast scalar success to 3 horizons) ----
    R_cand = torch.tensor(rewards, dtype=torch.float32, device="cuda:1")      # [K]
    R_hold = torch.tensor([r_0], dtype=torch.float32, device="cuda:1")        # [1]

    # ---- Phase 2: joint RL iterations (1 -> 5 -> 20) ----
    print(f"\n=== Phase 2: joint RL iterations (max {max_steps}) ===", flush=True)
    hist = []
    for it in range(max_steps):
        # ---- clear both optimizers' grads before this iteration ----
        value_opt.zero_grad()
        policy_opt.zero_grad()

        # ---- value forward + loss (grad -> ValueHead only) ----
        q_cand = torch.cat([value_head(z) for z in zhat_candidates], dim=0)  # [K, 3]
        q_hold = value_head(zhat_hold)                                        # [1, 3]
        q_cand_scalar = q_cand.mean(dim=1)     # [K]
        q_hold_scalar = q_hold.mean(dim=1)     # [1]

        target_cand = R_cand.unsqueeze(1).expand(-1, 3)  # [K,3]
        target_hold = R_hold.unsqueeze(1).expand(-1, 3)  # [1,3]
        loss_value = F.mse_loss(q_cand, target_cand) + F.mse_loss(q_hold, target_hold)

        loss_value.backward()
        # gate 1 & 2 (value graph)
        vh_grad_norm = 0.0
        for p in value_head.parameters():
            if p.grad is not None:
                vh_grad_norm += float(p.grad.norm().item() ** 2)
        vh_grad_norm = math.sqrt(vh_grad_norm)
        policy_grad_from_value = sum(1 for p in trainable_params if p.grad is not None)
        gate1 = vh_grad_norm > 1e-9
        gate2 = (policy_grad_from_value == 0)
        value_opt.step()
        value_opt.zero_grad()   # clear VH grads so gate4 sees only policy-graph grads

        # ---- ER advantage (detached, §15) ----
        with torch.no_grad():
            q_cand_s = q_cand_scalar.detach()          # [K]
            q_hold_s = q_hold_scalar.detach()          # [1]
            G = q_cand_s - q_hold_s                    # [K]  = Q_i - Q_0
            rms_g = torch.sqrt(torch.mean(G ** 2))
            A_er = G / (rms_g + ADV_EPS)               # [K]  A_i_ER
        A_er = A_er.to(torch.device("cuda:0"))

        # ---- policy loss (grad -> policy only) ----
        all_cur, all_old, all_adv = [], [], []
        for k_idx, traj in enumerate(trajectories):
            cur_lp, _ = wrapper.recompute_trajectory_logprobs(traj, enable_grad=True)
            for s in range(NUM_INFERENCE_STEPS):
                all_cur.append(cur_lp[s])
                all_old.append(traj.log_probs[s].detach())
                all_adv.append(A_er[k_idx])
        logp_cur = torch.stack(all_cur).unsqueeze(1)    # [K*L, 1]
        logp_old = torch.stack(all_old).unsqueeze(1)    # [K*L, 1]
        adv_batch = torch.stack(all_adv)                 # [K*L]

        from flow_grpo.tau_flow_grpo_loss import compute_grpo_loss
        loss_dict = compute_grpo_loss(logp_cur, logp_old, adv_batch,
                                      clip_range=CLIP_RANGE, adv_clip_max=ADV_CLIP_MAX,
                                      beta_kl=BETA_KL)
        loss_policy = loss_dict["loss"]
        loss_policy.backward()

        # gate 3 & 4 (policy graph)
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
        vh_grad_from_policy = sum(1 for p in value_head.parameters() if p.grad is not None)
        gate3 = float(policy_grad_norm) > 1e-9
        gate4 = (vh_grad_from_policy == 0)
        policy_opt.step()
        policy_opt.zero_grad()  # clear policy grads so next gate2 sees only value-graph grads

        # ---- monitor ----
        with torch.no_grad():
            q_cand_np = q_cand_scalar.cpu().numpy()
            q_hold_np = float(q_hold_scalar.item())
            G_np = G.cpu().numpy()
            g_std = float(G_np.std())
            sign_pos = float((G_np > 1e-6).mean())
            sign_neg = float((G_np < -1e-6).mean())
            nan_any = bool(not np.isfinite(G_np).all())
        rec = {
            "iter": it,
            "loss_value": float(loss_value.item()),
            "loss_policy": float(loss_policy.item()),
            "ratio_mean": float(loss_dict["ratio_mean"].item()),
            "clipfrac": float(loss_dict["clipfrac"].item()),
            "q_cand_mean": float(q_cand_np.mean()),
            "q_cand_std": float(q_cand_np.std()),
            "q_hold": q_hold_np,
            "g_std": g_std,
            "g_sign_pos": sign_pos,
            "g_sign_neg": sign_neg,
            "vh_grad_norm": vh_grad_norm,
            "policy_grad_norm": float(policy_grad_norm),
            "gate1_Lvalue_VH": bool(gate1),
            "gate2_Lvalue_policy": bool(gate2),
            "gate3_Lpolicy_policy": bool(gate3),
            "gate4_Lpolicy_VH": bool(gate4),
            "nan": nan_any,
        }
        hist.append(rec)
        gates_ok = gate1 and gate2 and gate3 and gate4
        gate_str = "".join("1" if g else "0" for g in (gate1, gate2, gate3, gate4))
        print(f"  iter {it:2d}: Lv={rec['loss_value']:.4f} Lp={rec['loss_policy']:.4f} "
              f"ratio={rec['ratio_mean']:.4f} clip={rec['clipfrac']:.3f} "
              f"Qmean={rec['q_cand_mean']:+.4f} Qstd={rec['q_cand_std']:.4f} "
              f"Q0={rec['q_hold']:+.4f} Gstd={rec['g_std']:.4f} "
              f"gates={gate_str} nan={nan_any}", flush=True)

    # ---- Phase 3: summary + write ----
    value_loss_final = hist[-1]["loss_value"]
    q_std_final = hist[-1]["q_cand_std"]
    g_std_final = hist[-1]["g_std"]
    gates_all_ok = all(r["gate1_Lvalue_VH"] and r["gate2_Lvalue_policy"]
                       and r["gate3_Lpolicy_policy"] and r["gate4_Lpolicy_VH"] for r in hist)
    nan_any = any(r["nan"] for r in hist)
    finite_pass = (not nan_any) and all(
        np.isfinite(r["loss_value"]) and np.isfinite(r["loss_policy"]) for r in hist)

    summary = {
        "phase": "MAINLINE-R2C joint RL smoke",
        "snapshot": SNAP,
        "K": K,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "execution_steps": EXECUTION_STEPS,
        "max_steps": max_steps,
        "reward_contract": "r_t = float(check_success()) in {0,1}",
        "reward_source": "outputs/v3b0_outcome_contract/sign_probe_results.jsonl (real RoboTwin)",
        "determinism_max_act_diff": max_act_match,
        "determinism_pass": bool(determinism_pass),
        "native_hook_shape": list(zhat_hold.shape),
        "value_head": "ercag/value_head.py fresh init, shared candidate/reference",
        "gradient_gates_4of4": bool(gates_all_ok),
        "finite_all_steps": bool(finite_pass),
        "value_loss_final": value_loss_final,
        "q_cand_std_final": q_std_final,
        "g_std_final": g_std_final,
        "history": hist,
    }
    with open(os.path.join(OUT_ROOT, "joint_rl_smoke.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===", flush=True)
    print(f"  determinism: {'PASS' if determinism_pass else 'WARN'} (max_act_diff={max_act_match:.2e})", flush=True)
    print(f"  native hook: Zhat {list(zhat_hold.shape)}", flush=True)
    print(f"  gradient gates 4/4: {gates_all_ok}", flush=True)
    print(f"  finite all steps:   {finite_pass}", flush=True)
    print(f"  value_loss final:   {value_loss_final:.4f}", flush=True)
    print(f"  Q_cand std final:   {q_std_final:.4f}", flush=True)
    print(f"  G std final:        {g_std_final:.4f}", flush=True)
    print(f"  output -> {OUT_ROOT}/joint_rl_smoke.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
