#!/usr/bin/env python3
"""MAINLINE-R1: Word-Aligned Native Paired Counterfactual Gain Audit.

Direct paired counterfactual instantiation of the ER-CAG core estimand, using the
tau-0 native action-conditioned WAM (no F_act / latent decomposition / state encoder):

    same H_t (obs + memory)
    + same environment randomness xi  (FIXED per-snapshot seed -> shared noise)
    + different action (candidate a_i  vs  Hold a_0)

    Q_i = R(Xhat_i) ;  Q_0 = R(Xhat_0) ;  G_i = Q_i - Q_0

Q evaluator = the ACVS reward head (tau-0-wm reference simulator checkpoint,
``checkpoints/tau0_wm/simulator/diffusion_pytorch_model.bin``), semantically a
Monte-Carlo-return value-like signal. Q = max(reward_trajectory) over the H=33 horizon
(canonical scalarization from the tau-0 reference, unchanged).

Shared-noise coupling (§8/§9/§10): the native pipeline ``infer()`` already accepts a
``seed`` for its generator (xi + reward noise both derive from that generator). The V3-B1
scoring did NOT pass it (random seed per call). Here we force ONE fixed seed per snapshot
so the hold and all K candidates share identical xi / reward noise; only the action
changes. The patch is a read-only monkey-patch on the pipeline object — no tau-0-wm
source file is modified.

No training. optimizer.step = 0.  H = 1 (single-step gain, §20/§21).

Usage:
    CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/tau0_wm/bin/python \
        ${CAUSALWAM_ROOT}/eval/native_ercag_gain_audit.py [--smoke-only]
"""
import sys, os, json, pickle, time, argparse
import numpy as np
import torch

CAUSAL_ROOT = os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAU0_ROOT = os.path.join(CAUSAL_ROOT, "tau-0-wm")
sys.path.insert(0, TAU0_ROOT)
sys.path.insert(0, CAUSAL_ROOT)
os.chdir(TAU0_ROOT)

ACVS_CKPT = os.path.join(CAUSAL_ROOT, "checkpoints/tau0_wm/simulator/diffusion_pytorch_model.bin")
ACVS_CFG = os.path.join(CAUSAL_ROOT, "configs/runtime/acvs_deploy.yaml")

V3B0_ROOT = os.path.join(CAUSAL_ROOT, "outputs/v3b0_outcome_contract")
V3B1_ROOT = os.path.join(CAUSAL_ROOT, "outputs/v3b1_acvs_positive_neutral")
SNAP_DIR = os.path.join(V3B0_ROOT, "native_snapshots")
CAND_JSONL = os.path.join(V3B1_ROOT, "tau_candidates", "candidates.jsonl")
HOLD_JSON = os.path.join(V3B1_ROOT, "tau_candidates", "hold_actions.json")
SIGN_JSONL = os.path.join(V3B0_ROOT, "sign_probe_results.jsonl")

OUT_ROOT = os.path.join(CAUSAL_ROOT, "outputs/native_ercag_gain_audit")

NUM_INFERENCE_STEPS = 10
EXECUTION_STEP = 33
N_MEM = 3
SNAPSHOTS = ["S0", "S1", "S2", "S3"]
BASE_SEED = 1000            # per-snapshot seed = BASE_SEED + snapshot index (one shared xi/snapshot)
NEUTRAL_EPS = 1e-6          # numerical near-zero threshold on G (per §23, not tuned for sign balance)


def _patch_seed(pipeline, seed):
    """Force a fixed generator seed on every infer() call (read-only monkey-patch).

    The native ``infer(..., seed=-1)`` does ``seed = seed if seed>=0 else random.randint(...)``
    and then ``seed_g.manual_seed(seed)`` before drawing both the video noise (xi) and the
    reward noise. Injecting a fixed seed makes the world-model noise + reward noise identical
    across calls, so candidate vs hold differ ONLY through the action conditioning.
    """
    if not hasattr(pipeline, "_orig_infer"):
        pipeline._orig_infer = pipeline.infer
    orig = pipeline._orig_infer

    def infer_seeded(*args, **kwargs):
        kwargs["seed"] = seed
        return orig(*args, **kwargs)

    pipeline.infer = infer_seeded
    return orig


def load_acvs():
    import utils.model_utils
    utils.model_utils.forward_pass = lambda *a, **kw: None
    from models.wan_2_2_models.transformers.attention import set_attention_backend
    set_attention_backend(attention_impl="sdpa")
    try:
        set_attention_backend(attention_impl="sdpa", sdpa_backend="math")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    from yaml import load, Loader, dump, Dumper
    from web_infer_utils.simulator.TauSimulator import TauSimulator
    from adapters.robotwin.observation_adapter import adapt_observation

    acvs_cfg = load(open(ACVS_CFG), Loader=Loader)
    acvs_cfg["diffusion_model"]["model_path"] = ACVS_CKPT
    tmp = f"/tmp/ercag_r1_acvs_cfg_{os.getpid()}.yaml"
    with open(tmp, "w") as f:
        dump(acvs_cfg, f, Dumper=Dumper)

    device = torch.device("cuda:1")
    print(f"Loading ACVS (TauSimulator) from {ACVS_CKPT}", flush=True)
    t0 = time.time()
    acvs = TauSimulator(config_file=tmp, device=device, rank=1)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    return acvs, adapt_observation


def load_inputs(adapt_observation):
    candidates = []
    with open(CAND_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    with open(HOLD_JSON) as f:
        holds = json.load(f)
    snap_obs = {}
    for snap in SNAPSHOTS:
        with open(os.path.join(SNAP_DIR, f"{snap}.pkl"), "rb") as f:
            snap_data = pickle.load(f)
        obs = snap_data["full_state"]["observation"]
        cameras_wrapped = {k: {"rgb": v} for k, v in obs["cameras"].items()}
        robotwin_obs = {"observation": cameras_wrapped, "endpose": obs["endpose"]}
        tau_input = adapt_observation(robotwin_obs, task_name="turn_switch")
        snap_obs[snap] = {"obs_img": tau_input["obs"], "prompt": tau_input["prompt"]}
    # load simulator ground-truth gains (G_true)
    g_true = {}
    with open(SIGN_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("type") == "header":
                continue
            if d.get("G_true") is not None:
                g_true[d["candidate_id"]] = {"G_true": d["G_true"], "Y": d["Y"], "Y0": d["Y0"],
                                             "sign": d["sign"]}
    return candidates, holds, snap_obs, g_true


def score(acvs, action_rel, obs_img, prompt, return_frame=False):
    """Score a (33,20) RELATIVE eef6d action -> reward trajectory (33,), optionally final frame."""
    acvs.reset()
    with torch.inference_mode():
        pred_final_frame, reward = acvs.play(
            obs=obs_img, prompt=prompt,
            actions=action_rel.astype(np.float32),
            num_inference_steps=NUM_INFERENCE_STEPS,
            execution_step=EXECUTION_STEP, n_mem=N_MEM,
        )
    out = {"reward": reward}
    if return_frame:
        out["final_frame"] = pred_final_frame
    return out


def downsample_frame(frame, size=4):
    """Mean-pool a [V,C,H,W] final frame to [V,C,size,size] and flatten (lightweight future fingerprint)."""
    V, C, H, W = frame.shape
    h, w = H // size, W // size
    f = frame.reshape(V, C, size, h, size, w).mean(axis=(3, 5))
    return f.reshape(-1)


def pairwise_l2(frames):
    """Pairwise L2 distance between flattened final frames (normalized per-frame)."""
    X = np.stack([f.reshape(-1) for f in frames], axis=0).astype(np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    G = X @ X.T
    n = len(X)
    iu = np.triu_indices(n, 1)
    cos = G[iu]
    return {"mean_cos": float(cos.mean()) if len(cos) else None,
            "min_cos": float(cos.min()) if len(cos) else None,
            "mean_l2": float(np.sqrt(2 - 2 * cos).mean()) if len(cos) else None}


def spearman(a, b):
    def _rank(x):
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(len(x), dtype=np.float64)
        ranks[order] = np.arange(len(x))
        return ranks
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if len(a) < 2:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()

    acvs, adapt_observation = load_acvs()
    candidates, holds, snap_obs, g_true = load_inputs(adapt_observation)

    # =================================================================
    # Phase 1 — self-consistency + native regression (§9/§13/§14)
    #   (a) same action + same seed -> identical reward  => deterministic shared noise
    #   (b) same action + different seed -> different reward => noise matters (xi varies)
    # =================================================================
    print("\n=== Phase 1: shared-noise self-consistency (S0 hold action) ===", flush=True)
    snap = "S0"
    hold_ab = np.asarray(holds[snap]["tau_relative_hold_action"], dtype=np.float32)
    obs_img, prompt = snap_obs[snap]["obs_img"], snap_obs[snap]["prompt"]

    _patch_seed(acvs.pipeline, BASE_SEED)
    r_a = score(acvs, hold_ab, obs_img, prompt)["reward"]
    r_b = score(acvs, hold_ab, obs_img, prompt)["reward"]

    _patch_seed(acvs.pipeline, BASE_SEED + 999)
    r_c = score(acvs, hold_ab, obs_img, prompt)["reward"]

    same_seed_maxdiff = float(np.max(np.abs(r_a - r_b)))
    diff_seed_maxdiff = float(np.max(np.abs(r_a - r_c)))
    consistency = {
        "same_seed_identical": bool(same_seed_maxdiff < 1e-9),
        "same_seed_max_abs_diff": same_seed_maxdiff,
        "diff_seed_max_abs_diff": diff_seed_maxdiff,
        "noise_matters": bool(diff_seed_maxdiff > 1e-6),
        "q_same_seed_a": float(np.max(r_a)),
        "q_same_seed_b": float(np.max(r_b)),
        "q_diff_seed_c": float(np.max(r_c)),
    }
    print(f"  same-seed max|r_a-r_b| = {same_seed_maxdiff:.3e} -> identical={consistency['same_seed_identical']}", flush=True)
    print(f"  diff-seed max|r_a-r_c| = {diff_seed_maxdiff:.3e} -> noise_matters={consistency['noise_matters']}", flush=True)
    print(f"  Q(hold,S0) same-seed = {consistency['q_same_seed_a']:.6f} / {consistency['q_same_seed_b']:.6f}, "
          f"diff-seed = {consistency['q_diff_seed_c']:.6f}", flush=True)

    if args.smoke_only:
        out = {"phase": "MAINLINE-R1 smoke (shared-noise self-consistency)", "consistency": consistency}
        os.makedirs(os.path.join(OUT_ROOT, "native_regression"), exist_ok=True)
        with open(os.path.join(OUT_ROOT, "native_regression", "shared_noise_consistency.json"), "w") as f:
            json.dump(out, f, indent=2)
        print("\nSMOKE DONE. self-consistency written.", flush=True)
        return 0

    # =================================================================
    # Phase 2 — full paired counterfactual (one shared xi per snapshot)
    # =================================================================
    print("\n=== Phase 2: paired counterfactual gain audit ===", flush=True)
    results = {}   # snapshot -> list of candidate records
    for si, snap in enumerate(SNAPSHOTS):
        seed = BASE_SEED + si
        _patch_seed(acvs.pipeline, seed)   # ONE shared xi for hold + all K candidates
        obs_img, prompt = snap_obs[snap]["obs_img"], snap_obs[snap]["prompt"]
        hold_ab = np.asarray(holds[snap]["tau_relative_hold_action"], dtype=np.float32)

        # reference generated ONCE (§11)
        t0 = time.time()
        ref = score(acvs, hold_ab, obs_img, prompt, return_frame=True)
        q0 = float(np.max(ref["reward"]))
        print(f"\n{snap} (seed={seed}): Q0={q0:.6f}  [{time.time()-t0:.1f}s]", flush=True)

        snap_cands = [c for c in candidates if c["snapshot_id"] == snap]
        recs = []
        for c in snap_cands:
            act_ab = np.asarray(c["tau_relative_action"], dtype=np.float32)
            t0 = time.time()
            out = score(acvs, act_ab, obs_img, prompt, return_frame=True)
            r = out["reward"]
            q = float(np.max(r))
            g = float(q - q0)
            recs.append({
                "candidate_id": c["candidate_id"],
                "snapshot_id": snap,
                "seed": seed,
                "Q0": q0,
                "Q_i": q,
                "G_i": g,
                "reward_min": float(r.min()), "reward_max": float(r.max()),
                "reward_mean": float(r.mean()), "reward_std": float(r.std()),
                "final_frame_fp": downsample_frame(out["final_frame"]).tolist(),
                "finite": bool(np.all(np.isfinite(r))),
                "s": round(time.time() - t0, 3),
            })
            print(f"  {c['candidate_id']}: Q={q:.6f} G={g:+.6f} ({time.time()-t0:.1f}s)", flush=True)
        results[snap] = recs

    # =================================================================
    # Phase 3 — gain statistics + future diversity + advantage + algebra
    # =================================================================
    print("\n=== Phase 3: gain statistics ===", flush=True)
    gain_stats = {}
    vanilla_vs_er = {}
    algebra = {}
    all_recs = [r for snap in SNAPSHOTS for r in results[snap]]

    for snap in SNAPSHOTS:
        recs = results[snap]
        Qs = np.array([r["Q_i"] for r in recs])
        Gs = np.array([r["G_i"] for r in recs])
        q0 = recs[0]["Q0"]
        eps = NEUTRAL_EPS
        pos = float((Gs > eps).sum()) / len(Gs)
        neg = float((Gs < -eps).sum()) / len(Gs)
        near = float((np.abs(Gs) <= eps).sum()) / len(Gs)
        frames = [np.array(r["final_frame_fp"]) for r in recs]
        gain_stats[snap] = {
            "mean_Q": float(Qs.mean()), "std_Q": float(Qs.std()),
            "Q0": float(q0),
            "mean_G": float(Gs.mean()), "std_G": float(Gs.std()),
            "RMS_G": float(np.sqrt(np.mean(Gs ** 2))),
            "min_G": float(Gs.min()), "max_G": float(Gs.max()),
            "positive_fraction": pos, "negative_fraction": neg, "near_zero_fraction": near,
            "n": int(len(Gs)),
            "future_pairwise": pairwise_l2(frames),
        }
        # vanilla vs ER advantage (top1 match / regret vs G_true)
        mean_q, std_q = Qs.mean(), Qs.std()
        A_vanilla = (Qs - mean_q) / (std_q + 1e-6)
        A_ER = Gs / (np.sqrt(np.mean(Gs ** 2)) + 1e-6)
        gts = [g_true[r["candidate_id"]]["G_true"] for r in recs]
        best_van = int(np.argmax(A_vanilla)); best_er = int(np.argmax(A_ER))
        best_true = int(np.argmax(gts))
        vanilla_vs_er[snap] = {
            "best_advantage_idx": best_van, "best_ER_idx": best_er, "best_true_idx": best_true,
            "vanilla_top1_match": bool(best_van == best_true),
            "ER_top1_match": bool(best_er == best_true),
            "A_vanilla": A_vanilla.tolist(), "A_ER": A_ER.tolist(),
            "G_true": gts,
        }
        # algebra check (§33): mean-centered G == mean-centered Q
        g_mc = Gs - Gs.mean()
        q_mc = Qs - Qs.mean()
        algebra[snap] = {"max_diff": float(np.max(np.abs(g_mc - q_mc)))}

    # pooled gain stats
    all_Q = np.array([r["Q_i"] for r in all_recs])
    all_G = np.array([r["G_i"] for r in all_recs])
    eps = NEUTRAL_EPS
    gain_stats["_pooled"] = {
        "mean_Q": float(all_Q.mean()), "std_Q": float(all_Q.std()),
        "mean_G": float(all_G.mean()), "std_G": float(all_G.std()),
        "RMS_G": float(np.sqrt(np.mean(all_G ** 2))),
        "min_G": float(all_G.min()), "max_G": float(all_G.max()),
        "positive_fraction": float((all_G > eps).sum()) / len(all_G),
        "negative_fraction": float((all_G < -eps).sum()) / len(all_G),
        "near_zero_fraction": float((np.abs(all_G) <= eps).sum()) / len(all_G),
        "n": int(len(all_G)),
    }

    # =================================================================
    # Phase 4 — predicted vs true gain (simulator sanity §27/§28)
    # =================================================================
    print("\n=== Phase 4: predicted vs true gain ===", flush=True)
    preds, trues, cids = [], [], []
    for r in all_recs:
        gt = g_true.get(r["candidate_id"])
        if gt is None:
            continue
        preds.append(r["G_i"]); trues.append(gt["G_true"]); cids.append(r["candidate_id"])
    preds = np.array(preds); trues = np.array(trues)
    sign_agree = float(np.mean(np.sign(preds) == np.sign(trues)))
    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    sim_sanity = {
        "n": int(len(preds)),
        "pearson_G_pred_vs_G_true": float(np.corrcoef(preds, trues)[0, 1]) if len(preds) > 1 else None,
        "spearman_G_pred_vs_G_true": spearman(preds, trues),
        "sign_agreement": sign_agree,
        "MAE": mae, "RMSE": rmse,
    }
    print(f"  n={len(preds)} pearson={sim_sanity['pearson_G_pred_vs_G_true']:.4f} "
          f"spearman={sim_sanity['spearman_G_pred_vs_G_true']:.4f} sign_agree={sign_agree:.3f} "
          f"MAE={mae:.4f} RMSE={rmse:.4f}", flush=True)

    # =================================================================
    # Phase 5 — classification (§26) + credit assignment (§30)
    # =================================================================
    print("\n=== Phase 5: classification + credit assignment ===", flush=True)
    # future diversity: are candidate futures distinct (world-model sensitivity)?
    mean_future_cos = np.mean([gain_stats[s]["future_pairwise"]["mean_cos"]
                               for s in SNAPSHOTS if gain_stats[s]["future_pairwise"]["mean_cos"] is not None])
    # Q/G dispersion
    std_G_pooled = gain_stats["_pooled"]["std_G"]
    std_Q_pooled = gain_stats["_pooled"]["std_Q"]

    future_differs = mean_future_cos is not None and mean_future_cos < 0.999
    qg_differs = std_G_pooled > 1e-3   # G dispersion beyond numerical precision
    if future_differs and qg_differs:
        diagnosis = "NATIVE_GAIN_HEALTHY"
    elif future_differs and not qg_differs:
        diagnosis = "EVALUATOR_RESOLUTION_BLOCKER"
    elif not future_differs:
        diagnosis = "NATIVE_WAM_ACTION_SENSITIVITY_BLOCKER"
    else:
        diagnosis = "IMPLEMENTATION_BLOCKED"
    print(f"  future_differs(mean_cos<0.999)={future_differs} (mean_cos={mean_future_cos:.6f})", flush=True)
    print(f"  std_G={std_G_pooled:.6f} std_Q={std_Q_pooled:.6f} qg_differs={qg_differs}", flush=True)
    print(f"  diagnosis={diagnosis}", flush=True)

    # credit assignment (§30): vanilla advantage < 0 but G_true > 0 (beneficial-negative-update)
    #                          vanilla advantage > 0 but G_true < 0 (harmful-positive-update)
    bn, hp, npos, nneg = 0, 0, 0, 0
    for r in all_recs:
        gt = g_true.get(r["candidate_id"])
        if gt is None:
            continue
        gt_sign = gt["G_true"]
        # A_vanilla sign = sign of (Q_i - mean_j Q_j) within snapshot
        snap = r["snapshot_id"]
        recs = results[snap]
        Qs = np.array([x["Q_i"] for x in recs])
        a_van = r["Q_i"] - Qs.mean()
        if gt_sign > 0:
            npos += 1
            if a_van < 0:
                bn += 1
        elif gt_sign < 0:
            nneg += 1
            if a_van > 0:
                hp += 1
    credit = {
        "beneficial_negative_rate": float(bn / npos) if npos else None,
        "harmful_positive_rate": float(hp / nneg) if nneg else None,
        "n_true_positive": npos, "n_true_negative": nneg,
        "measurable": bool(npos > 0 or nneg > 0),
    }

    # =================================================================
    # write outputs
    # =================================================================
    def wj(path, obj):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    wj(os.path.join(OUT_ROOT, "native_regression", "shared_noise_consistency.json"),
       {"phase": "MAINLINE-R1 shared-noise self-consistency", "consistency": consistency})

    paired_out = {
        "phase": "MAINLINE-R1 paired counterfactual generation",
        "coupling": "PASS" if (consistency["same_seed_identical"] and consistency["noise_matters"]) else "FAIL",
        "seed_scheme": f"BASE_SEED + snapshot_index ({BASE_SEED}..{BASE_SEED+3}); one shared xi per snapshot",
        "num_inference_steps": NUM_INFERENCE_STEPS, "execution_step": EXECUTION_STEP, "n_mem": N_MEM,
        "Q0": {s: float(results[s][0]["Q0"]) for s in SNAPSHOTS},
        "records": all_recs,
    }
    wj(os.path.join(OUT_ROOT, "paired_generation", "paired_counterfactual.json"), paired_out)

    wj(os.path.join(OUT_ROOT, "gain_statistics", "gain_statistics.json"),
       {"phase": "MAINLINE-R1 gain statistics", "neutral_eps": NEUTRAL_EPS,
        "per_snapshot": {k: v for k, v in gain_stats.items() if k != "_pooled"},
        "pooled": gain_stats["_pooled"],
        "future_diversity_mean_cos": mean_future_cos,
        "diagnosis": diagnosis,
        "diagnosis_definitions": {
            "NATIVE_GAIN_HEALTHY": "futures differ AND Q/G differ",
            "EVALUATOR_RESOLUTION_BLOCKER": "futures differ but Q/G ~ numerically flat",
            "NATIVE_WAM_ACTION_SENSITIVITY_BLOCKER": "futures ~ identical across actions",
        }})

    wj(os.path.join(OUT_ROOT, "simulator_sanity", "predicted_vs_true_gain.json"),
       {"phase": "MAINLINE-R1 predicted vs true gain", "sim_sanity": sim_sanity})

    wj(os.path.join(OUT_ROOT, "credit_assignment", "credit_assignment.json"),
       {"phase": "MAINLINE-R1 credit assignment", "credit": credit,
        "vanilla_vs_er_per_snapshot": vanilla_vs_er,
        "algebra_check": algebra,
        "algebra_note": "mean-centered G must equal mean-centered Q (G_i = Q_i - Q0 => G_i - mean(G) = Q_i - mean(Q)); max_diff should be ~0"})

    print("\n=== DONE ===", flush=True)
    print(f"diagnosis: {diagnosis}", flush=True)
    print(f"std_Q(pooled)={std_Q_pooled:.6f} std_G(pooled)={std_G_pooled:.6f} RMS_G={gain_stats['_pooled']['RMS_G']:.6f}", flush=True)
    print(f"positive={gain_stats['_pooled']['positive_fraction']:.3f} "
          f"negative={gain_stats['_pooled']['negative_fraction']:.3f} "
          f"near-zero={gain_stats['_pooled']['near_zero_fraction']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
