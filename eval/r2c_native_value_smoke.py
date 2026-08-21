#!/usr/bin/env python3
"""R2C smoke — native future hook + shared ValueHead on native Tau future.

Verifies the core ER-CAG mechanism WITHOUT training:
    1. native future hook (store_buffer monkey-patch) yields Zhat [B, seq_len, 3072]
    2. shared ValueHead (ercag/value_head.py) reads Zhat -> Q_i / Q_0
    3. G_i = Q_i - Q_0 is computable and (hopefully) non-degenerate

No policy update. No ValueHead training. Reward head OFF (reward output ignored).

Usage:
    CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/tau0_wm/bin/python \
        ${CAUSALWAM_ROOT}/eval/r2c_native_value_smoke.py --n-candidates 3
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
OUT_ROOT = os.path.join(CAUSAL_ROOT, "outputs/r2c_native_value_smoke")

NUM_INFERENCE_STEPS = 10
EXECUTION_STEP = 33
N_MEM = 3
BASE_SEED = 2000


def _patch_seed(pipeline, seed):
    """Force a fixed generator seed on every infer() call (read-only monkey-patch).

    Same as R1: shared noise xi across hold + candidates so G_i = Q_i - Q_0 is a
    pure action effect (only the action conditioning differs).
    """
    if not hasattr(pipeline, "_orig_infer"):
        pipeline._orig_infer = pipeline.infer
    orig = pipeline._orig_infer

    def infer_seeded(*args, **kwargs):
        kwargs["seed"] = seed
        return orig(*args, **kwargs)

    pipeline.infer = infer_seeded
    return orig


def load_sim():
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
    tmp = f"/tmp/r2c_acvs_cfg_{os.getpid()}.yaml"
    with open(tmp, "w") as f:
        dump(acvs_cfg, f, Dumper=Dumper)
    device = torch.device("cuda:1")
    print(f"Loading simulator from {ACVS_CKPT}", flush=True)
    t0 = time.time()
    sim = TauSimulator(config_file=tmp, device=device, rank=1)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    return sim, adapt_observation


def load_inputs(adapt_observation):
    candidates = [json.loads(l) for l in open(CAND_JSONL) if l.strip()]
    holds = json.load(open(HOLD_JSON))
    obs_img, prompt = None, None
    with open(os.path.join(SNAP_DIR, "S0.pkl"), "rb") as f:
        snap = pickle.load(f)
    obs = snap["full_state"]["observation"]
    cameras_wrapped = {k: {"rgb": v} for k, v in obs["cameras"].items()}
    robotwin_obs = {"observation": cameras_wrapped, "endpose": obs["endpose"]}
    tau_input = adapt_observation(robotwin_obs, task_name="turn_switch")
    obs_img, prompt = tau_input["obs"], tau_input["prompt"]
    return candidates, holds, obs_img, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-candidates", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)

    sim, adapt_observation = load_sim()
    from ercag.native_hook import enable_native_future_hook, get_native_future_hidden
    enable_native_future_hook(sim)
    _patch_seed(sim.pipeline, BASE_SEED)   # shared xi for hold + all candidates

    candidates, holds, obs_img, prompt = load_inputs(adapt_observation)
    hold_ab = np.asarray(holds["S0"]["tau_relative_hold_action"], dtype=np.float32)
    snap_cands = [c for c in candidates if c["snapshot_id"] == "S0"][: args.n_candidates]

    # ---- reference (Hold) ----
    sim.reset()
    with torch.inference_mode():
        sim.play(obs=obs_img, prompt=prompt, actions=hold_ab.astype(np.float32),
                 num_inference_steps=NUM_INFERENCE_STEPS, execution_step=EXECUTION_STEP, n_mem=N_MEM)
    z0 = get_native_future_hidden(sim)  # [1, seq_len, 3072]
    print(f"\n[hook] Zhat_0 (hold)  shape={tuple(z0.shape)}  dtype={z0.dtype}", flush=True)

    B, L, D = z0.shape
    print(f"[hook] seq_len={L}  dim={D}  L%6={L % 6}  L//6={L // 6}", flush=True)

    # ---- candidates ----
    zis = []
    for c in snap_cands:
        act_ab = np.asarray(c["tau_relative_action"], dtype=np.float32)
        sim.reset()
        with torch.inference_mode():
            sim.play(obs=obs_img, prompt=prompt, actions=act_ab.astype(np.float32),
                     num_inference_steps=NUM_INFERENCE_STEPS, execution_step=EXECUTION_STEP, n_mem=N_MEM)
        zi = get_native_future_hidden(sim)
        print(f"[hook] Zhat_i {c['candidate_id']} shape={tuple(zi.shape)}", flush=True)
        zis.append(zi)

    # ---- shared ValueHead (fresh init) ----
    from ercag.value_head import ValueHead
    vh = ValueHead(dim=D).to(z0.device)
    vh.eval()

    result = {"seq_len": L, "dim": D, "L_divisible_by_6": bool(L % 6 == 0)}

    if L % 6 == 0:
        with torch.no_grad():
            q0 = vh(z0)                     # [1, 3]
            qi = torch.stack([vh(z) for z in zis], dim=0)  # [K, 3]
        q0_flat = q0[:, -1]                 # use final future-frame horizon (full-horizon proxy)
        qi_flat = qi[:, -1]
        G = qi_flat - q0_flat
        result.update({
            "q0": q0.tolist(), "qi": qi.tolist(),
            "G": G.tolist(),
            "std_Q": float(qi_flat.std()), "std_G": float(G.std()),
            "non_degenerate": bool(float(G.std()) > 1e-6),
        })
        print(f"\n[value] Q_0 = {q0.tolist()}", flush=True)
        print(f"[value] Q_i = {qi.tolist()}", flush=True)
        print(f"[value] G_i = {G.tolist()}", flush=True)
        print(f"[value] std_Q={float(qi_flat.std()):.6f}  std_G={float(G.std()):.6f}  "
              f"non_degenerate={result['non_degenerate']}", flush=True)
    else:
        print(f"\n[value] seq_len {L} NOT divisible by 6 -> ValueHead reshape blocked; "
              f"reporting raw seq_len only.", flush=True)

    with open(os.path.join(OUT_ROOT, "smoke.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSMOKE DONE -> {os.path.join(OUT_ROOT, 'smoke.json')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
