
#!/usr/bin/env python3
"""VAM inference server: reads obs from stdin, writes action to stdout."""
import sys, os, struct, pickle, argparse, json, time
import numpy as np
import torch

TAU0_ROOT = os.path.join(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tau-0-wm")
sys.path.insert(0, TAU0_ROOT)
sys.path.insert(0, os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(TAU0_ROOT)

# Inject forward_pass sentinel
import utils.model_utils; utils.model_utils.forward_pass = lambda *a,**kw: None

from adapters.robotwin.observation_adapter import adapt_observation
from adapters.robotwin.action_adapter import adapt_tau_action_to_robotwin, compute_action_delta
from web_infer_utils.TauPolicy import TauPolicy
from models.wan_2_2_models.transformers.attention import set_attention_backend

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--action-type", default=None, choices=["absolute", "relative"],
                        help="Override action_type from config")
    args = parser.parse_args()

    set_attention_backend(attention_impl="sdpa")
    # PB-B2: pin SDPA to the conservative "math" backend (environment-compat fix,
    # same as tau-0-wm/main.py). The post-reboot driver (580.173.02 / CUDA 13.0)
    # intermittently faults PyTorch's flash/mem-efficient SDPA kernels on H100,
    # surfacing as "CUDA error: misaligned address" / "unspecified launch failure"
    # when the VAM inference subprocess runs concurrently with SAPIEN's Vulkan
    # renderer. This does NOT change the policy, stats, or action contract.
    try:
        set_attention_backend(attention_impl="sdpa", sdpa_backend="math")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    # Build runtime config
    import yaml
    from yaml import Loader, Dumper, load, dump
    cfg_path = os.path.join(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs/runtime/vam_deploy.yaml")
    cd = load(open(cfg_path), Loader=Loader)
    cd["diffusion_model"]["model_path"] = args.checkpoint
    cd["statistics_file"] = args.stats
    if args.action_type:
        cd["action_type"] = args.action_type

    # Write temp config
    tmp_cfg = f"/tmp/v0e0_vam_cfg_{os.getpid()}.yaml"
    with open(tmp_cfg, "w") as f:
        dump(cd, f, Dumper=Dumper)

    device = torch.device("cuda:0")
    model = TauPolicy(config_file=tmp_cfg, device=device, rank=0,
                      compile_model=False, attention_impl="sdpa",
                      enable_self_attn_fused_qkv=True,
                      enable_context_null_cache=True)

    print("VAM_READY", flush=True)
    sys.stdout.flush()

    while True:
        try:
            # Read observation
            len_bytes = sys.stdin.buffer.read(4)
            if len(len_bytes) < 4:
                break
            msg_len = struct.unpack(">I", len_bytes)[0]
            msg_bytes = sys.stdin.buffer.read(msg_len)
            obs = pickle.loads(msg_bytes)
        except Exception as e:
            break

        try:
            # Set random seed for reproducibility if provided
            seed = obs.get("seed", None)
            if seed is not None:
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(int(seed))

            # Adapt observation
            # Wrap raw camera arrays into {rgb: array} dicts expected by adapter
            cameras_wrapped = {k: {"rgb": v} for k, v in obs["cameras"].items()}
            robotwin_obs = {
                "observation": cameras_wrapped,
                "endpose": obs["endpose"],
            }
            tau_input = adapt_observation(robotwin_obs, task_name="turn_switch")

            # VAM inference
            with torch.inference_mode():
                action_tau = model.play(**tau_input)

            # Convert to RoboTwin
            rtw_action = adapt_tau_action_to_robotwin(action_tau)
            first = rtw_action[0]

            # Compute deltas for diagnostics
            deltas = compute_action_delta(rtw_action, robotwin_obs)

            response = {
                "robotwin_action": first.tolist(),
                "tau_action_shape": list(action_tau.shape),
                "left_dxyz_norm": deltas["left_dxyz_norm"],
                "right_dxyz_norm": deltas["right_dxyz_norm"],
                "left_gripper": float(first[7]),
                "right_gripper": float(first[15]),
                "error": None,
            }
        except Exception as e:
            response = {"error": str(e), "robotwin_action": None}

        resp_bytes = pickle.dumps(response)
        sys.stdout.buffer.write(struct.pack(">I", len(resp_bytes)))
        sys.stdout.buffer.write(resp_bytes)
        sys.stdout.buffer.flush()

if __name__ == "__main__":
    main()
