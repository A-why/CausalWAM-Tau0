#!/usr/bin/env python3
"""V0-E0.1: Launch official τ₀ WebSocket VAM server with project-side config.
Run in tau0_wm env."""
import sys, os, argparse

TAU0_ROOT = os.path.join(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tau-0-wm")
sys.path.insert(0, TAU0_ROOT)
os.chdir(TAU0_ROOT)

# Inject forward_pass sentinel
import utils.model_utils; utils.model_utils.forward_pass = lambda *a,**kw: None

from models.wan_2_2_models.transformers.attention import set_attention_backend
set_attention_backend(attention_impl="sdpa")

from web_infer_utils.server import TauPolicyServer, init_distributed_and_get_device
import socket


def build_runtime_config(checkpoint_path, stats_path, seed=42):
    """Build a runtime config YAML for the VAM server."""
    from yaml import safe_load, Dumper, dump, Loader

    base_cfg_path = os.path.join(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs/runtime/vam_deploy.yaml")
    cfg = dump(safe_load(open(base_cfg_path)), Dumper=Dumper)

    # Write temp config with injected paths
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    cfg = safe_load(open(base_cfg_path))
    cfg["diffusion_model"]["model_path"] = checkpoint_path
    cfg["statistics_file"] = stats_path
    cfg["seed"] = seed
    dump(cfg, tmp, Dumper=Dumper)
    tmp.close()
    return tmp.name


def main():
    parser = argparse.ArgumentParser(description="Launch τ₀ VAM WebSocket server")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config_path = build_runtime_config(args.checkpoint, args.stats, args.seed)
    print(f"[launcher] Config: {config_path}")
    print(f"[launcher] Checkpoint: {args.checkpoint}")
    print(f"[launcher] Stats: {args.stats}")
    print(f"[launcher] Host: {args.host}, Port: {args.port}")

    device, is_dist, rank, local_rank, world_size = init_distributed_and_get_device()
    print(f"[launcher] Device: {device}")

    actor = TauPolicyServer(
        args.host, args.port, {"server": "tau0_vam_v0e0"},
        config_file=config_path,
        device=device, rank=rank,
        compile_model=False,
        enable_self_attn_fused_qkv=True,
        enable_context_null_cache=True,
        attention_impl="sdpa",
        sdpa_backend="auto",
        flash_attn_version="auto",
        enable_action_rope_cache=True,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"[launcher] Server ready: {hostname} ({local_ip}):{args.port}")
    print("SERVER_READY", flush=True)

    actor.serve_forever()


if __name__ == "__main__":
    main()
