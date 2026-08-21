#!/usr/bin/env python3
"""
τ₀-WM Checkpoint Validator for V0-B

Validates that all required checkpoint files exist and are structurally valid.
Does NOT load model weights into memory.
Does NOT require GPU.

Exit codes:
  0 = All V0-B prerequisites satisfied
  1 = Required files missing
  2 = Config error
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_yaml_config(config_path: str) -> dict:
    """Load checkpoint configuration from YAML file."""
    from yaml import safe_load
    with open(config_path) as f:
        return safe_load(f)


def check_file(path: str, label: str, mandatory: bool = True) -> Tuple[str, bool]:
    """Check if a file exists. Returns (status_tag, exists)."""
    exists = os.path.isfile(path)
    if exists:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        return (f"[OK]      {label}: {path} ({size_mb:.1f} MB)", True)
    else:
        tag = "[MISSING] " if mandatory else "[WARNING] "
        return (f"{tag}{label}: {path} (not found)", False)


def check_dir(path: str, label: str, mandatory: bool = True) -> Tuple[str, bool]:
    """Check if a directory exists. Returns (status_tag, exists)."""
    exists = os.path.isdir(path)
    if exists:
        return (f"[OK]      {label}: {path}", True)
    else:
        tag = "[MISSING] " if mandatory else "[WARNING] "
        return (f"{tag}{label}: {path} (not found)", False)


def check_json_parseable(path: str, label: str) -> Tuple[str, bool]:
    """Check if a JSON file can be parsed."""
    if not os.path.isfile(path):
        return (f"[MISSING] {label}: {path}", False)
    try:
        with open(path) as f:
            json.load(f)
        return (f"[OK]      {label}: valid JSON", True)
    except Exception as e:
        return (f"[INVALID] {label}: {e}", False)


def check_safetensors_index(index_path: str, label: str) -> Tuple[str, bool]:
    """Check safetensors index file and verify all shards exist."""
    if not os.path.isfile(index_path):
        return (f"[MISSING] {label}: {index_path}", False)

    try:
        with open(index_path) as f:
            index_data = json.load(f)
    except Exception as e:
        return (f"[INVALID] {label}: cannot parse JSON - {e}", False)

    # Get weight map
    weight_map = index_data.get("weight_map", index_data)
    if not isinstance(weight_map, dict):
        return (f"[INVALID] {label}: no weight_map found", False)

    # Get unique shard files
    checkpoint_dir = os.path.dirname(index_path)
    shard_files = sorted(set(weight_map.values()))
    missing_shards = []
    for shard in shard_files:
        shard_path = os.path.join(checkpoint_dir, shard)
        if not os.path.isfile(shard_path):
            missing_shards.append(shard)

    total_params = len(weight_map)
    if missing_shards:
        return (
            f"[INVALID] {label}: {len(missing_shards)}/{len(shard_files)} shards missing: "
            f"{missing_shards[:3]}...",
            False,
        )

    # Count by file
    shard_sizes = {}
    for param, shard in weight_map.items():
        shard_sizes[shard] = shard_sizes.get(shard, 0) + 1

    return (
        f"[OK]      {label}: {total_params} params, "
        f"{len(shard_files)} shard files, all present",
        True,
    )


def check_tokenizer_dir(tokenizer_dir: str, label: str) -> Tuple[str, bool]:
    """Check if tokenizer directory has required files."""
    if not os.path.isdir(tokenizer_dir):
        return (f"[MISSING] {label}: {tokenizer_dir}", False)

    required_patterns = [
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    # sentencepiece model can have various names
    sp_files = list(Path(tokenizer_dir).glob("*.model")) + list(Path(tokenizer_dir).glob("spiece*"))

    missing = []
    for pat in required_patterns:
        if not os.path.isfile(os.path.join(tokenizer_dir, pat)):
            missing.append(pat)
    if not sp_files:
        missing.append("sentencepiece model (*.model or spiece*)")

    if missing:
        return (f"[INVALID] {label}: missing {missing}", False)

    files_found = required_patterns + [os.path.basename(str(sp_files[0]))]
    return (f"[OK]      {label}: {', '.join(files_found)}", True)


def check_yaml_parseable(path: str, label: str) -> Tuple[str, bool]:
    """Check if a YAML file can be parsed."""
    if not os.path.isfile(path):
        return (f"[MISSING] {label}: {path}", False)
    try:
        from yaml import safe_load
        with open(path) as f:
            safe_load(f)
        return (f"[OK]      {label}: valid YAML", True)
    except Exception as e:
        return (f"[INVALID] {label}: {e}", False)


def validate_vam(cfg: dict) -> List[Tuple[str, bool]]:
    """Validate VAM checkpoint."""
    results = []
    vam_dir = cfg["tau0"]["vam"]["model_dir"]

    # Check for safetensors index (preferred) or bin index
    sf_index = os.path.join(vam_dir, "diffusion_pytorch_model.safetensors.index.json")
    bin_index = os.path.join(vam_dir, "diffusion_pytorch_model.bin.index.json")

    if os.path.isfile(sf_index):
        results.append(check_safetensors_index(sf_index, "VAM safetensors index"))
    elif os.path.isfile(bin_index):
        results.append(check_safetensors_index(bin_index, "VAM bin index"))
    else:
        # Check for single-file checkpoint
        sf_file = os.path.join(vam_dir, "diffusion_pytorch_model.safetensors")
        bin_file = os.path.join(vam_dir, "diffusion_pytorch_model.bin")
        pth_files = list(Path(vam_dir).glob("*.pth")) if os.path.isdir(vam_dir) else []

        if os.path.isfile(sf_file):
            results.append(check_file(sf_file, "VAM single safetensors"))
        elif os.path.isfile(bin_file):
            results.append(check_file(bin_file, "VAM single bin"))
        elif pth_files:
            results.append(check_file(str(pth_files[0]), f"VAM pth file ({len(pth_files)} found)"))
        elif os.path.isdir(vam_dir):
            contents = os.listdir(vam_dir)
            results.append((f"[INVALID] VAM dir exists but no recognizable checkpoint format: {contents[:5]}...", False))
        else:
            results.append((f"[MISSING] VAM model dir: {vam_dir}", False))

    return results


def validate_wan(cfg: dict) -> List[Tuple[str, bool]]:
    """Validate Wan2.2 shared dependencies."""
    results = []

    # VAE
    vae_path = cfg["wan"]["vae"]["file"]
    results.append(check_file(vae_path, "Wan2.2 VAE"))

    # T5 encoder
    t5_path = cfg["wan"]["t5"]["checkpoint"]
    results.append(check_file(t5_path, "T5 text encoder"))

    # Tokenizer
    tok_dir = cfg["wan"]["t5"]["tokenizer_dir"]
    results.append(check_tokenizer_dir(tok_dir, "UMT5 tokenizer"))

    return results


def validate_simulator(cfg: dict) -> List[Tuple[str, bool]]:
    """Validate ACVS/Simulator checkpoint."""
    results = []
    sim_dir = cfg["tau0"]["simulator"]["model_dir"]

    sf_index = os.path.join(sim_dir, "diffusion_pytorch_model.safetensors.index.json")
    bin_index = os.path.join(sim_dir, "diffusion_pytorch_model.bin.index.json")
    sf_single = os.path.join(sim_dir, "diffusion_pytorch_model.safetensors")
    bin_single = os.path.join(sim_dir, "diffusion_pytorch_model.bin")

    if os.path.isfile(sf_index):
        results.append(check_safetensors_index(sf_index, "Simulator safetensors index"))
    elif os.path.isfile(bin_index):
        results.append(check_safetensors_index(bin_index, "Simulator bin index"))
    elif os.path.isfile(sf_single):
        results.append(check_file(sf_single, "Simulator single safetensors"))
    elif os.path.isfile(bin_single):
        results.append(check_file(bin_single, "Simulator single bin"))
    elif os.path.isdir(sim_dir):
        results.append((f"[INVALID] Simulator dir exists but no recognizable checkpoint", False))
    else:
        results.append((f"[MISSING] Simulator model dir: {sim_dir}", False))

    return results


def validate_statistics(cfg: dict) -> List[Tuple[str, bool]]:
    """Validate normalization statistics files."""
    results = []

    policy_stat = cfg["statistics"]["policy"]
    results.append(check_json_parseable(policy_stat, "Policy statistics JSON"))

    sim_stat = cfg["statistics"]["simulator"]
    results.append(check_json_parseable(sim_stat, "Simulator statistics JSON"))

    return results


def validate_tau0_configs() -> List[Tuple[str, bool]]:
    """Check that τ₀ configs are valid YAML."""
    results = []
    tau0_root = os.path.join(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tau-0-wm")

    configs_to_check = [
        "configs/deployment/tau_pretrain_rela_eef6d.yaml",
        "configs/deployment/tau_simulator.yaml",
    ]
    for cfg_path in configs_to_check:
        full = os.path.join(tau0_root, cfg_path)
        results.append(check_yaml_parseable(full, cfg_path))

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate τ₀-WM checkpoints for V0-B readiness")
    parser.add_argument(
        "-c", "--config",
        default=os.path.join(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs/checkpoints.yaml"),
        help="Path to checkpoints.yaml",
    )
    parser.add_argument(
        "--vam-only",
        action="store_true",
        help="Only validate VAM (minimum for V0-B smoke 0A/0B)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"[FATAL] Config file not found: {args.config}")
        sys.exit(2)

    try:
        cfg = load_yaml_config(args.config)
    except Exception as e:
        print(f"[FATAL] Cannot parse config: {e}")
        sys.exit(2)

    all_ok = True
    all_results = []

    print("=" * 70)
    print("τ₀-WM V0-B Checkpoint Validator")
    print("=" * 70)

    # 1. VAM
    print("\n--- VAM Checkpoint ---")
    results = validate_vam(cfg)
    for msg, ok in results:
        print(msg)
        all_ok = all_ok and ok
    all_results.extend(results)

    # 2. Wan2.2 shared dependencies
    print("\n--- Wan2.2 Shared Dependencies ---")
    results = validate_wan(cfg)
    for msg, ok in results:
        print(msg)
        all_ok = all_ok and ok
    all_results.extend(results)

    # 3. Statistics
    print("\n--- Statistics Files ---")
    results = validate_statistics(cfg)
    for msg, ok in results:
        print(msg)
        all_ok = all_ok and ok
    all_results.extend(results)

    # 4. Config files
    print("\n--- τ₀ Config Files ---")
    results = validate_tau0_configs()
    for msg, ok in results:
        print(msg)
        all_ok = all_ok and ok
    all_results.extend(results)

    # 5. Simulator (optional for V0-B basic smoke)
    if not args.vam_only:
        print("\n--- Simulator Checkpoint (optional for V0-B basic) ---")
        results = validate_simulator(cfg)
        for msg, ok in results:
            print(msg)
        all_results.extend(results)

    # Summary
    print("\n" + "=" * 70)
    missing_count = sum(1 for msg, ok in all_results if not ok)
    ok_count = sum(1 for msg, ok in all_results if ok)

    if missing_count == 0:
        print(f"RESULT: ALL {ok_count} CHECKS PASSED")
        print("V0-B checkpoint prerequisites: SATISFIED")
        sys.exit(0)
    else:
        print(f"RESULT: {ok_count} OK, {missing_count} MISSING/INVALID")
        print("V0-B checkpoint prerequisites: NOT SATISFIED")
        sys.exit(1)


if __name__ == "__main__":
    main()
