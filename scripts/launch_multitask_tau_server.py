#!/usr/bin/env python3
"""One shared Tau checkpoint server with generic per-request normalization."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames
import yaml


ROOT = Path(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TAU = ROOT / "tau-0-wm"
sys.path.insert(0, str(TAU))
os.chdir(TAU)

import utils.model_utils

utils.model_utils.forward_pass = lambda *args, **kwargs: None
from web_infer_utils.TauPolicy import TauPolicy
from web_infer_utils.openpi_client import msgpack_numpy


class MultiTaskStatisticsServer:
    def __init__(self, policy: TauPolicy, host: str, port: int):
        self.policy = policy
        self.host = host
        self.port = port

    def set_statistics(self, statistics: dict) -> None:
        action_mean = np.asarray(statistics["action"]["mean"], dtype=np.float32)
        action_std = np.asarray(statistics["action"]["std"], dtype=np.float32)
        state_mean = np.asarray(statistics["state"]["mean"], dtype=np.float64)
        state_std = np.asarray(statistics["state"]["std"], dtype=np.float64)
        if not all(value.shape == (20,) for value in (action_mean, action_std, state_mean, state_std)):
            raise ValueError("statistics must contain four finite 20D vectors")
        if not all(np.isfinite(value).all() for value in (action_mean, action_std, state_mean, state_std)):
            raise ValueError("non-finite per-dataset statistics")
        self.policy.act_mean = torch.from_numpy(action_mean).unsqueeze(0).unsqueeze(0)
        self.policy.act_std = torch.from_numpy(action_std).unsqueeze(0).unsqueeze(0) + 1e-6
        self.policy.sta_mean = state_mean
        self.policy.sta_std = state_std + 1e-6

    async def handler(self, websocket):
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack({"server": "theta_init_multi_v0", "per_dataset_statistics": True}))
        while True:
            try:
                request = msgpack_numpy.unpackb(await websocket.recv())
                statistics = request.pop("statistics")
                reset = bool(request.pop("reset", False))
                self.set_statistics(statistics)
                if reset:
                    self.policy.reset()
                action = self.policy.play(**request)
                await websocket.send(packer.pack({"actions": action}))
            except websockets.ConnectionClosed:
                break
            except BaseException:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="multitask policy server error",
                )
                raise

    async def run(self):
        async with websocket_server.serve(
            self.handler, self.host, self.port, compression=None, max_size=None
        ) as server:
            print("MULTITASK_TAU_SERVER_READY", flush=True)
            await server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--initial-statistics", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/runtime/vam_deploy.yaml").read_text())
    config["diffusion_model"]["model_path"] = args.checkpoint
    config["statistics_file"] = args.initial_statistics
    config["action_type"] = "relative"
    config["action_space"] = "eef6d"
    config_path = Path(f"/tmp/theta_init_multi_server_{os.getpid()}.yaml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    device = torch.device(args.device)
    policy = TauPolicy(
        config_file=str(config_path),
        device=device,
        rank=device.index or 0,
        compile_model=False,
        attention_impl="sdpa",
        sdpa_backend="math",
        enable_self_attn_fused_qkv=True,
        enable_context_null_cache=True,
    )
    asyncio.run(MultiTaskStatisticsServer(policy, args.host, args.port).run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
