#!/usr/bin/env python3
"""PART4A — Multi-task VAM inference worker (subprocess, tau0_wm env).

Loads the theta_init_multi_v0 checkpoint ONCE and serves per-request inference
over a stdin/stdout pickle protocol. Each request carries its own task
statistics, instruction, and execution contract, so a single 5.5B-model load
serves all 49 tasks (per-task statistics are swapped in-place, mirroring
TauPolicy.__init__ normalization).

Protocol (byte stream, big-endian):
    request : 4-byte length + pickle(dict)
    response: 4-byte length + pickle(dict)

Request dict:
    cameras             : {head_camera: rgb_uint8(H,W,3), left_camera: ..., right_camera: ...}
    endpose             : {left_endpose: [xyz+quat_wxyz], right_endpose: [...],
                           left_gripper: float, right_gripper: float}
    statistics          : str path to task statistics_relative_v2.json
    instruction         : str natural-language task instruction
    task_name           : str (used only to build the canonical obs dict)
    num_inference_steps : int (default 5)
    execution_step      : int (default 33; broker cache horizon)
    sample_solver       : str (default "unipc")
    new_episode         : bool (re-seed torch RNG)
    episode_seed        : int

Response dict:
    robotwin_action : (16,) float32 ndarray (one brokered action)
    tau_action_shape: [33,20] (shape of the most recent native VAM forward)
    model_forward    : bool (whether this request exhausted the broker cache)
    error           : None or str
"""
import sys, os, struct, pickle, argparse, json, random
import numpy as np
import torch

TAU0_ROOT = os.path.join(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tau-0-wm")
CAUSAL_ROOT = os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAUNCH_DIR = os.getcwd()  # captured before chdir, for resolving relative CLI paths
sys.path.insert(0, TAU0_ROOT)
sys.path.insert(0, CAUSAL_ROOT)
os.chdir(TAU0_ROOT)

# Disable the training-only forward-pass path (avoids loading optimizer/etc).
import utils.model_utils
utils.model_utils.forward_pass = lambda *a, **kw: None

from adapters.robotwin.observation_adapter import adapt_observation
from adapters.robotwin.action_adapter import adapt_tau_action_to_robotwin
from web_infer_utils.TauPolicy import TauPolicy
from web_infer_utils.openpi_client.action_chunk_broker import ActionChunkBroker
from models.wan_2_2_models.transformers.attention import set_attention_backend


def pin_math_sdpa():
    # Environment-compat only: pin SDPA to the conservative math backend so the
    # VAM inference subprocess does not fault on the post-reboot driver. Does not
    # change the policy, statistics, or action contract.
    set_attention_backend(attention_impl="sdpa")
    set_attention_backend(attention_impl="sdpa", sdpa_backend="math")
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def swap_statistics(model: TauPolicy, statistics: str) -> None:
    """Re-point the policy's normalization to a task's statistics in-place.

    Mirrors TauPolicy.__init__ mean/std loading exactly (same dtype inference) so
    per-task normalization is bit-identical to a fresh per-task load.
    """
    info = json.load(open(statistics, "r"))
    model.act_mean = torch.tensor(info["action"]["mean"]).unsqueeze(0).unsqueeze(0)
    model.act_std = torch.tensor(info["action"]["std"]).unsqueeze(0).unsqueeze(0) + 1e-6
    model.sta_mean = np.array(info["state"]["mean"])
    model.sta_std = np.array(info["state"]["std"]) + 1e-6


class BrokeredTauPolicy:
    """Adapt TauPolicy to the official one-action-at-a-time broker interface."""

    def __init__(self, model: TauPolicy):
        self.model = model
        self.forward_count = 0
        self.last_tau_action_shape = None
        self.last_stats = None
        self.last_episode_seed = None

    def infer(self, req):
        statistics = req["statistics"]
        if statistics != self.last_stats:
            swap_statistics(self.model, statistics)
            self.last_stats = statistics

        episode_seed = int(req.get("episode_seed", 0))
        if req.get("new_episode", False):
            torch.manual_seed(episode_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(episode_seed)
            # Tau's pipeline chooses its per-forward torch.Generator seed via
            # Python random when no explicit seed is supplied. Reset that RNG
            # only for the opt-in deterministic cadence comparison; ordinary
            # evaluation retains the established sampling behavior.
            if req.get("deterministic_comparison", False):
                random.seed(episode_seed)
                np.random.seed(episode_seed)
            self.last_episode_seed = episode_seed

        cameras_wrapped = {k: {"rgb": v} for k, v in req["cameras"].items()}
        robotwin_obs = {"observation": cameras_wrapped, "endpose": req["endpose"]}
        tau_input = adapt_observation(robotwin_obs, task_name=req.get("task_name", ""))
        tau_input["prompt"] = req["instruction"]

        action_horizon = int(req.get("execution_step", self.model.action_chunk))
        if action_horizon != self.model.action_chunk:
            raise ValueError(
                f"broker horizon {action_horizon} must equal native action_chunk "
                f"{self.model.action_chunk}"
            )

        with torch.inference_mode():
            action_tau = self.model.play(
                **tau_input,
                num_inference_steps=int(req.get("num_inference_steps", 5)),
                execution_step=action_horizon,
                sample_solver=req.get("sample_solver", "unipc"),
            )
        rtw_action = adapt_tau_action_to_robotwin(action_tau)
        expected_tau = (self.model.action_chunk, self.model.action_dim)
        expected_rtw = (self.model.action_chunk, 16)
        if action_tau.shape != expected_tau or rtw_action.shape != expected_rtw:
            raise ValueError(
                f"invalid native chunks tau={action_tau.shape}, robotwin={rtw_action.shape}; "
                f"expected {expected_tau} and {expected_rtw}"
            )
        self.forward_count += 1
        self.last_tau_action_shape = list(action_tau.shape)
        return {"robotwin_action": np.asarray(rtw_action, dtype=np.float32)}

    def reset(self):
        self.model.reset()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    args = parser.parse_args()

    # Resolve CLI paths relative to the launch directory (the worker chdirs to
    # TAU0_ROOT at import, so bare relative paths would otherwise resolve wrong).
    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(LAUNCH_DIR, p)
    args.checkpoint = os.path.abspath(_abs(args.checkpoint))
    args.stats = os.path.abspath(_abs(args.stats))

    pin_math_sdpa()

    import yaml
    from yaml import Loader, Dumper, load, dump
    cfg_path = os.path.join(os.environ.get("CAUSALWAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs/runtime/vam_deploy.yaml")
    cd = load(open(cfg_path), Loader=Loader)
    cd["diffusion_model"]["model_path"] = args.checkpoint
    cd["statistics_file"] = args.stats
    cd["action_type"] = "relative"
    cd["action_space"] = "eef6d"
    tmp_cfg = f"/tmp/theta_init_vam_cfg_{os.getpid()}.yaml"
    with open(tmp_cfg, "w") as f:
        dump(cd, f, Dumper=Dumper)

    device = torch.device("cuda:0")
    model = TauPolicy(config_file=tmp_cfg, device=device, rank=0,
                      compile_model=False, attention_impl="sdpa",
                      sdpa_backend="math",
                      enable_self_attn_fused_qkv=True,
                      enable_context_null_cache=True)
    model.diffusion_model.eval()
    print(f"VAM_READY checkpoint={args.checkpoint}", flush=True)

    broker_policy = BrokeredTauPolicy(model)
    broker = ActionChunkBroker(broker_policy, action_horizon=model.action_chunk)

    while True:
        try:
            len_bytes = sys.stdin.buffer.read(4)
            if len(len_bytes) < 4:
                break
            msg_len = struct.unpack(">I", len_bytes)[0]
            req = pickle.loads(sys.stdin.buffer.read(msg_len))
        except Exception:
            break

        try:
            execution_step = int(req.get("execution_step", model.action_chunk))
            if execution_step != model.action_chunk:
                raise ValueError(
                    f"execution_step={execution_step} does not match native "
                    f"action_chunk={model.action_chunk}"
                )
            if req.get("new_episode", False):
                broker.reset()
            forwards_before = broker_policy.forward_count
            broker_result = broker.infer(req)

            resp = {
                "robotwin_action": np.asarray(
                    broker_result["robotwin_action"], dtype=np.float32
                ),
                "tau_action_shape": broker_policy.last_tau_action_shape,
                "model_forward": broker_policy.forward_count > forwards_before,
                "model_forward_count": broker_policy.forward_count,
                "error": None,
            }
        except Exception as e:
            import traceback
            resp = {"robotwin_action": None, "tau_action_shape": None,
                    "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=4)}

        resp_bytes = pickle.dumps(resp)
        sys.stdout.buffer.write(struct.pack(">I", len(resp_bytes)))
        sys.stdout.buffer.write(resp_bytes)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
