import os
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")



import argparse
from utils import import_custom_class
from utils.config_utils import expand_env_vars



def main():

    parser = argparse.ArgumentParser(
        description="Arguments for the main train program."
    )
    parser.add_argument('--config_file', type=str, required=True, help='Path for the config file')
    parser.add_argument('--runner_class_path', type=str, default="runner/posttrain.py")
    parser.add_argument('--runner_class', type=str, default="Trainer")
    parser.add_argument('--mode', type=str, default="train")
    
    args = parser.parse_args()
    args.config_file = expand_env_vars(args.config_file)

    # PB-B2: pin attention to the conservative SDPA "math" backend.
    # The post-reboot driver (580.173.02 / CUDA 13.0) triggers intermittent
    # "CUDA error: misaligned address" from PyTorch's flash/mem-efficient SDPA
    # kernels on H100 (bfloat16). The math backend avoids those kernels.
    # This is an environment-compatibility fix ONLY -- it does NOT change the
    # training recipe (hyperparameters / statistics / data / objective are
    # unchanged). The inference path (TauPolicy) already pins attention_impl
    # explicitly; the training path was using the implicit "auto" default.
    import torch
    from models.wan_2_2_models.transformers.attention import set_attention_backend
    set_attention_backend(attention_impl="sdpa", sdpa_backend="math")
    # Belt-and-suspenders: also disable PyTorch's flash/mem-efficient SDPA
    # kernels globally, in case any code path calls F.scaled_dot_product_attention
    # outside of the Wan attention() helper (e.g. VAE / action head).
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass
    Runner = import_custom_class(
        args.runner_class, args.runner_class_path, 
    )

    if args.mode == "train":
                
        ### Trainer
        runner = Runner(args.config_file)
        runner.prepare_dataset()
        
        if not hasattr(sys.stdout, 'isatty'):
            sys.stdout.isatty = lambda: False
        if not hasattr(sys.stderr, 'isatty'):
            sys.stderr.isatty = lambda: False
        runner.prepare_models()
        
        runner.prepare_trainable_parameters()
        runner.prepare_optimizer()
        runner.prepare_for_training()
        runner.prepare_trackers()

        # logical_cpus = os.cpu_count()
        # os.environ["OMP_NUM_THREADS"] = str(logical_cpus//8)

        runner.train()

    else:
        raise NotImplementedError



if __name__ == "__main__":
  
    main()
    
