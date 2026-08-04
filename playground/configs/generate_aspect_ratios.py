import numpy as np
import os
from pathlib import Path
import sys
from typing import Dict


def param_counter(
    num_layers: int,
    hidden_size: int,
    ffn_hidden_size: int | None = None,
    ffn_mult: float = 4.0,
    swiglu: bool = True,
    tied_embeddings: bool = False,
    vocab_size: int = 50304,
    attn_residual: bool = False,
    attn_res_learnable_norm: bool = False
) -> Dict[str, int]:
    h = hidden_size
    f = ffn_hidden_size if ffn_hidden_size is not None else int(ffn_mult * h)

    attn_params = 4 * h * h                    # W_q, W_k, W_v, W_o  (MHA, no bias)
    ffn_params = (2 + int(swiglu)) * h * f     # up, down (+ gate if SwiGLU)
    norm_params = 2 * h                        # 2 x RMSNorm per layer

    per_layer_params = attn_params + ffn_params + norm_params

    hidden_params = num_layers * per_layer_params
    embed_params = (2 - int(tied_embeddings)) * vocab_size * h
    final_norm_params = h                      # pre-head RMSNorm

    attn_res_params = attn_res_norm_params = 0
    if attn_residual:    
        attn_res_norm_params = (num_layers * 2 * h + h) if attn_res_learnable_norm else 0
        attn_res_params = 2 * num_layers * h

    hidden_params += attn_res_params + attn_res_norm_params

    return {
        "hidden_params": hidden_params,
        "embed_params": embed_params,
        "final_norm_params": final_norm_params,
        "total_params": hidden_params + embed_params + final_norm_params,
    }


if __name__ == "__main__":
    config_file = Path(sys.argv[1])

    # using Python's os package simulate the bash `source ${config_file}` command
    config_dict = {}
    with open(config_file) as f:
        for line in f:
            if "=" in line:
                key, value = line.split("=", 1)
                config_dict[key.strip()] = value.strip()

    print(param_counter(
        num_layers=int(config_dict["NUM_LAYERS"]),
        hidden_size=int(config_dict["HIDDEN_SIZE"]),
        ffn_hidden_size=int(config_dict["FFN_HIDDEN_SIZE"]),
    ))
# end of file