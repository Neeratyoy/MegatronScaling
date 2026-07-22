import numpy as np
import os
from pathlib import Path
import sys
from typing import Dict


def param_counter(
    num_layers: int,
    hidden_size: int,
    ffn_hidden_size: int,
    swiglu: bool = True,
    tied_embeddings: bool = True,
    vocab_size: int = 50304,
) -> Dict[str, int]:
    per_layer_params = (
        4 * hidden_size * hidden_size  # FFN
        + (int(swiglu) + 2) * hidden_size * ffn_hidden_size  # Swiglu
        + 4 * hidden_size  # norms
    )

    embed_params = (int(tied_embeddings) + 1) * vocab_size * hidden_size

    total_params = num_layers * per_layer_params + embed_params

    return {
        "hidden_params": num_layers * per_layer_params,
        "embed_params": embed_params,
        "total_params":total_params,
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