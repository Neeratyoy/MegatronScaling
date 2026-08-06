import argparse
import math
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict


# CONSTRAINTS
L_MIN = lambda L, _L_min=None: L//2 if _L_min is None else min(L//2, _L_min)
L_MAX = lambda L, _L_max=None: max(L, 100) if _L_max is None else _L_max
H_MIN = lambda H, _H_min=None: max(H // 8, 64) if _H_min is None else  _H_min
H_MAX = lambda N, V: N // (2 * V + 1)
L_GIVEN_H = lambda N, H, V=50304: (N - ((2 * V + 1) * H)) // (16 * H ** 2 + 6 * H)
L_INVERSE = lambda N, L, V=50304: int(np.round(
    max(np.roots([16 * L, 2 * V + 1 + 6 * L, -N]))
))  # calculates H

RHO_MIN = lambda N, H_min, V=50304, r_min=10: max(r_min, (16 * H_min ** 3 + 6 * H_min ** 2) / (N - ((2 * V + 1) * H_min)))
RHO_MAX = lambda N, L_min, V=50304: L_INVERSE(N, L_min, V) / L_min

RHO_INVERSE = lambda N, rho, V=50304: int(np.round(
    max([r.real for r in np.roots([16, 6, (2*V + 1) * rho, -N * rho])])
))  # calculates H

RHO_LOG_BASE = 3/2


def param_counter(
    num_layers: int,
    hidden_size: int,
    ffn_hidden_size: int | None = None,
    ffn_mult: float = 4.0,
    swiglu: bool = True,
    tied_embeddings: bool = False,
    vocab_size: int = 50304,
    attn_residual: bool = True,
    attn_res_learnable_norm: bool = True
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


def legitimize_config(
    H: int, L:int, AR: float, N: int, NH: int, band: int = 64
) -> Dict[str, int]:
    def _generate_new_H(H, NH, band: int=64):
        _H_candidates = list(range(H-band, H+band,1))
        _head_div = np.array(_H_candidates)[np.array(_H_candidates) % NH == 0]
        _idx = np.argmin(np.abs(_head_div - H))
        new_H = _head_div[_idx]
        return new_H

    _new_H = _generate_new_H(H, NH, band=band)
    _L = L_GIVEN_H(N, _new_H)
    _AR = _new_H / _L

    return {
        "L": _L, "H": _new_H, "AR": _AR
    }


def resolve_num_heads(H, NH, head_dim_min):
    """Shrink NH by 1 until H splits evenly with head_dim >= head_dim_min.
    Returns a valid NH, or None (failure) if it bottoms out at NH == 2."""
    while NH >= 2:
        if H % NH == 0 and H // NH >= head_dim_min:
            return NH
        NH -= 1
    return None


def get_args():
    parser = argparse.ArgumentParser(description="Generate aspect ratios for model configurations.")

    parser.add_argument("config_file", type=str, help="Path to the configuration file.")

    parser.add_argument("--num_points", type=int, default=5, help="Number of points to generate for each aspect ratio.")
    parser.add_argument("--log_spacing", action="store_true", help="Use logarithmic spacing for aspect ratios.")
    parser.add_argument("--vocab_size", default=50304, type=int, help="Vocabulary size for the model.")
    parser.add_argument("--min_rho", default=10, type=float, help="Minimum aspect ratio (H/L) to consider.")

    parser.add_argument("--L_min", type=int, default=None, help="Minimum number of layers.")
    parser.add_argument("--L_max", type=int, default=None, help="Maximum number of layers.")
    parser.add_argument("--H_min", type=int, default=None, help="Minimum hidden size.")

    parser.add_argument("--head_dim_min", type=int, default=16, help="Minimum dimension of each attention head.")
    parser.add_argument("--remove_close_to_default", action="store_true", help="Remove configurations that are too close to each other.")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--dump_dir", type=str, default=None, help="Directory to dump the generated aspect ratios.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # using Python's os package simulate the bash `source ${config_file}` command
    config_dict = {}
    with open(args.config_file) as f:
        for line in f:
            if "=" in line:
                key, value = line.split("=", 1)
                config_dict[key.strip()] = value.strip()

    L = int(config_dict["NUM_LAYERS"])
    H = int(config_dict["HIDDEN_SIZE"])
    F = int(config_dict["FFN_HIDDEN_SIZE"])
    NH = int(config_dict.get("NUM_ATTENTION_HEADS"))
    V = int(config_dict.get("VOCAB_SIZE", args.vocab_size))

    _params = param_counter(
        num_layers=L, 
        hidden_size=H, 
        ffn_hidden_size=F, 
        vocab_size=V, 
        tied_embeddings=False,
        attn_residual=True,
        attn_res_learnable_norm=True
    )
    if args.verbose:
        print(f"Total parameters for L={L}, H={H}, F={F}: {_params['total_params'] / 1e6:.2f}M")

    N = _params['total_params']

    ar_df = pd.DataFrame.from_dict({
        "L": [L],
        "H": [H],
        "AR": [H / L],
    }, orient="columns")

    # calculating AR bounds
    r_min = RHO_MIN(N, H_min=H_MIN(H, args.H_min), V=V, r_min=args.min_rho)
    r_max = RHO_MAX(N, L_min=L_MIN(L, args.L_min), V=V)
    if args.verbose:
        print(f"Aspect ratio bounds for N={N}, L={L}, H={H}: r_min={r_min:.2f}, r_max={r_max:.2f}")

    ar_list = []
    if args.log_spacing:
        ar_list = RHO_LOG_BASE ** np.linspace(
            math.log(r_min, RHO_LOG_BASE), math.log(r_max, RHO_LOG_BASE), args.num_points
        )
    else:
        ar_list = np.linspace(r_min, r_max, args.num_points)

    for ar in ar_list:
        _H = RHO_INVERSE(N, ar, V=V)
        _L = L_GIVEN_H(N, _H, V=V)
        _df = pd.DataFrame.from_dict({"L": [_L], "H": [_H], "AR": [ar]}, orient="columns")
        ar_df = pd.concat([ar_df, _df])

    ar_df["legit"] = ar_df.apply(lambda row: True if row["H"] % NH == 0 else False, axis=1)

    if args.verbose:
        print("Generated aspect ratio grid:")
        print(ar_df.sort_values(by="AR").to_string(index=False))
        print()

    for ar, row in ar_df.iterrows():
        _temp = legitimize_config(
            H=row["H"],
            L=row["L"],
            AR=row["AR"],
            NH=NH,
            N=N,
            band=128,
        )
        if ar != _temp["AR"]:
            if args.verbose:
                print(f"Adding new row for legit config: {row['AR']:.2f} -> {_temp['AR']:.2f}")
            _df = pd.DataFrame.from_dict({"L": [_temp["L"]], "H": [_temp["H"]], "AR": [_temp["AR"]], "legit": [True]}, orient="columns")
            ar_df = pd.concat([ar_df, _df])
    ar_df.set_index("AR", inplace=True)
    ar_df.sort_index(inplace=True)
    ar_df["NH"] = ar_df["H"].apply(lambda h: resolve_num_heads(int(h), NH, args.head_dim_min))
    ar_df["head_dim"] = ar_df["H"] / ar_df["NH"]
    if args.verbose:
        print()
        print(ar_df)
        print()
        print("Cleaning:")
    ar_df = ar_df.loc[ar_df["NH"].notnull()]
    if args.remove_close_to_default:
        ar_df = ar_df.loc[[False if np.abs(idx - (H/L)) < 5 else True for idx in ar_df.index.values]]
    ar_df = ar_df.loc[ar_df["legit"] == True]
    ar_df.drop_duplicates(subset=["L"], keep="first", inplace=True)
    print(ar_df)

    if args.dump_dir is not None:
        dump_path = Path(args.dump_dir)
        dump_path.mkdir(parents=True, exist_ok=True)
        for ar, row in ar_df.iterrows():
            _config = config_dict.copy()
            _config.update({
                "NUM_LAYERS": row["L"],
                "HIDDEN_SIZE": row["H"],
                "NUM_ATTENTION_HEADS": row["NH"],
                "FFN_HIDDEN_SIZE": row["H"] * (int(config_dict["FFN_HIDDEN_SIZE"]) // int(config_dict["HIDDEN_SIZE"])),
            })
            _name = f"{Path(args.config_file).stem.split('AR')[0]}AR{int(np.round(ar))}.info"
            print()
            print(f"Filename: {_name}")
            _line = ""
            for k, v in _config.items():
                _line += f"{k}={v}\n"
            with open(Path(args.dump_dir) / _name, "w") as f:
                f.write(_line)
            if args.verbose:
                print(f"Saved config to {Path(args.dump_dir) / _name}")
        print("Done generating configs into directory:", args.dump_dir)
# end of file