import argparse
import json
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from playground.configs.generate_aspect_ratios import param_counter


CONFIG_VARS_OF_INTEREST = [
    "num_layers",
    "hidden_size",
    "ffn_hidden_size",
    "train_iters",
    "seq_length",
    "micro_batch_size",
    "global_batch_size",
    "weight_decay",
    "lr",
    "lr_warmup_iters",
    "lr_wsd_decay_iters",
    "vocab_size",
    "seed",
]
DEFAULT_VOCAB_SIZE = 50304


def smooth_metric(
    s: pd.Series,
    method: str = "rolling",   # "rolling" or "ewm"
    window: int = 25,
    alpha: float | None = None,
    min_periods: int = 1,
) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")

    if method == "rolling":
        return s.rolling(window=window, min_periods=min_periods).mean()

    if method == "ewm":
        # If alpha is not set, derive a reasonable one from window
        # (similar smoothness scale as rolling window).
        if alpha is None:
            alpha = 2 / (window + 1)
        return s.ewm(alpha=alpha, adjust=False, min_periods=min_periods).mean()

    raise ValueError("method must be 'rolling' or 'ewm'")


def _recursive_path_explore(path: Path, endfile: str = "run_metrics.parquet"):
    if path.is_file() and path.name == endfile:
        return [path]
    elif path.is_dir():
        _paths = []
        for child in path.iterdir():
            _paths.extend(_recursive_path_explore(child, endfile))
    else:
        return []
    return _paths


def _read_parquet(
    path: Path,
    col: str = "lm loss",
    store_last: bool = False,
    smooth: bool = False,
    smooth_key: str = "lm loss",
    smooth_window: int = 25,
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_values("step").reset_index(drop=True)
    df = df.loc[df[col].dropna().index]

    if smooth:
        df[f"{smooth_key}_smoothed"] = smooth_metric(df[smooth_key], window=smooth_window)

    with open(path.parent / "run_config.json", "r") as f:
        config = json.load(f)

    for k in CONFIG_VARS_OF_INTEREST:
        df[k] = config.get(k, None)
    _params = param_counter(
        num_layers=config.get("num_layers", 0),
        hidden_size=config.get("hidden_size", 0),
        ffn_hidden_size=config.get("ffn_hidden_size", 0),
        swiglu=config.get("swiglu", True),
        vocab_size=(
            config.get("vocab_size", DEFAULT_VOCAB_SIZE) 
            if config.get("vocab_size", DEFAULT_VOCAB_SIZE) 
            else config.get("padded_vocab_size", DEFAULT_VOCAB_SIZE)
        ),
    )
    df["total_params"] = _params["total_params"]
    df["embed_params"] = _params["embed_params"]
    df["run_type"] = path.parent.parent.name
    df["total_tokens"] = df["global_batch_size"] * df["seq_length"] * df["step"]

    if store_last:
        df = df.sort_values("step").tail(1)
        df["path"] = str(path)

    return df


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path", 
        type=Path, 
        required=True, 
        help="Base path where runs are stored"
    )
    parser.add_argument(
        "--metric_file", 
        type=str, 
        default="run_metrics.parquet",
        help="Name of the metric file to look for in each run directory. " \
        "[IMPORTANT]: this file is used as a stop condition for the recursive search."
    )
    parser.add_argument(
        "--col", 
        type=str, 
        default="lm loss",
        help="Column name to filter the DataFrame on. Only rows with non-null values in this column will be kept."
    )
    parser.add_argument(
        "--store_last",
        action="store_true",
        help="If set, only the last row of each run's DataFrame will be kept."
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="If set, add a smoothed version of --smooth_key as a new '<smooth_key>_smoothed' column."
    )
    parser.add_argument(
        "--smooth_key",
        type=str,
        default="lm loss",
        help="Column name to smooth when --smooth is set."
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=25,
        help="Rolling window size used when smoothing --smooth_key."
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="aggregated_run_metrics.parquet",
        help="Filename for the aggregated metrics output."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    all_paths = _recursive_path_explore(args.base_path, endfile=args.metric_file)

    _df = pd.DataFrame()
    for i, path in enumerate(all_paths, start=1):
        print(f"Processing {i}/{len(all_paths)}: {path}", end="\r")
        _df = pd.concat(
            [_df, _read_parquet(
                path,
                col=args.col,
                store_last=args.store_last,
                smooth=args.smooth,
                smooth_key=args.smooth_key,
                smooth_window=args.smooth_window,
            )],
            ignore_index=True
        )
    print()
    _df.to_parquet(args.base_path / args.filename, index=False)
    print(f"Aggregated metrics saved to: {args.base_path / args.filename}")
# end of file