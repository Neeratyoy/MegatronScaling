import json
import pandas as pd
from pathlib import Path
from typing import List
import argparse


def is_complete(path: Path) -> int | None:
    """Status from the checkpoint dir: current vs target `train_iters`."""
    p = Path(path)
    if not p.exists():
        return None
    if not (p / "latest_checkpointed_iteration.txt").exists():
        return None
    if not (p / "run_config.json").exists():
        return None
    target = int(json.loads((p / "run_config.json").read_text())["train_iters"])
    current = int((p / "latest_checkpointed_iteration.txt").read_text().strip())
    if current > target:
        return None
    return 1 if current == target else 0


def is_complete_metrics(path: Path, metric_file: str = "run_metrics.parquet") -> int | None:
    """Stricter status: requires `run_metrics.parquet` itself to reach `train_iters`."""
    p = Path(path)
    metrics_path = p / metric_file
    config_path = p / "run_config.json"
    if not metrics_path.exists() or not config_path.exists():
        return None
    target = int(json.loads(config_path.read_text())["train_iters"])
    try:
        df = pd.read_parquet(metrics_path)
    except Exception:
        return None
    if df.empty:
        return None
    # filter/preprocess
    df = df.loc[df["learning-rate"].notna()]
    current = int(df["step"].max())
    if current > target:
        return None
    return 1 if current == target else 0


def is_complete_both(path: Path, metric_file: str = "run_metrics.parquet") -> int | None:
    """Requires checkpoint status and metrics status to agree; disagreement -> None (failed)."""
    ckpt_status = is_complete(path)
    metrics_status = is_complete_metrics(path, metric_file)
    if ckpt_status != metrics_status:
        return None
    return ckpt_status


CHECK_FNS = {
    "checkpoint": is_complete,
    "metrics": is_complete_metrics,
    "both": is_complete_both,
}


def check_run_status_given_a_run_file(
    base_path: Path, run_file: Path, seed: int = 123456, check: str = "checkpoint"
) -> pd.DataFrame:
    # load the run file
    run_file = Path(run_file)
    if not run_file.exists():
        raise FileNotFoundError(f"Run file {run_file} does not exist.")
    with open(run_file, "r") as f:
        run_paths = [line.strip() for line in f.readlines() if line.strip()]

    base_path = Path(base_path)
    check_fn = CHECK_FNS[check]

    # creating status map
    path_map = {}
    for i, run_name in enumerate(run_paths):
        _run = base_path / run_name / f"seed={seed}"
        path_map[i] = {"path": _run, "status": check_fn(_run)}
    path_map = pd.DataFrame.from_dict(path_map, orient="index")
    
    return path_map


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=Path, required=True, help="Base path where runs are stored")
    parser.add_argument("--run_file", type=Path, required=True, help="File containing list of runs to check")
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument(
        "--check",
        choices=list(CHECK_FNS),
        default="checkpoint",
        help="'checkpoint': latest_checkpointed_iteration.txt vs train_iters (default, cheapest). "
        "'metrics': run_metrics.parquet max(step) vs train_iters (stricter, catches broken logging). "
        "'both': status only counts if checkpoint and metrics agree, else treated as failed."
    )
    parser.add_argument(
        "--pending", 
        action="store_true",
        help="print only comma-separated indices with status 0"
    )    
    parser.add_argument(
        "--failed",
        action="store_true",
        help="print only comma-separated indices with status None"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    df = check_run_status_given_a_run_file(args.base_path, args.run_file, args.seed, check=args.check)

    print()
    print("=== Run Status Summary ===")
    print(f"Number of runs: {len(df)}")
    print(f"Number of completed runs: {len(df[df['status'] == 1])}")
    print(f"Number of pending runs: {len(df[df['status'] == 0])}")
    print(f"Number of failed runs: {len(df[df['status'].isna()])}")
    print("==========================")

    if args.pending:
        print("Pending run indices (comma-separated):", end=" ")
        print(",".join(str(i) for i in df.index[df["status"] == 0]))
        print("Pending paths (comma-separated):")
        pending_runs = df[df["status"] == 0]
        pending_list = [f"{row['path']}" for i, row in pending_runs.iterrows()]
        print("\n".join(pending_list))
        print("==========================")
    if args.failed:
        print("Failed (or not started) run indices (comma-separated):", end=" ")
        print(",".join(str(i) for i in df.index[df["status"].isna()]))
        print("Failed paths (comma-separated):")
        failed_runs = df[df["status"].isna()]
        failed_list = [f"{row['path']}" for i, row in failed_runs.iterrows()]
        print("\n".join(failed_list))
    print("==========================")
# end of file