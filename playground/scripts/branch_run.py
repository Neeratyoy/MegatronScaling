import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from playground.scripts.check_run_status import is_complete


def stage_branches(trunk_dir: Path, seed: int = 123456, strict: bool = True) -> list[str]:
    trunk_dir = Path(trunk_dir)
    if is_complete(trunk_dir) != 1:
        raise ValueError(f"trunk not complete: {trunk_dir}")

    # NOTE: strict assumption on results directory structure:
    tag_dir = trunk_dir.parent      # .../lr3e-3_gb256_vanilla
    model_dir = tag_dir.parent      # .../100M_AR32
    tags = []

    for ckpt in sorted(trunk_dir.glob("iter_*")):
        if not ckpt.is_dir():
            continue
        tag = f"{tag_dir.name}_branch={ckpt.name}"
        run_dir = model_dir / tag / f"seed={seed}"
        tags.append(tag)
        if run_dir.exists() and is_complete(run_dir) == 1 and strict:
            continue
        iter_num = int(ckpt.name.split("_")[1])
        # check if this `iter_num` is lower than the `latest_checkpointed_iteration.txt`
        latest_iter_file = run_dir / "latest_checkpointed_iteration.txt"
        if latest_iter_file.exists():
            latest_iter = int(latest_iter_file.read_text().strip())
            if iter_num < latest_iter and strict:
                print(f"Skipping {tag} as it has already been checkpointed at iteration {latest_iter}")
                continue
        # creating and mapping the checkpoint to the new branch run directory
        run_dir.mkdir(parents=True, exist_ok=True)
        link_path = run_dir / ckpt.name
        if link_path.exists() or link_path.is_symlink():
            print(f"Skipping symlink for {tag} as {link_path} already exists")
        else:
            link_path.symlink_to(ckpt.absolute())
        (run_dir / "latest_checkpointed_iteration.txt").write_text(f"{iter_num}\n")

    return tags


def stage_grid(
    base_path: Path, run_file: Path, seed: int = 123456, strict: bool = True
) -> tuple[Path, list[str], list[str], pd.DataFrame]:
    base_path = Path(base_path)
    run_file = Path(run_file)
    trunk_tags = [l.strip() for l in run_file.read_text().splitlines() if l.strip()]

    branch_tags, skipped = [], []
    for tag in trunk_tags:
        trunk = base_path / tag / f"seed={seed}"
        if is_complete(trunk) != 1:
            skipped.append(tag)
            continue
        branch_tags.extend(stage_branches(trunk, seed, strict=strict))

    out = run_file.parent / f"branch_runs_{base_path.name}.txt"
    out.write_text("\n".join(branch_tags) + "\n")

    tags_df = pd.DataFrame.from_dict(
        {i: {
            "config": _tag.split("_branch=")[0], 
            "iter": _tag.split("_branch=")[-1].strip("iter_")} 
            for i, _tag in enumerate(branch_tags)
        }, 
        orient="index"
    )

    return out, branch_tags, skipped, tags_df


def sample_grid(
    tags_df: pd.DataFrame, num_points: int = None, min_frac: float = None
) -> tuple[pd.DataFrame, list[str] | None]:
    if min_frac is not None:
        out = []
        tags_df["iter"] = tags_df["iter"].astype(int)
        for config, grp in tags_df.groupby("config"):
            avail = np.sort(grp["iter"].unique())
            min_steps = min_frac * avail[-1]
            _min_id = np.searchsorted(avail, min_steps, side="left")
            # check if there are enough available iterations to sample from
            if _min_id >= len(avail):
                print(f"Skipping config {config} as it has no available iterations >= {min_steps}")
                continue
            if num_points is not None and num_points >= len(avail) - _min_id:
                print(f"Skipping config {config} as it has fewer available iterations than requested points")
                continue
            targets = np.linspace(avail[-1], avail[_min_id], num_points)
            # finding the closest available iteration to each target
            idx = np.abs(avail[:, None] - targets).argmin(axis=0)
            out.append(pd.DataFrame({"config": config, "iter": np.unique(avail[idx])}))
        result = pd.concat(out, ignore_index=True)

        branch_tags = []
        for _, row in result.iterrows():
            branch_tags.append(f"{row['config']}_branch=iter_{row['iter']:07d}")

        return result, branch_tags
    else:
        print("No filtering applied, returning original tags_df")
        return tags_df, None


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_path",
        type=Path,
        required=True,
        help="Base path where runs are stored"
    )
    parser.add_argument(
        "--run_file",
        type=Path,
        required=True,
        help="File containing list of runs to check"
    )
    parser.add_argument(
        "--num_points",
        default=None,
        type=int,
        help="Number of steps to linearly sample from the grid (default: all)"
    )
    parser.add_argument(
        "--min_frac",
        default=None,
        type=float,
        help="Minimum fraction of each config's max iter to consider for branching (default: None, i.e. no minimum)"
    )
    parser.add_argument(
        "--renew",
        action="store_true",
        help="Whether to renew the branches (default: False)"
    )
    parser.add_argument("--seed", type=int, default=123456)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.min_frac is not None:
        assert 0 < args.min_frac <= 1, "min_frac must be between 0 and 1"

    out, branch_tags, skipped, tags_df = stage_grid(
        args.base_path, args.run_file, args.seed, strict=not args.renew
    )
    tags_df, branch_tags_filtered = sample_grid(tags_df, args.num_points, args.min_frac)

    print(f"Branch runs staged in: {out}")
    # print(f"Branch tags:\n{branch_tags}")
    if branch_tags_filtered is not None:
        with open(out.parent / f"{out.name.strip('.txt')}_filtered_N{args.num_points}.txt", "w") as f:
            f.write("\n".join(branch_tags_filtered) + "\n")
        print(f"Filtered branch tags saved at: {out.parent / f'{out.name.strip('.txt')}_filtered_N{args.num_points}.txt'}")
    if skipped:
        print(f"Skipped trunk tags (not complete):\n{skipped}")
# end of file