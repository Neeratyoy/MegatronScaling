import json
import pandas as pd
import re
from pathlib import Path
from typing import List
import argparse
import shutil


_ITER_RE = re.compile(r"^iter_(\d+)$")


def is_complete(path: Path) -> int | None:
    """Status from the checkpoint dir, comparing on-disk iters to target `train_iters`.

    Returns:
        1  -> complete (max iter == train_iters)
        0  -> pending (max iter < train_iters)
       -1  -> discrepancy (something on disk exceeds train_iters)
        None -> failed / not started
    """
    p = Path(path)
    if not p.exists():
        return None

    config_path = p / "run_config.json"
    if not config_path.exists():
        return None
    try:
        target = int(json.loads(config_path.read_text())["train_iters"])
    except Exception:
        return None

    # iteration dirs: iter_0000500, iter_0010173, ...
    iters = [
        int(m.group(1))
        for d in p.iterdir()
        if d.is_dir() and (m := re.match(r"^iter_(\d+)$", d.name))
    ]

    # pointer file, top level or under latest/
    latest = None
    for candidate in (p / "latest_checkpointed_iteration.txt",
                      p / "latest" / "latest_checkpointed_iteration.txt"):
        if candidate.exists():
            try:
                latest = int(candidate.read_text().strip())
            except ValueError:
                return None
            break

    if not iters and latest is None:
        return None

    observed = max(iters + ([latest] if latest is not None else []))

    if observed > target:
        return -1
    return 1 if observed == target else 0


def check_run_status_given_a_run_file(
    base_path: Path, run_file: Path, seed: int = 123456
) -> pd.DataFrame:
    # load the run file
    run_file = Path(run_file)
    if not run_file.exists():
        raise FileNotFoundError(f"Run file {run_file} does not exist.")
    with open(run_file, "r") as f:
        run_paths = [line.strip() for line in f.readlines() if line.strip()]

    base_path = Path(base_path)

    # creating status map
    path_map = {}
    for i, run_name in enumerate(run_paths):
        _run = base_path / run_name / f"seed={seed}"
        path_map[i] = {"path": _run, "status": is_complete(_run)}
    path_map = pd.DataFrame.from_dict(path_map, orient="index")
    
    return path_map


def collect_checkpoints(
    path: Path, 
    keep_only_last_checkpoint: bool = False,
    dry_run: bool = True
) -> tuple[dict[int, Path], list[Path]]:
    """Map iteration -> checkpoint dir for a run directory.

    Scans persistent checkpoints (path/iter_*) and non-persistent ones
    (path/latest/iter_*). On an iteration collision the persistent copy wins.

    With keep_only_last_checkpoint=True, deletes every checkpoint except the
    highest iteration found and any iteration named by a tracker file.
    """
    if is_complete(path) != 1:
        return {}, []

    ckpts: dict[int, Path] = {}
    for root in (path / "latest", path):          # path second so persistent wins
        if not root.is_dir():
            continue
        for d in root.iterdir():
            m = _ITER_RE.match(d.name)
            if m and d.is_dir() and not d.is_symlink():
                ckpts[int(m.group(1))] = d

    if not ckpts or not keep_only_last_checkpoint:
        return ckpts, []

    keep = {max(ckpts)}
    for tracker in (
        path / "latest_checkpointed_iteration.txt",
        path / "latest" / "latest_checkpointed_iteration.txt"
    ):
        if tracker.is_file():
            txt = tracker.read_text().strip()
            if txt.isdigit():
                keep.add(int(txt))

    to_delete = [d for it, d in sorted(ckpts.items()) if it not in keep]

    if not dry_run:
        for d in to_delete:
            shutil.rmtree(d)

    return {it: d for it, d in ckpts.items() if it in keep}, to_delete


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=Path, required=True, help="Base path where runs are stored")
    parser.add_argument("--run_file", type=Path, required=True, help="File containing list of runs to check")
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="List checkpoints that would be deleted from completed runs (NO DELETION)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="[WARNING] With --clean, actually DELETE them",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    df = check_run_status_given_a_run_file(args.base_path, args.run_file, args.seed)

    print()
    print("=== Run Status Summary ===")
    print(f"Number of runs: {len(df)}")
    print(f"Number of completed runs: {len(df[df['status'] == 1])}")
    print(f"Number of pending runs: {len(df[df['status'] == 0])}")
    print(f"Number of failed runs: {len(df[df['status'].isna()])}")
    print("==========================")

    print("Pending run indices (comma-separated):", end=" ")
    print(",".join(str(i) for i in df.index[df["status"] == 0]))
    print("Pending paths (comma-separated):")
    pending_runs = df[df["status"] == 0]
    pending_list = [f"{row['path']}" for i, row in pending_runs.iterrows()]
    print("\n".join(pending_list))
    print("==========================")
    
    print("Failed (or not started) run indices (comma-separated):", end=" ")
    print(",".join(str(i) for i in df.index[df["status"].isna()]))
    print("Failed paths (comma-separated):")
    failed_runs = df[df["status"].isna()]
    failed_list = [f"{row['path']}" for i, row in failed_runs.iterrows()]
    print("\n".join(failed_list))
    print("==========================")

    if args.clean:
        dry_run = not args.delete
        print("=== Checkpoint Cleanup ({}) ===".format("DRY RUN" if dry_run else "DELETING"))
        total = 0
        for i, row in df[df["status"] == 1].iterrows():
            kept, removed = collect_checkpoints(
                row["path"], keep_only_last_checkpoint=True, dry_run=dry_run
            )
            if not removed:
                continue
            total += len(removed)
            print(f"[{i}] {row['path']}")
            print(f"    keep:   {', '.join(str(d.name) for d in kept.values())}")
            for d in removed:
                print(f"    {'would remove' if dry_run else 'removed'}: {d}")
        print(f"Total checkpoint dirs {'to remove' if dry_run else 'removed'}: {total}")
        print("==========================")
# end of file