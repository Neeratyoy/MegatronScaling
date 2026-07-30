import argparse
from pathlib import Path
import sys
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from playground.scripts.check_run_status import is_complete


def stage_branches(trunk_dir: Path, seed: int = 123456) -> list[str]:
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
        if run_dir.exists():
            continue
        run_dir.mkdir(parents=True)
        (run_dir / ckpt.name).symlink_to(ckpt.absolute())
        iter_num = int(ckpt.name.split("_")[1])
        (run_dir / "latest_checkpointed_iteration.txt").write_text(f"{iter_num}\n")

    return tags


def stage_grid(base_path: Path, run_file: Path, seed: int = 123456) -> tuple[Path, list[str], list[str]]:
    base_path = Path(base_path)
    run_file = Path(run_file)
    trunk_tags = [l.strip() for l in run_file.read_text().splitlines() if l.strip()]

    branch_tags, skipped = [], []
    for tag in trunk_tags:
        trunk = base_path / tag / f"seed={seed}"
        if is_complete(trunk) != 1:
            skipped.append(tag)
            continue
        branch_tags.extend(stage_branches(trunk, seed))

    out = run_file.parent / f"branch_runs_{base_path.name}.txt"
    out.write_text("\n".join(branch_tags) + "\n")
    return out, branch_tags, skipped


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
    parser.add_argument("--seed", type=int, default=123456)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    out, branch_tags, skipped = stage_grid(args.base_path, args.run_file, args.seed)
    print(f"Branch runs staged in: {out}")
    print(f"Branch tags:\n{branch_tags}")
    if skipped:
        print(f"Skipped trunk tags (not complete):\n{skipped}")
# end of file