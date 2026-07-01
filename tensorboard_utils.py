import os
import pandas as pd
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def dump_tensorboard_to_parquet(
    tb_dir: Path, 
    out_path: Path = None, 
    out_name: str = "run_metrics.parquet"
) -> Path | None:
    """ Dump all collected TensorBoard scalar metrics to a Parquet file.

    NOTE: the passed iteration/steps is the index of the DataFrame, and 
    duplicates are resolved by keeping the last value for each step.
    """
    if out_path is None:
        out_path = tb_dir

    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()

    frames = []
    for tag in ea.Tags().get("scalars", []):
        ev = ea.Scalars(tag)
        df_tag = pd.DataFrame({"step": [e.step for e in ev], tag: [e.value for e in ev]})
        df_tag = df_tag.drop_duplicates("step", keep="last").set_index("step")
        frames.append(df_tag)

    if not frames:
        return None

    df = pd.concat(frames, axis=1).sort_index().reset_index()
    out_path = os.path.join(out_path, out_name)
    df.to_parquet(out_path, index=False)
    return out_path
# en of file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", 
        type=str, 
        help="Path to the TensorBoard log directory", 
        required=True
    )
    parser.add_argument(
        "--savedir", 
        type=str, 
        help="Path to save the output Parquet file",
        default=None
    )
    parser.add_argument(
        "--out_name", 
        type=str, 
        default="run_metrics.parquet", 
        help="Output Parquet file name"
    )
    args = parser.parse_args()

    if not args.savedir:
        args.savedir = args.dir

    out_path = dump_tensorboard_to_parquet(args.dir, args.savedir, args.out_name)
    if out_path:
        print(f"Metrics dumped to: {out_path}")
    else:
        print("No scalar metrics found in the specified TensorBoard directory.")
# end of file