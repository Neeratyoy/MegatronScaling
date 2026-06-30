import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def dump_tensorboard_to_parquet(tb_dir, out_name="metrics.parquet") -> str | None:

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
    out_path = os.path.join(tb_dir, out_name)
    df.to_parquet(out_path, index=False)
    return out_path
# en of file


