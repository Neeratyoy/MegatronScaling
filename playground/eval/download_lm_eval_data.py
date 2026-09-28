"""Pre-download lm-eval task datasets into HF_HOME (run on a login node).

    HF_HOME=$wspace/hf_cache python playground/eval/download_lm_eval_data.py [task ...]

Then on compute nodes: export HF_HOME=... HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
"""

import sys

from lm_eval.tasks import TaskManager

from lm_eval_megatron import ALL_TASKS



def flatten(d):
    """Groups (e.g. mmlu) load as nested dicts; yield the leaf tasks."""
    for name, t in d.items():
        yield from flatten(t) if isinstance(t, dict) else [(name, t)]


tasks = sys.argv[1:] or ALL_TASKS
for name, task in flatten(TaskManager().load(tasks)):
    print(f"downloading {name} ...")
    task.download()
print("done")
