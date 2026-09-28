"""Pre-download lm-eval task datasets into HF_HOME (run on a login node).

    HF_HOME=$wspace/hf_cache python playground/eval/download_lm_eval_data.py [task ...]

Then on compute nodes: export HF_HOME=... HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
"""

import sys

from lm_eval.tasks import TaskManager

from lm_eval_megatron import ALL_TASKS

tasks = sys.argv[1:] or ALL_TASKS
for name, task in TaskManager().load(tasks).items():
    print(f"downloading {name} ...")
    task.download()
print("done")
