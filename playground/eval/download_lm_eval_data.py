"""Pre-download lm-eval task datasets into HF_HOME (run on a login node).

    HF_HOME=$wspace/hf_cache python playground/eval/download_lm_eval_data.py [task ...]

Then on compute nodes: export HF_HOME=... HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
"""

import sys

from lm_eval.tasks import TaskManager, get_task_dict

from lm_eval_megatron import TASKS

tasks = sys.argv[1:] or TASKS
for name, task in get_task_dict(tasks, TaskManager()).items():
    print(f"downloading {name} ...")
    task.download()
print("done")
