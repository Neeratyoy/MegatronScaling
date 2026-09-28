#!/bin/bash
# Pre-download lm-eval task datasets on a login node.
# Usage: bash playground/scripts/download_lm_eval_data.sh [task ...]   (default: TASKS in lm_eval_megatron.py)

source ${HOME}/depth.sh

export HF_HOME="${HF_HOME:-${wspace}/hf_cache}"
mkdir -p "${HF_HOME}"

echo "HF_HOME=${HF_HOME}"
python ${codedir}/playground/eval/download_lm_eval_data.py "$@"
# end of file
