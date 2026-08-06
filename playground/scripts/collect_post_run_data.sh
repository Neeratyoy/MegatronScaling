#!/bin/bash

source ${HOME}/depth.sh

BASE_DIR=$1

NPROC=${2:-4}  # for parallelism

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ARROW_NUM_THREADS=1
export MALLOC_CONF="background_thread:false"

find "${BASE_DIR}" -mindepth 2 -maxdepth 2 -type d -print0 \
    | xargs -0 -P "${NPROC}" -n1 \
      python "${codedir}/tensorboard_utils.py" --on-complete --dir
# end of file