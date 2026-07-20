#!/bin/bash

source ${HOME}/depth.sh

BASE_DIR=...  # TODO: set this; assumes 2-level of depth to get to the tensorboard logs

for d in ${BASE_DIR}/*/* ; do
    echo "Processing directory: $d"
    # call the Python script to process the TensorBoard logs in the directory
    python ${codedir}/tensorboard_utils.py \
        --dir $d
done
# end of file