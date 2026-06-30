#!/bin/bash

# setup directories
source ${HOME}/depth.sh
# conda deactivate

module load cuda/13.0

# setup environment from MegatronLM
# source ${wspace}/MegatronLM/.venv/bin/activate

#######################

# Model
NUM_LAYERS=8
HIDDEN_SIZE=512
NUM_ATTENTION_HEADS=8
SEQ_LENGTH=1024
 
# Training
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=16
TRAIN_ITERS=500

# LR schedule
LR=3e-4
LR_DECAY_STYLE=cosine        # cosine | linear | constant
LR_WARMUP_FRACTION=0.1
MIN_LR=3e-5
 
# Norm — edit to experiment
NORMALIZATION=LayerNorm       # LayerNorm | RMSNorm
 
# Logging & checkpointing
LOG_INTERVAL=10
SAVE_INTERVAL=100
EVAL_INTERVAL=100
EVAL_ITERS=5

#######################

CKPT_DIR="${wspace}/results/debug/gpt_synthetic"

# the run
# RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
# MASTER_ADDR=localhost MASTER_PORT=29500 \
# python ${codedir}pretrain_gpt.py \
torchrun --nproc_per_node=2 pretrain_gpt.py \
    --save              $CKPT_DIR \
    --num-layers        $NUM_LAYERS \
    --hidden-size       $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --seq-length        $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --micro-batch-size  $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters       $TRAIN_ITERS \
    --lr                $LR \
    --lr-decay-style    $LR_DECAY_STYLE \
    --lr-warmup-fraction $LR_WARMUP_FRACTION \
    --min-lr            $MIN_LR \
    --normalization     $NORMALIZATION \
    --tokenizer-type    NullTokenizer \
    --vocab-size        256 \
    --mock-data \
    --log-interval      $LOG_INTERVAL \
    --save-interval     $SAVE_INTERVAL \
    --eval-interval     $EVAL_INTERVAL \
    --eval-iters        $EVAL_ITERS \
    --tensorboard-dir   $CKPT_DIR \
    --log-throughput \
    --log-timers-to-tensorboard \
    --no-load-optim \
    --no-load-rng \
    --no-persist-layer-norm \
    --no-gradient-accumulation-fusion \
    --no-rope-fusion \
    --transformer-impl  local
# end of file