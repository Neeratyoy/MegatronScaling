#!/bin/bash

# setup directories
source ${HOME}/depth.sh
# conda deactivate

module load cuda/13.0
#######################

## expects to source :
## `NUM_LAYERS`, 
## `HIDDEN_SIZE`, 
## `FFN_HIDDEN_SIZE`,
## `NUM_ATTENTION_HEADS`
## `SEQ_LENGTH`
## `MICRO_BATCH_SIZE`
source "${1}"  # e.g. playground/config/model_size/base.info

VOCAB_SIZE=52000

# Training
# MICRO_BATCH_SIZE=4
# GLOBAL_BATCH_SIZE=$MICRO_BATCH_SIZE
GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIZE * 2))  # no gradient accumulation
TRAIN_ITERS=10

# LR schedule
LR=3e-4
LR_WARMUP_FRACTION=0.1
LR_DECAY_STYLE=WSD        # cosine | linear | constant
LR_DECAY_ITERS=100
MIN_LR=3e-5
LR_WSD_DECAY_STYLE=linear
 
# Norm — edit to experiment
NORMALIZATION=LayerNorm       # LayerNorm | RMSNorm
 
# Logging & checkpointing
LOG_INTERVAL=2
SAVE_INTERVAL=10
EVAL_INTERVAL=10
EVAL_ITERS=10

#######################

CKPT_DIR="${wspace}/results/model_size_check/$(basename $1)"

# the run
RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
MASTER_ADDR=localhost MASTER_PORT=29500 \
python ${codedir}/pretrain_gpt.py \
    --save                    $CKPT_DIR \
    --num-layers              $NUM_LAYERS \
    --hidden-size             $HIDDEN_SIZE \
    --num-attention-heads     $NUM_ATTENTION_HEADS \
    --ffn-hidden-size         $FFN_HIDDEN_SIZE \
    --seq-length              $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --swiglu \
    --micro-batch-size        $MICRO_BATCH_SIZE \
    --global-batch-size       $GLOBAL_BATCH_SIZE \
    --train-iters             $TRAIN_ITERS \
    --lr                      $LR \
    --lr-warmup-fraction      $LR_WARMUP_FRACTION \
    --lr-decay-style          $LR_DECAY_STYLE \
    --lr-wsd-decay-iters      $LR_DECAY_ITERS \
    --lr-wsd-decay-style      $LR_WSD_DECAY_STYLE \
    --min-lr                  $MIN_LR \
    --weight-decay            $WEIGHT_DECAY \
    --normalization           $NORMALIZATION \
    --qk-layernorm \
    --attention-dropout 0.0 \
    --hidden-dropout    0.0 \
    --tokenizer-type          NullTokenizer \
    --vocab-size              $VOCAB_SIZE \
    --rotary-base             $ROTARY_BASE \
    --rotary-percent          $ROTARY_PERCENT \
    --position-embedding-type rope \
    --log-interval            $LOG_INTERVAL \
    --save-interval           $SAVE_INTERVAL \
    --eval-interval           $EVAL_INTERVAL \
    --eval-iters              $EVAL_ITERS \
    --tensorboard-dir         $CKPT_DIR \
    --log-throughput \
    --log-timers-to-tensorboard \
    --no-load-optim \
    --no-load-rng \
    --use-distributed-optimizer \
    --overlap-param-gather \
    --overlap-grad-reduce \
    --use-flash-attn \
    --bf16 \
    --recompute-activations \
    --mock-data
# end of file