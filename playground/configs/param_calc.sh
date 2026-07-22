#!/bin/bash

source $1

VOCAB_SIZE=50304

swiglu=3  # if no swiglu, set to 2  
tied=2  # if tied embeddings, set to 1; if untied, set to 2

PER_LAYER=$((4 * HIDDEN_SIZE * HIDDEN_SIZE + swiglu * HIDDEN_SIZE * FFN_HIDDEN_SIZE + 4 * HIDDEN_SIZE))

HIDDEN_PARAMS=$((NUM_LAYERS * PER_LAYER))

EMBED_PARAMS=$((tied * HIDDEN_SIZE * VOCAB_SIZE))

TOTAL_PARAMS=$((HIDDEN_PARAMS + EMBED_PARAMS))


# echo "Non_embed: $((HIDDEN_PARAMS / 10**9)) + Embed: $((EMBED_PARAMS / 10**9)) = $((TOTAL_PARAMS / 10**9)) B parameters"
echo "Hidden: $(echo "scale=4; $HIDDEN_PARAMS/10^9" | bc) B parameters"
echo "Embed: $(echo "scale=4; $EMBED_PARAMS/10^9" | bc) B parameters"
echo "Total: $(echo "scale=4; $TOTAL_PARAMS/10^9" | bc) B parameters"