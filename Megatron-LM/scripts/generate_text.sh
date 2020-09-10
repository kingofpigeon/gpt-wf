#!/bin/bash

CHECKPOINT_PATH=checkpoints/generic/
MPSIZE=1
NLAYERS=24
NHIDDEN=1024
NATT=16
MAXSEQLEN=4096

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=50
TOPP=0.95

python3 generate_samples.py \
       --model-parallel-size $MPSIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --load $CHECKPOINT_PATH \
       --no-load-rng \
       --num-attention-heads $NATT \
       --max-position-embeddings 1024 \
       --batch-size 1 \
       --tokenizer-type GPT2WordPieceTokenizer \
       --tokenizer-model-type gpt2-chinese-vocab.txt \
       --fp16 \
       --cache-dir cache \
       --out-seq-length $MAXSEQLEN \
       --temperature $TEMP \
       --top_k $TOPK \
       --num-samples 0 \
       --top_p $TOPP \
       --recompute
