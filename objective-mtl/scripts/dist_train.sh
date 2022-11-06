#!/usr/bin/env bash

CONFIG=$1
NNODES=$2
GPUS=$3
PORT=${PORT:-29500}
GPU_IDS=
SPLIT=", "

for i in $(seq 0 $GPUS):
do
   GPU_IDS=${GPU_IDS}${i}${SPLIT};
done

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$GPU_IDS \
torchrun --nnodes=$NNODES --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py --launcher pytorch ${@:4} $CONFIG 

# python3 -m torch.distributed.launch  --no-test