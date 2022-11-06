#!/usr/bin/env bash

CONFIG=tasks/classifications/cls_swinl_fc_clarity.yaml
NNODES=1
GPUS=4
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nnodes=$NNODES --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py --no-test --launcher pytorch ${@:1} $CONFIG 
