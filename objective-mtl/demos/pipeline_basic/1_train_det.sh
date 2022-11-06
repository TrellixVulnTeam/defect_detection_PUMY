#!/usr/bin/env bash

CONFIG=tasks/detections/det_voc/det_atss_swinbv2_dyhead_voc.yaml
NNODES=1
GPUS=1
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nnodes=$NNODES --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py --no-test --launcher pytorch ${@:1} $CONFIG 
