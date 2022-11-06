#!/usr/bin/env bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

CONFIG=$1
NNODES=$2
NGPUS=$3
NRANK=$4
MASTER_ADDR=$5
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nnodes=$NNODES \
         --nproc_per_node=$NGPUS \
         --node_rank=$NRANK \
         --master_addr=$MASTER_ADDR \
         --master_port=$PORT \
         tools/train.py --no-test \
         --launcher pytorch ${@:6} $CONFIG 
