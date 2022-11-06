#!/usr/bin/env bash
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_TOPO_FILE=/tmp/topo.txt

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
