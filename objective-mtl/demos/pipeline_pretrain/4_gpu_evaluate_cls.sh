#!/usr/bin/env bash

GPU_ID=$1
CHECKPOINT=${@:2}
CONFIG=tasks/classifications/cls_vitb_imagenet.yaml
CHECKPOINT_ROOT=meta/train_infos/cls_vitb_imagenet

EVAL_METRIC_1='accuracy'
# EVAL_METRIC_2='precision'
# EVAL_METRIC_3='recall'
# EVAL_METRIC_4='f1_score'

for cpt in $CHECKPOINT
do
PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH CUDA_VISIBLE_DEVICES=$GPU_ID python3 tools/test.py $CONFIG \
${CHECKPOINT_ROOT}/epoch_${cpt}.pth --eval $EVAL_METRIC_1
done
