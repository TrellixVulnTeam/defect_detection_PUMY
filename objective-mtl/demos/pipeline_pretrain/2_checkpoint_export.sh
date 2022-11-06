#!/usr/bin/env bash

CHECKPOINT=meta/pretrained/swin_t_moco_epoch_50.pth
SAVE_PATH=meta/pretrained/swin_t_moco_ep50_washed.pth
PREFIX_1=backbone_q.
PREFIX_2=head_q.

PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
python3 tools/model_export/checkpoint_wash.py \
--checkpoint $CHECKPOINT --save_path $SAVE_PATH \
--keep_prefix_list $PREFIX_1 $PREFIX_2
