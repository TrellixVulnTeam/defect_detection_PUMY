#!/usr/bin/env bash

CHECKPOINT=meta/train_infos/cls_swint_fc_kd_xxx/epoch_xx.pth
SAVE_PATH=meta/train_infos/cls_swint_fc_kd_xxx/model_exported_final.pth
PREFIX_1=backbone.
PREFIX_2=head.

PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
python3 tools/model_export/checkpoint_wash.py \
--checkpoint $CHECKPOINT --save_path $SAVE_PATH \
--keep_prefix_list $PREFIX_1 $PREFIX_2 \
--keep_prefix
