#!/usr/bin/env bash

CHECKPOINT=meta/train_infos/det_atss_swinbv2_dyhead_voc/epoch_xx.pth
SAVE_PATH=meta/pretrained/det_atss_swinbv2_dyhead_voc/model_exported_final.pth
PREFIX_1=backbone.
PREFIX_2=neck.
PREFIX_3=head.

PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
python3 tools/model_export/checkpoint_wash.py \
--checkpoint $CHECKPOINT --save_path $SAVE_PATH \
--keep_prefix_list $PREFIX_1 $PREFIX_2 $PREFIX_3\
--keep_prefix
