#!/usr/bin/env bash

CHECKPOINT=meta/pretrained/scratch_imagenet1k_swinv3_tiny_window7_224_bs4096_ep300.pth
SAVE_PATH=meta/pretrained/scratch_imagenet1k_swinv3_tiny_window7_224_bs4096_ep300.pth
PREFIX_1=backbone.

PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
python3 tools/model_export/checkpoint_wash.py \
--checkpoint $CHECKPOINT --save_path $SAVE_PATH \
--keep_prefix_list $PREFIX_1
