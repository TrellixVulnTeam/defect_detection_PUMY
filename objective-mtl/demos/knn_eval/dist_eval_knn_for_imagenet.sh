#!/usr/bin/env bash

CONFIG=tasks/embeddings/emb_knn_vitb_imagenet.yaml
CHECKPOINT=meta/pretrained/emb_mtlir_vitb_tencentml_ep20.pth
GPUS=1
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS \
--master_port=$PORT tools/model_eval/eval_knn.py $CONFIG \
${CHECKPOINT} --nb_knn 20 --dump_path meta/test_infos \
--launcher pytorch
