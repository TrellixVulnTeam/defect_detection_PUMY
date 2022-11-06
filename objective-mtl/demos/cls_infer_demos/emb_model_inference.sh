CONFIG=tasks/embeddings/emb_moco_swinl_imagenet.yaml
CHECKPOINT=meta/train_infos/emb_moco_swinl_imagenet/epoch_x.pth
IMG_DIR=meta/test_data

PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH python3 \
demos/model_inference/emb_model_inference.py \
$CONFIG $CHECKPOINT --img_dir $IMG_DIR --device cpu
