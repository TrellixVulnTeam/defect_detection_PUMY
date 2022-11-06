GPU_ID=$1
CHECKPOINT=${@:2}
CONFIG=tasks/detections/det_voc/det_atss_swinbv2_dyhead_voc.yaml
CHECKPOINT_ROOT=meta/train_infos/det_atss_swinbv2_dyhead_voc

EVAL_METRIC='map'

for cpt in $CHECKPOINT
do
PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH CUDA_VISIBLE_DEVICES=$GPU_ID python3 tools/test.py $CONFIG \
${CHECKPOINT_ROOT}/epoch_${cpt}.pth --eval $EVAL_METRIC
done