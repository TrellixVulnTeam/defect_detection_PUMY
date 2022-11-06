#python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py \
#        --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/coco.yaml \
#        --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
#
#python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train_aux.py \
#        --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch-size 128 --data data/coco.yaml \
#        --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml