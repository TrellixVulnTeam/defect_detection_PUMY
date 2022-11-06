#python train.py --workers 8 --device 0 --batch-size 22 --data data/floor_cut_640.yaml \
#      --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'yolov7.pt' \
#      --name floor_cut640_v7 --hyp data/hyp.scratch.p5.yaml

#python train.py --workers 8 --device 0 --batch-size 96 --data data/floor_cut_640.yaml \
#      --img 640 640 --cfg cfg/training/yolov7-tiny.yaml --weights 'yolov7.pt' \
#      --name floor_cut640_v7tiny --hyp data/hyp.scratch.tiny.yaml

#python train.py --workers 8 --device 0 --batch-size 96 --data data/floor_cut_640.yaml \
#      --img 640 640 --cfg cfg/training/yolov7-tiny.yaml --weights 'yolov7.pt' \
#      --name floor_cut640_v7tiny --hyp data/hyp.scratch.p5.yaml

#python train.py --workers 8 --device 0 --batch-size 16 --data data/floor_cut_640.yaml \
#      --img 640 640 --cfg cfg/training/yolov7x.yaml --weights 'yolov7x.pt' \
#      --name floor_cut640_v7x --hyp data/hyp.scratch.p5.yaml

#python train_aux.py --workers 8 --device 0 --batch-size 20 --data data/floor_cut_640.yaml \
#      --img 640 640 --cfg cfg/training/yolov7_aux.yaml --weights 'yolov7.pt' \
#      --name floor_cut640_v7aux --hyp data/hyp.scratch.p5.yaml

#python train_aux.py --workers 8 --device 0 --batch-size 96 --data data/floor_cut_640.yaml \
#      --img 640 640 --cfg cfg/training/yolov7-tiny_aux.yaml --weights 'yolov7.pt' \
#      --name floor_cut640_v7tinyaux --hyp data/hyp.scratch.tiny.yaml

#python train_aux.py --workers 8 --device 0 --batch-size 96 --data data/floor_cut_640.yaml \
#      --img 640 640 --cfg cfg/training/yolov7-tiny_aux.yaml --weights 'yolov7.pt' \
#      --name floor_cut640_v7tinyaux --hyp data/hyp.scratch.p5.yaml

python train_aux.py --workers 8 --device 0 --batch-size 14 --data data/floor_cut_640.yaml \
      --img 640 640 --cfg cfg/training/yolov7x_aux.yaml --weights 'yolov7x.pt' \
      --name floor_cut640_v7xaux --hyp data/hyp.scratch.p5.yaml