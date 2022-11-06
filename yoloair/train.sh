#python train.py --weights yolov7.pt --batch-size 56 --cfg configs/used/yolov7-RepVGG.yaml --name floor_cut_512_v7RepVGG --data data/floor_cut_480.yaml --img 512 --save-period 30
#python train.py --weights yolov7.pt --batch-size 40 --cfg configs/used/yolov7-RepVGG.yaml --name floor_cut_640_v7RepVGG --data data/floor_cut_640.yaml --img 640 --save-period 30
python train.py --weights yolov7.pt --batch-size 26 --cfg configs/used/yolov7-RepVGG.yaml --name floor_cut_768_v7RepVGG --data data/floor_cut_768.yaml --img 768 --save-period 30
