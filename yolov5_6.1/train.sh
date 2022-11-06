#python train.py --weights yolov5l.pt --cfg models/yolov5l.yaml --name bottle_cut640_l --data data/bottle_cut_640.yaml --img-size 640 --save-period 20
#python train.py --weights yolov5m.pt --cfg models/yolov5m.yaml --name bottle_cut640_m --data data/bottle_cut_640.yaml --img-size 640 --save-period 20
#python train.py --weights yolov5s.pt --cfg models/yolov5s.yaml --name bottle_cut640_s --data data/bottle_cut_640.yaml --img-size 640 --save-period 20
#python train.py --weights yolov5n.pt --cfg models/yolov5n.yaml --name bottle_cut640_n --data data/bottle_cut_640.yaml --img-size 640 --save-period 20
#python train.py --weights yolov5x.pt --cfg models/yolov5x.yaml --name bottle_cut640_x --data data/bottle_cut_640.yaml --img-size 640 --save-period 20
#python train.py --weights yolov5l.pt --cfg models/yolov5l.yaml --name floor_cut480_l --data data/floor_cut_480.yaml --img-size 480 --save-period 20
#python train.py --weights yolov5m.pt --cfg models/yolov5m.yaml --name floor_cut480_m --data data/floor_cut_480.yaml --img-size 480 --save-period 20
#python train.py --weights yolov5s.pt --cfg models/yolov5s.yaml --name floor_cut480_s --data data/floor_cut_480.yaml --img-size 480 --save-period 20
#python train.py --weights yolov5n.pt --cfg models/yolov5n.yaml --name floor_cut480_n --data data/floor_cut_480.yaml --img-size 480 --save-period 20
#python train.py --weights yolov5x.pt --cfg models/yolov5x.yaml --name floor_cut480_x --data data/floor_cut_480.yaml --img-size 480 --save-period 20

python train.py --weights yolov5s.pt --cfg models/yolov5s.yaml --name floor_cut640_aug10_s --data data/floor_cut_aug10_640.yaml --img-size 640 --save-period 30
python train.py --weights yolov5m.pt --cfg models/yolov5m.yaml --name floor_cut640_aug10_m --data data/floor_cut_aug10_640.yaml --img-size 640 --save-period 30
python train.py --weights yolov5l.pt --cfg models/yolov5l.yaml --name floor_cut640_aug10_l --data data/floor_cut_aug10_640.yaml --img-size 640 --save-period 30
python train.py --weights yolov5x.pt --cfg models/yolov5x.yaml --name floor_cut640_aug10_x --data data/floor_cut_aug10_640.yaml --img-size 640 --save-period 30
python train.py --weights yolov5n.pt --cfg models/yolov5n.yaml --name floor_cut640_aug10_n --data data/floor_cut_aug10_640.yaml --img-size 640 --save-period 30


