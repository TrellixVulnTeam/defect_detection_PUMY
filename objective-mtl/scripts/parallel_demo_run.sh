CUDA_VISIBLE_DEVICES=0 nohup python3 -u demos/det_infer_xxx.py 0 100000 0 > nohup0.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 -u demos/det_infer_xxx.py 100000 200000 1 > nohup1.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 -u demos/det_infer_xxx.py 200000 300000 2 > nohup2.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 -u demos/det_infer_xxx.py 300000 400000 3 > nohup3.out 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python3 -u demos/det_infer_xxx.py 400000 500000 4 > nohup4.out 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python3 -u demos/det_infer_xxx.py 500000 600000 5 > nohup5.out 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python3 -u demos/det_infer_xxx.py 600000 700000 6 > nohup6.out 2>&1 &