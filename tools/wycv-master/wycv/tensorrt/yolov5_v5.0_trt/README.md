**step1**: create docker container

```
download xuchao_grpc_trt_20210629.tar 
docker load -i xuchao_grpc_trt_20210629.tar 
docker run --gpus all --shm-size="10g" -p xxxx:22 --name="xxxxxxx" -v /your code path:/workspace --restart=always -itd grpc_trt:v1.0
docker exec -it xxxxxxx  bash
```

**step2**:generate .wts from pytorch with .pt

```
cp wycv/trt/yolov5_v5.0_trt/gen_wts.py  {your code path}/yolov5-v5.0
python gen_wts.py -w yolov5s.pt -o yolov5s.wts
// a file '.wts' will be generated.
```

**step3**:generate .engine and .so  from .wts

1.modify **Input shape** defined in yololayer.h
2.modify **number of classes** defined in yololayer.h

```
cd wycv/trt/yolov5_v5.0_trt
sh wts2trt.sh [.wts] [.engine] [s/m/l/x/s6/m6/l6/x6 or c/c6 gd gw]
// a file '.engine' and 'build/libmyplugins.so' will be generated.
```

**step4:**test  .engine

```
sh trt_test.sh [.engine] [image file/folder] 
```

