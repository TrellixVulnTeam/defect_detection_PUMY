#!/bin/bash
#传参测试脚本
echo "the process is `basename $0` -I was called as $0"
echo "the wst filename is : $1"
echo "the engine filename is : $2"
echo "the model type is : $2"
rm -rf build
mkdir build
cd build
cmake ..
make -j4
cd ..
./build/yolov5 -s $1 $2 $3
