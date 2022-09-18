#!/bin/bash
#传参测试脚本
echo "the process is `basename $0` -I was called as $0"
echo "the engine filename is : $1"
echo "the test image file is : $2"
./build/yolov5 -d $1 $2

