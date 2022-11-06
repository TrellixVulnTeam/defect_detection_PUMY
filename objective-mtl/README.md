
<!-- TOC -->

  - [Introduction](#introduction)
    - [主要特征](#主要特征)
    - [环境配置](#环境配置)
    - [开发](#开发)
  - [Get Started](#get-started)
    - [模型训练](#模型训练)
        - [单机本地训练](#单机本地训练)
        - [单机多卡或多机多卡训练](#单机多卡或多机多卡训练)
    - [模型测试与评估](#模型测试与评估)
        - [数据集评估](#数据集评估)
        - [单例测试](#单例测试)
        - [Onnx模型测试](#Onnx模型测试)
    - [训练与测试全流程Demo](#训练与测试全流程Demo)
        - [训练任务基本流程](#训练任务基本流程)
        - [自监督预训练与下游任务流程](#自监督预训练与下游任务流程)
    - [开发示例](#开发示例)
  - [Tools](#tools)
    - [模型导出](#模型导出)
        - [转换保存checkpoint中模型权重的key名称](#转换保存checkpoint中模型权重的key名称)
        - [显示模型权重](#显示模型权重)
        - [更改模型权重中参数名称](#更改模型权重中参数名称)
        - [清洗掉模型无用前缀和无用参数](#清洗掉模型无用前缀和无用参数)
        - [ONNX模型导出](#ONNX模型导出)
    - [数据转换](#数据转换)
    - [数据可视化与评估](#数据可视化与评估)
    - [VSCode调试](#VSCode调试)
  - [FAQ](#FAQ)
  - [TODO](#TODO)

<!-- TOC -->


# Introduction
  ObjectiveMTL是一款面向图像结构化任务的工具箱. [框架说明](docs/0_doc_description.md)

## 主要特征
  - **支持多种视觉任务，包括分类、检测、分割、位姿估计等**
  - **高性能高精度**
  - **支持多种数据集**
  - **精心设计的结构和良好文档**


## 环境配置
  - Linux or macOS
  - Python 3.6+
  - PyTorch 1.5+ (torch>=1.7.1, torchvision>=0.8.2)
  - CUDA 9.2+
  - GCC 5+
  - and so on (seen in requirements)

### Pip安装
  - pip3 install -r requirements.txt
  - 需要用到deformable-detr的，可以在mtl/cores/ops/ms_deform_attn目录下通过make.sh进行安装
  - pytorch扩展库安装需要GCC版本在7.0以上，可通过‘scl enable devtoolset-7 bash’进行快速安装

### Docker配置
  docker pull mirrors.tencent.com/tkd_cpc_cv/py36-torch16-mtl:latest

## 开发
  在Bash环境中添加`PYTHONPATH`路径
  ```shell
  export PYTHONPATH=$(pwd):$PYTHONPATH
  ```
  或者在运行脚本里添加路径
  ```shell
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
  ```


# Get Started

## 模型训练
  ```shell
  python3 tools/train.py [optional arguments] ${task_yaml_file}
  or
  python3 -m torch.distributed.launch --nproc_per_node=${GPU} --master_port=${PORT}  tools/train.py --load-from ${CHECKPOINT} --no-test --launcher pytorch ${CONFIG} 
  or
  ./scripts/dist_train.sh ${CONFIG} ${GPU} ${PORT} 
  ```
  参数说明:
  - `task_yaml_file`: 训练配置文件.
  - `--no-test` (**强烈推荐**): 是否在训练过程中测试当前checkpoint.
  - `--work-dir`: 重载配置文件中的当前工作目录.
  - `--gpus`: GPU数量，用于单机训练.
  - `--gpu-ids`: GPU IDs，用于单机训练.
  - `--seed`: python, numpy和pytorch的随机数种子.
  - `--deterministic`: 如果申明，则使用CUDNN backend.
  - `--launcher`: 分布式训练的launcher. 选择项包括`none`和`pytorch`. 
  - `--local_rank`: ID for local rank. 如果没有申明, 则设置为0.
  - `--opts`: 如果申明，则用于修改配置项内容.
  - `--resume-from` 继续训练所用的checkpoint文件, 包含训练epoch信息.
  - `--load-from` 载入模型权重，并重新开始训练，用于模型微调.

### 单机本地训练
  ```shell
  python3 ./tools/train.py --work-dir meta/train_infos --no-test tasks/classifications/xxx.yaml (ops)
  bash scripts/local_train.sh tasks/classifications/xxx.yaml
  ```

### 单机多卡或多机多卡训练
  ```shell
  (CUDA_VISIBLE_DEVICES=0,1,2,3) torchrun ./tools/train.py --work-dir ../train_moby_infos --no-test --launcher pytorch tasks/detections/xxx.yaml (ops)
  bash scripts/dist_train.sh tasks/classifications/xxx.yaml 4
  ```

## 模型测试与评估
  这里提供了各种脚步进行模型测试与评估.

### 数据集评估
  ```shell
  python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
  ```
  参数说明:
  - `task_config`: 测试任务的配置文件.
  - `checkpoint`: Checkpoint文件.
  - `--out`: 输出结果所保存的文件路径. 如果不申明, 则不保存.
  - `--fuse-conv-bn`: 是否融合卷积和归一化层. 如果融合, 能够提高推理速度.
  - `--format-only`: 格式化输出但不执行评估. 
  - `--eval`: 评估指标, e.g., "bbox", "segm", "proposal" for COCO, and "map", "recall" for PASCAL VOC.
  - `--show`: 是否可视化
  - `--show-dir`: 可视化图片的保存路径.
  - `--show-score-thr`: Score阈值 (缺省为0.3).
  - `--gpu-collect`: 是否使用GPU来收集结果.
  - `--tmpdir`: 临时文件夹，用于收集多个worker的结果.
  - `--launcher`: 分布式训练的launcher. 选择项包括`none`和`pytorch`.
  - `--local_rank`: ID for local rank. 如果没有申明, 则设置为0.
  1. 分类   
  ```shell
    python3 tools/test.py tasks/classifications/cls_xxx.yaml meta/train_infos/cls_xxx/epoch_xx.pth --eval 'accuracy' --show-dir 'meta/test_infos/cls_xxx'
  ```   
  2. 检测
  ```shell
    python3 ./tools/test.py tasks/detections/det_xxx.yaml  meta/train_infos/det_xxx/epoch_xx.pth --eval 'map' --out 'meta/test_infos/det_xxx.pkl' --show-dir 'meta/test_infos/det_xxx'
  ```
  3. 分割
  ```shell
    python3 tools/test.py tasks/segmentations/seg_xxx.yaml meta/train_infos/seg_xxx/epoch_xx.pth --eval 'mIoU' --show-dir 'meta/test_infos/seg_xxx'
  ```

### 单例测试
  1. 图像分类
  ```shell
    python3 demos/cls_infer_demos/cls_infer_to_txt.py
  ```
  2. 目标检测
  ```shell
    python3 demos/det_infer_demos/det_predictor_test.py
  ```

### Onnx模型测试
  1. 图像分类
  ```shell
    python3 demos/onnx_test/cls_onnx_test.py
  ```
  2. 目标检测
  ```shell
    python3 demos/onnx_test/det_onnx_test.py
  ```

## 训练与测试全流程Demo

### 训练任务基本流程
```参考demos/pipeline_basic```
1. 数据打包(ops, bash demos/pipeline_basic/0_generate_dataset.sh, 数据集组织格式见bash脚本)
2. 模型训练(bash demos/pipeline_basic/1_train_det.sh)
3. 模型评估(bash demos/pipeline_basic/2_gpu_evaluate_det.sh)
4. 模型导出(bash demos/pipeline_basic/3_model_export.sh)

### 自监督预训练与下游任务流程
```应用要求：预训练模型backbone与下游任务一致```
```参考demos/pipeline_pretrain```
1. 生成config文件(ops, bash demos/pipeline_pretrain/0_generate_config.sh {data_type} {data_name} {template_file} {output_file}
2. 通用大规模数据集自监督预训练(bash demos/pipeline_pretrain/1_train_ssl.sh)
3. 预训练模型参数清洗与导出(bash demos/pipeline_pretrain/2_checkpoint_export.sh)
4. 下游任务微调(bash demos/pipeline_pretrain/3_train_cls.sh)
5. 模型评估(bash demos/pipeline_pretrain/4_gpu_evaluate_cls.sh)
6. 下游任务模型导出(demos/pipeline_pretrain/5_model_export.sh, 丢弃与推理无关的模型参数，包括教师网络和优化器参数等)

## 开发示例
  ```这里列举了一些开发示例```
  - [设置配置文件](docs/1_configure_config.md)
  - [增加新的数据集](docs/2_new_dataset.md)
  - [增加数据增强操作](docs/3_new_transform.md)
  - [增加新的模型](docs/4_new_model.md)
  - [增加新的模型](docs/5_train_and_eval.md)
  - [其他扩展项](docs/6_other_extension.md)


# Tools

## 模型导出

### 转换保存checkpoint中模型权重的key名称
  ```shell
  python3 tools/model_export/checkpoint_change_key.py input_model_path output_model_path
  ```

### 显示模型权重
  ```shell
  python3 tools/model_export/checkpoint_show.py input_model_path
  ```

### 更改模型权重中参数名称
  ```shell
  python3 tools/model_export/checkpoint_rename_params.py xxx xxx
  ```

### 清洗掉模型无用前缀和无用参数
  ```shell
  python3 tools/model_export/checkpoint_wash.py xxx xxx
  ```

### ONNX模型导出
  1. 图像分类
  ```shell   
    python3 tools/model_export/cls_pytorch2onnx.py tasks/classifications/cls_xxx.yaml  meta/train_infos/cls_xxx/epoch_xx.pth --input-img meta/test_data/a0519qvbyom_001.jpg --show --output-file meta/onnx_models/cls_xxx.onnx --opset-version 11 --verify --resize_shape 224 224
  ```
  2. 目标检测
  ```单阶段模型ONNX导出时的命名规范```
  ```shell
    input_names=['INPUT__0'],
    output_names=['OUTPUT__0'],
    dynamic_axes={
        'INPUT__0': {0: 'batch', 2: 'height', 3: 'width'},
        'OUTPUT__0': {0: 'batch', 1: 'anchors'}
    }
  ```
  ```ONNX导出示例```
  ```shell
    python3 tools/model_export/det_onestage_pytorch2onnx.py tasks/detections/det_xxxj.yaml  meta/train_infos/det_xxx/epoch_xx.pth --input-img meta/test_data/a0519qvbyom_001.jpg --show --output-file meta/onnx_models/det_xxx.onnx --opset-version 11 --verify --shape 416 416 --mean 0 0 0 --std 255 255 255
  ```
  3. 图像分割
  ```shell   
    python3 tools/model_export/seg_pytorch2onnx.py tasks/segmentations/seg_xxx.yaml  meta/train_infos/seg_xxx/epoch_xx.pth --input-img meta/test_data/4.jpg --show --output-file meta/onnx_models/seg_xxx.onnx --opset-version 11 --verify --resize_shape 512 512
  ```
  
## 数据转换
  1. 图像分类数据转换
  ```shell
  python3 tools/data_convert/tfrecord_generate.py --dataset_path 'data/objcls-datasets/xxx' --dataset_type 'cls' --image_dir_name 'images' --split_dir_name 'meta' --record_path 'tfrecords' --is_file_ext True
  ```
  2. YOLO格式的目标检测数据转换
  ```shell
  python3 tools/data_convert/tfrecord_generate.py --dataset_path 'data/objdet-datasets/xxx' --dataset_type 'det' --image_dir_name 'images' --label_dir_name 'annotations' --split_dir_name 'metas' --record_path 'tfrecords' --label_format 'yolo' --split_str ' '
  ```
  3. 文本格式的目标检测数据转换
  ```shell
  python3 tools/data_convert/tfrecord_generate.py --dataset_path 'data/objdet-datasets/xxx' --dataset_type 'det' --image_dir_name 'images' --label_dir_name 'annotations' --split_dir_name 'metas' --record_path 'tfrecords' --split_str ' '
  ```
  4. VOC格式的目标检测数据转换
  ```shell
  python3 tools/data_convert/tfrecord_generate.py --dataset_path 'data/objdet-datasets/xxx' --dataset_type 'mtl' --image_dir_name 'images' --label_dir_name 'annotations' --split_dir_name 'meta' --record_path 'ori-tfrecords' --label_format 'voc'
  ```
  5. 图像分割数据转换
  ```shell
  python3 tools/data_convert/tfrecord_generate.py --dataset_path 'data/other-datasets/xxx' --dataset_type 'seg' --image_dir_name 'images' --label_dir_name 'annotations' --split_dir_name 'metas' --record_path 'tfrecords'
  ```

## 数据可视化与评估
  ```shell
  python3 tools/visualization/browse_dataset.py
  ```

## VSCode调试
  ```在Debug窗口执行```
  ```shell
  export PYTHONPATH=$(pwd):$PYTHONPATH
  ```


# FAQ
  ```这里罗列了一些开发过程中的常见问题```
  1. 训练时报错————TypeError: cannot pickle '_thread._local' object
  A: 将'WORKERS_PER_DEVICE'设置为0
  2. 训练时报错————TypeError: cannot serialize '_io.BufferedReader' object
  A: 检查'DATA.TRAIN_DATA.TYPE'是否为'Balanced'，若是，则改为'Concat'或'Normal'; 这个错误在最新版本已经解决，支持Balanced模式
  3. 中文文字显示问题
  A: export LANG=zh_CN.UTF-8
  4. 训练数据集超过一亿样本后出现CPU内存溢出问题。
  A：设置worker数为1，将DistributedGroupSampler改为DistributedSampler。
  5. _error_if_any_worker_fails() RuntimeError: DataLoader worker (pid xxxx) is killed by signal: Killed.
  A: 将batch_size改小一点
  6. 使用A100的显卡进行训练时，优化器报错UnboundLocalError: local variable 'beta1' referenced before assignment
  A: 这是torch-1.8.0版本的一个错误，将torch升级到1.9.0和torchvision升级到0.10.0
  7. RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. 
  A: using contiguous() for permute and reshape ops, or set `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`
  8. Memory Leak
  A: 训练时设置'--deterministic'，CUDNN和上采样的兼容性不好
  9. 训练报错 from . import pypocketfft as pfft
  ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.22' not found (required by /root/anaconda3/lib/python3.7/site-packages/scipy/fft/_pocketfft/pypocketfft.cpython-37m-x86_64-linux-gnu.so)
  A: scipy版本不匹配 pip install scipy==1.3.3
  10. TypeError: meshgrid() got an unexpected keyword argument 'indexing'
  A: pytorch的版本大于1.9，不支持indexing参数，可以在调用中删除这个参数
  11. [libprotobuf FATAL google/protobuf/stubs/common.cc:83] This program was compiled against version 3.9.2 of the Protocol Buffer runtime library, which is not compatible with the installed version (3.19.4).
  A:  pip install protobuf==3.9.2
  12. models/detections/base_detectors/base_detector.py", line 116 失败 
  A:  注释后，正常运行
  13. 使用torch.1.12.1转onnx时，atss模型转换成功但推理不成功
  A： DeltaXYWHBBoxCoder中的expand_as函数未能正确转换

# TODO
  1. 转onnx的输入可能存在bug。
