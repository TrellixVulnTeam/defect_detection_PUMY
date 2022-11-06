# 模型训练与评估
  模型训练与推理引擎主要包含了三个脚本函数，即训练器trainer、预测器predictor和评估器evaluator。

## 训练器trainer
  Trainer中包含了三个主要函数：
  - set_random_seed用于设置numpy、torch、torch.cuda的随机种子，deterministic参数可用于设置torch.backends.cudnn.deterministic，用以决定是否让cuDNN中的auto-tuner自动优化。
  - get_builtin_config_dict用于将CfgNode类型的配置转化为dict类型。
  - train_processor 根据配置文件、模型、数据集等开展模型训练。计算过程如下：获取日志模型部署（GPU需确定是否采用分布式训练）构建优化器构建运行器运行器设置与各类HOOK注册训练数据集加载器验证

## 预测器predictor
  Predictor中包含了三个主要函数：
  - get_predictor: 获取推理模型
  - inference_predictor: 给定图像，获取推理结果
  - show_predictor_result: 显示推理结果

## 评估器evaluator
  评估器根据模型类型分别进行评估，一般分单机评估和多机评估两个版本:
  - single_device_test: 根据模型类型，调用对应的单机评估函数
  - multi_device_test: 根据模型类型，调用对应的多机评估函数
