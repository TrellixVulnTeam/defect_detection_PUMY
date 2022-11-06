# 分类接口(cls)
mtl/classifications/*.py

# init_weights
  初始化权重，有预训练模型，通过此方法加载预训练权重
# forward
  前向推理函数
# extract_feat
  从backbone或者backbone和neck中获得前向推理的结果
# forward_train
  用于训练的前向推理代码，重构一个新的分类时候需要实现
# simple_test
  用于模型推理测试的代码，重构一个分类模型的时候需要实现
# get_losses
  计算losses，返回losses

#####
# 检测接口(det)
mtl/detections/*.py

# init_weights
  初始化权重，有预训练模型，通过此方法加载预训练权重
# forward_train
  用于训练的前向推理代码，重构一个新的分类时候需要实现
# simple_test
  用于模型推理测试的代码，重构一个分类模型的时候需要实现

######
# backbone接口
mtl/block/backbones/*.py

# init_weights
  初始化权重，有预训练模型，通过此方法加载预训练权重
# forward
  前向推理函数

######
# neck接口
mtl/block/necks/*.py

# forward
  前向推理函数

######
# head接口
mtl/block/rpn_heads/*.py
mtl/block/roi_heads/*.py

# init_weights
  初始化权重，有预训练模型，通过此方法加载预训练权重
# forward
  前向推理函数
# forward_train
  用于训练的前向推理代码，重构一个新的分类时候需要实现
# get_losses
  计算losses，返回losses
  # get_boxes
  计算boxes，返回boxes