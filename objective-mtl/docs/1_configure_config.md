# 配置说明
本系统的配置文件分两类，任务配置（tasks）和管道配置（configs）。

## 任务配置
  任务配置由四类管道(Pipeline)配置组成，包括数据(DATA)、模型(MODEL)、训练策略(RUNTIME)和运行时设置(SCHEDULE)。
  对于每一个新的项目，可以发起一个任务配置。不同的任务配置分类放置在{root_mtl}/tasks/{task_type}中，其中task_type包括classifications、detections、embeddings、 pose_estimations、multitasks、regressions、segmentations等。
  任务配置一般通过引用加载现有管道配置，并通过重载的方式更新管道配置。任务配置则通过配置项依次调用管道配置各模块，并可通过层次结构关系更新相关管道配置内容，最终生成一个面向任务的CfgNode类（模版见{root_mtl}/configs/defaults.py）。该类主要定义了模型配置的各个主要节点，主要节点名称与数据、模型、运行时设置和训练策略配置文件中的节点名称一致。
  配置文件通过YAML格式组织，所用解析包为yacs，整个配置由CfgNode组成，其属性为类似字典的键值对。参数键值对属性值可以为常用数据类型（如Int、Float、str等）或列表，也可以为字典。当某一个CfgNode的节点属性值为字典时，相当于根据字典创建子节点。值得注意的是，YAML的节点属性为tuple时，不会自动解析，其值为str类型，可用列表代替或在使用时通过eval进一步解析。
  在该配置系统中，动态扩展的节点将父节点值用字典表示，固定节点则在缺省配置中直接定义。
  任务配置文件按类别进行划分。一般性的任务只需要根据现有管道配置进行组合与参数更新即可实现。若要扩展，则需实现对应的数据、模型、训练策略和运行时设置。

## 任务配置文件示例与说明
  ```YAML
  MODEL: 模型节点
    BASE: "" 模型配置文件相对路径
  DATA: 数据节点
    BASE: "" 数据配置文件相对路径
  SCHEDULE: 训练策略
    BASE: "" 训练策略配置文件相对路径
    OPTIMIZER: 优化器子节点
      lr: 0.001 学习率更新
  RUNTIME: 运行时设置
    BASE: “” 运行时设置文件相对路径
  ```
  注：目前参数结构关系与名称必须与现有设置一样，否则会视为新增节点和属性。

### 任务配置
  具体命名格式：{任务标识}_{模型名称}_{模型补充说明}(optional)_{数据名称}*N_{数据补充说明}(optional)
  任务标识说明：
  - cls：图像分类
  - det：目标检测
  - emb：图像Embedding
  - mtl：多任务学习
  - pos：位姿估计
  - seg：图像分割
  模型名称说明：
  - 指各类模型的名称
  模型补充说明：
  - 指模型版本号、Backbone名称、Neck名称、Head名称或以上名称的连接组合
  - 该字段仅由小写字母+数字组成，不能包含下划线
  数据名称说明：
  - 指各类数据集的名称
  数据补充说明：
  - 指数据增强、补充字段或以上名称的组合
  - 该字段仅由小写字母+数字组成，不能包含下划线

## 管道配置
管道配置又分四类：数据、模型、训练策略和运行时设置。

### 数据配置
  具体命名格式：{任务标识}_{数据集名称}*N_{数据增强标识}_{补充说明}
  任务标识与任务配置中一样
  数据集名称说明：
  - 单个数据集名称由小写字母+数字组成，例如coco，imagenet，tencentml等等
  - 多个数据集名称由下划线连接起来
  数据增强标识说明：
  - 数据增强可以是具体的增强方式，如randaug，mosaic，autoaug等等
  - 数据增强也可以是面向特定方法的增强方式，并以方法名为增强标识，如yolo，dino，mae等等
  补充说明：
  - 数据增强或方法的特定参数说明，如patch16

#### 数据配置文件示例与说明
  ```YAML
  DATA: 数据节点
    NAME: "" 数据接口类名称，用于从数据集注册类中查找接口类
    ROOT_PATH: "" 数据集存储路径
    TRAIN_TRANSFORMS: 训练数据处理管道
      LoadAnnotations: {} 加载标注
      PhotoMetricDistortion: {} 图像光度畸变
      MinIoURandomCrop: {} 最小IoU约束下的随机裁剪
      Resize: {} 图像大小变换
      RandomFlip: {} 图像翻转变换
      Normalize: {} 图像归一化
      DefaultFormatBundle: {} 图像格式转换
      Collect: {} 信息收集
    TEST_TRANSFORMS: 测试数据处理管道
      MultiScaleFlipAug: { 多尺度翻转增强
      img_scale: [300, 300], 图像大小
      flip: False, 是否翻转
      transforms: { 变换
        Resize: {keep_ratio: False},  图像大小变换
        Normalize: { 图像归一化变换
          mean: [123.675, 116.28, 103.53], 均值
          std: [1, 1, 1], 方差
          to_rgb: False}, 是否转换为RGB通道
        ImageToTensor: {keys: ['img']}, 图像变换到张量
        Collect: {keys: ['img']} 图像内容收集} }
    TRAIN_DATA: 训练数据集
      TYPE: "" 接口类型
      FLAG: 0 接口类型参数
      DATA_INFO: 数据集信息列表
      DATA_PREFIX: 数据集前缀信息列表
      SAMPLES_PER_DEVICE: 4 单设备采样数
      WORKERS_PER_DEVICE: 4 单设备进程数
    VAL_DATA: 验证数据集
      …
    TEST_DATA: 测试数据集
      …
  ```
  注：这里默认VAL数据的处理方式和TEST一样，DATA下一级子节点均是事先定义的，不建议扩展，因为扩展后的内容在程序调用时并不会关联。另外，这里的处理管道不会在默认配置中事先定义，而是采用值为dict的形式嵌套在各处理通道节点中，用于动态创建数据处理管道节点。

### 模型配置
  具体命名格式：{任务标识}_{模型名称}_{Backbone名称}(optional)_{Neck名称}(optional)_{Head名称}(optional)_{补充说明}
  任务标识与任务配置中一样
  模型名称说明：
  - 名称由模型名称+版本组成，例如mobilenetv3
  - 没有指定模型名称时，可由backbone名称代替
  - 模型名称中若是包含了Backbone、Neck或Head名称，则配置命名中不再包含对应的名称

#### 模型配置文件示例与说明
  ```YAML
  MODEL: 模型节点
    NUM_CLASSES: 1目标或场景类别
    CUDNN_BENCHMARK: True 是否用torch.backends.cudnn.benchmark加速
    TYPE: "" 模型类型，例如classifier、detector等
    NAME: "" 模型类名称，用于从模型注册类中查找处理该配置项的模型类
    PRETRAINED_MODEL_PATH: "" 预训练模型路径
    BACKBONE: 骨干网络
      type: "" 骨干网络名称，用于查找对应的类
      …
    BBOX_HEAD:
      type: "" 目标框头部网络名称，用于查找对应的类
      anchor_generator: 预选框生成器
        type: '' 生成器名称，用于查找对应的类
        …
      bbox_coder: 目标框编码器
        type: ‘’ 编码器名称，用于查找对应的类
        …
    TRAIN_CFG:
      assigner: 
        type: ''
        …
    TEST_CFG:
      nms: 
      type: ''
      …
    RPN_HEAD: 区域推荐网络头部
    ROI_HEAD: 感兴趣区域网络头部
    LOSS: 损失函数模块
    EXTEND: 模型扩展模块
  ```
  注：模型所需配置是由模型名称来确定的，未用到的模型节点不定义即可，并不会因为缺省类中有节点定义而影响模型调用。

### 训练策略配置
  训练主要包括训练epoch总数、优化器、优化器配置和学习率调整策略等。
  具体命名格式：schedule_{策略标识}_{补充说明}
  策略标识说明：
  - 标识名称可以是指定命名，如1x，2x，20e等等
  - 标识名称可以是优化器名称，如adam，adamw等等
  - 标识名称可以是具体方法名称，如vild，yolo等等
  补充说明：
  - 学习率调整策略，如cosinerestart，ld（layer decay）等等
  - 梯度裁剪，如grad设置

#### 训练策略配置示例与说明
  ```YAML
  SCHEDULE: 训练策略节点
    OPTIMIZER: 优化器
  type: "SGD" 优化器类型名称，用于查找对应的类
  …
    OPTIMIZER_CONFIG: 优化器配置
  …
    LR_POLICY: 学习率调整策略
  …
  ```
  注：训练策略主要包括优化器和学习率，后续策略扩展时可能会进一步合理配置其子节点结构。

### 运行时配置命名
  运行时设置主要包括训练与测试日志设置、模型训练过程数据设置以及运行环境设置等。
  具体格式：runtime_{运行时标识}
  运行时标识说明：
  - 运行时的基础设置或评估设置，例如base，eval等
  - 运行时的指定设置，例如vild，yolo等

#### 运行时配置文件示例与说明
  ```YAML
  RUNTIME: 运行时设置节点
    WORK_DIR: "" 过程数据存放路径
    LOAD_CHECKPOINT_PATH: "" 模型初始时加载路径（无梯度信息）
    RESUME_MODEL_PATH: "" 模型训练恢复时加载路径（有梯度信息）
    CHECKPOINT_CONFIG: 模型节点检查设置
      interval: 10 模型保存的Epoch间隔
    LOG_CONFIG: 日志设置
      interval: 50 模型日志输出的Step间隔
      hooks: {} 日志处理hooks
    LOG_LEVEL: "INFO" 日志层级
    DIST_PARAMS: ["backend", "nccl"] 分布式参数设置
    SEED: 42 随机种子
    WORKFLOW: [['train', 1]] 工作流设置
  ```
  注：运行时参数目前设置得相对扁平化，后续可进行层次化扩展。
