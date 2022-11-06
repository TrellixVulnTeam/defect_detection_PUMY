# 其他模块说明

## 辅助工具
  辅助工具类或脚本包括:
  - 文件处理类handlers
  - 目标框处理工具bbox_util
  - 模型节点检测工具checkpoint_util
  - 系统配置工具config_util
  - 数据处理工具data_util
  - 图像几何变换工具geometric_util
  - 初始化工具init_util
  - 文件输入输出工具io_util
  - 日志工具log_util
  - 损失函数工具loss_util
  - 掩膜工具mask_util
  - 度量工具metric_util
  - 并行处理工具parallel_util
  - 路径配置工具path_util
  - 视觉变换工具photometric_util
  - 注册类工具reg_util
  - 运行时工具runtime_util
  - 显著性检测工具saliency_util
  - 可视化工具vis_util
  - 其他辅助工具misc_util

### log_util
  get_logger: 用于根据名称搜索或初始化得到一个logger，log_file为log文件文件路径，log_level定义了输出到log_file的日志层级。
  print_log: 用于打印log日志。
  get_root_logger: 用于获取当前logger处理句柄。
  _minimal_ext_cmd：构建最小运行环境。
  get_git_hash: 用于获取git代码hash值。

### runtime_util
  collate: 将输入数据转换到有效的训练数据格式。

### vis_util
  可视化能够帮助用户浏览目标检测数据或保存数据可视化结果到指定目录。

## 核心部件
  核心部件主要包括：
  - 预选框生成anchor
  - 目标框处理bbox
  - 钩子函数类hooks
  - 扩展类操作ops
  - 优化器optimizer
  - 运行器runner
  注：可通过配置项生成对应的操作

### 预选框生成层anchor
  预选框生成层包含了3个生成器，即'AnchorGenerator', 'LegacyAnchorGenerator', 'YOLOAnchorGenerator'。

### 目标框处理层bbox
  目标框处理层包含了目标框分配、目标框编码、iou计算、目标框采样和目标框转换等功能，具体如下：
  - 目标框分配器bbox_assigners：AssignResult, CenterRegionAssigner, BaseAssigner, MaxIoUAssigner等。
  - 目标框编码器bbox_coders：BaseBBoxCoder, DeltaXYWHBBoxCoder, PseudoBBoxCoder, TBLRBBoxCoder等。
  - 交叉占比计算器iou_calculators：BboxOverlaps2D, bbox_overlaps等。
  - 目标框采样器bbox_samplers：BaseSampler, InstanceBalancedPosSampler, CombinedSampler, IoUBalancedNegSampler, OHEMSampler, PseudoSampler, RandomSampler, SamplingResult, ScoreHLRSampler等。
  - 目标框转换函数bbox_transforms： bbox2distance, bbox2result, bbox2roi, bbox_flip, bbox_mapping, bbox_mapping_back, bbox_rescale, distance2bbox, roi2bbox等。

### 钩子函数类hook
  HOOK可以理解为一种触发器，也可以理解为一种训练框架的架构规范，它规定了在算法训练过程中的种种操作，并且可以通过继承HOOK类，然后注册HOOK自定义想要的操作。
  可以说基类函数中定义了许多在模型训练中需要用到的一些功能，如果想定义一些操作我们就可以继承这个类并定制化我们的功能，可以看到HOOK中每一个参数都是有runner作为参数传入的。在你的每一个hook函数中，都可以对runner进行你想要的操作。而HOOK是怎么嵌套进runner中的呢？其实是在Runner中定义了一个hook的list，list中的每一个元素就是一个实例化的HOOK对象。其中提供了两种注册hook的方法，register_hook是传入一个实例化的HOOK对象，并将它插入到一个列表中，register_hook_from_cfg是传入一个配置项，根据配置项来实例化HOOK对象并插入到列表中。这里不仅仅是算法的复现，更是架构、编程范式的一种体现。
  钩子函数类用于处理训练期间的状态检测、评估、参数更新与日志输出等，目前框架支持：
  - 日志类6种（'LoggerHook', 'MlflowLoggerHook', 'PaviLoggerHook', 'TextLoggerHook', 'TensorboardLoggerHook', 'WandbLoggerHook'）
  - 检查类3种（'CheckpointHook', 'ClosureHook', 'IterTimerHook'）
  - 参数更新类8种（'LrUpdaterHook', 'OptimizerHook', 'Fp16OptimizerHook', 'DistSamplerSeedHook', 'EmptyCacheHook', 'MomentumUpdaterHook', 'SyncBuffersHook', 'EMAHook'）
  - 评估类2种（'EvalHook', 'DistEvalHook'）
  HOOK中不同类通过定义优先级来确定调用顺序的。具体的优先级定义有以下7种：
  - HIGHEST: 值为0
  - VERY_HIGH: 值为10
  - HIGH: 值为30
  - NORMAL: 值为50
  - LOW: 值为70
  - VERY_LOW: 值为90
  - LOWEST: 值为100

#### 钩子基类Hook解析
  ```python
  class Hook: # 类声明
    def before_run(self, runner): # 运行前处理
    def after_run(self, runner): # 运行后处理
    def before_epoch(self, runner): # 每个迭代周期前处理
    def after_epoch(self, runner): # 每个迭代周期后处理
    def before_iter(self, runner): # 每个迭代步骤（与步骤间隔配合使用）前处理
    def after_iter(self, runner): # 每个迭代步骤（与步骤间隔配合使用）后处理
    def before_train_epoch(self, runner): # 训练周期前处理
    def before_val_epoch(self, runner): # 验证周期前处理
    def after_train_epoch(self, runner): # 训练周期后处理
    def after_val_epoch(self, runner): # 验证周期后处理
    def before_train_iter(self, runner): # 训练步骤前处理
    def before_val_iter(self, runner): # 验证步骤前处理
    def after_train_iter(self, runner): # 训练步骤后处理
    def after_val_iter(self, runner): # 验证步骤后处理
    def every_n_epochs(self, runner, n): # 返回是否达到间隔周期数
    def every_n_inner_iters(self, runner, n): # 返回是否达到每个周期内的间隔步骤数
    def every_n_iters(self, runner, n): # 返回是否达到间隔步骤数
    def end_of_epoch(self, runner): # 返回是否达到周期的尾部
  ```

#### 日志类LoggerHook
  LoggerHook为日志类的基类，需要重载抽象函数log，该类会触发before_run、before_epoch、after_train_iter、after_train_epoch、after_val_epoch等函数。其他函数包括log、输入类型判断is_scalar、运行模式get_mode、获取周期get_epoch、获取迭代get_iter、获取学习率get_lr_tags、获取动量get_momentum_tags、获取日志标签get_loggable_tags等。

#### 其他日志类
  其他日志类主要通过相关库来实现，如mlflow、pavi、tensorboard、wandb。

#### 验证节点类CheckpointHook
  Interval与by_epoch配合使用，可按步骤保存或按周期保存。save_optimizer参数用于决定是否保存优化器参数，out_dir为输出路径（默认与工作路径一致），max_keep_ckpts为最大保存节点个数（多的会删除），sync_buffer用于决定是否同步不同设备的缓存。

#### 评估类EvalHook（DistEvalHook）
  目前评估类仅支持目标检测的评估，后续逐步扩展分类和多任务评估。

#### 时间类IterTimerHook
  该类用于记录开始前数据加载时间和单个模型单步时间。

#### 学习率更新类LrUpdaterHook
  该类用于单步或单周期调整学习率的基类，其扩展类前缀包括Fixed, Step, Exp, Poly, Inv, Cosine, CosineRestart, Cyclic等。

#### 动量更新类MomentumUpdaterHook
  该类用于单步或单周期调整动量的基类，其扩展类前缀包括Cosine, Cyclic等。

#### 优化器类OptimizerHook
  每个单步迭代完，优化器更新，损失函数反向传播，优化器单步执行。扩展类为支持16位浮点计算的Fp16OptimizerHook。

#### 其他类
  此外，还包括清理设备内存的钩子类EmptyCacheHook，采用种子设置类DistSamplerSeedHook和缓存同步类SyncBuffersHook。

### 核心扩展层ops
  核心扩展层包含了很多基于张量计算的网络层和扩展层操作，具体如下：
  - 注意力操作attention：’CrissCrossAttention’;
  - 池化操作pool：‘RoIAlign’, ‘roi_align’, ‘RoIPool’, ‘roi_pool’, ‘BaseMergeCell’, ‘SumCell’, ‘ConcatCell’, ‘GlobalPoolingCell’;
  - 卷积操作conv：‘fuse_conv_bn’,  ‘ConvModule’, ‘ConvWS2d’, ‘ConvAWS2d’, ‘conv_ws_2d’ ;
  - 非极大值抑制操作nms：‘common_nms’, ‘batched_nms’, ‘multiclass_nms’;
  - 像素点采样操作point_sample：’’SimpleRoIAlign’, ‘point_sample’, ‘rel_roi_point_to_rel_img_point’;
  - 注册集ops_builder：CONV_LAYERS, NORM_LAYERS, ACTIVATION_LAYERS, PADDING_LAYERS, UPSAMPLE_LAYERS, PLUGIN_LAYERS
  - 构建器ops_builder；build_ops_from_cfg, build_activation_layer, build_conv_layer, build_norm_layer, build_padding_layer, build_plugin_layer, build_upsample_layer

### 优化器层optimizer
  目前优化器采用了一种缺省注册类构造器，即DefaultOptimizerConstructor，里面可通过配置设置torch自带的所有优化器。

### 运行器runner
  Runner是一个模型训练的工厂，在其中我们可以加载数据、训练、验证以及梯度backward等等全套流程。在设计的时候也为runner传入丰富的参数，定义了一个非常好的训练范式。目前运行器只使用了EpochBaseRunner，所以在调用时没有通过config选择运行器，而是直接使用。

#### BaseRunner
  运行器基类，所有子类都需要实现run、train、val和save_checkpoint函数，类参数包括模型、优化器、工作目录、日志、重要信息、最大训练周期、最小迭代等。模型需要实现train_step。基类实现了模型名称model_name、进程序号rank、进程个数world_size、钩子类句柄hooks、迭代次数epoch、当前迭代周期epoch、当前迭代次数iter、每个迭代周期中的迭代次数inner_iter、最大迭代周期max_epochs、最大迭代次数max_iters、当前学习率current_lr、当前动量current_momentum、钩子类注册reguster_hook、从配置项注册一个钩子类register_hook_from_cfg、调用钩子函数call_hook、加载验证节点load_checkpoint、中断后继续训练resume、注册学习率钩子类register_lr_hook、注册动量钩子类register_momentum_hook、注册优化器钩子类register_optimizer_hook、注册验证节点钩子类register_checkpoint_hook、注册日志钩子类register_logger_hooks和注册训练钩子类register_training_hooks，最后一个函数将上述必要的钩子类统一进行了注册。

#### EpochBasedRunner
  run_iter：单步运行；
  train：单epoch训练；
  val：单epoch评估；
  run：根据配置进行模型训练；
  save_checkpoint: 模型验证节点保存。
