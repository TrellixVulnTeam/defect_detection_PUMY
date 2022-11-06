# 模型类说明
  神经网络模块主要包括骨干网络backbones、颈部网络necks、分类头部cls_heads、解码头部decode_heads、embedding头部emb_heads、位姿估计头部pose_heads、感兴趣区域头部网络roi_heads、区域推荐网络rpn_heads和损失函数模块losses。
  模型类主要由build_model来构建，不同类型的模型由不同的注册类来注册，见mtl/models/model_builder.py

## 模型类基类
  ```python
  class BaseModel(nn.Module, metaclass=ABCMeta): # 模型基类为抽象类，继承于torch.nn.Module
    def __init__(self): # 初始化
      super(BaseModel, self).__init__()

    @property
    def with_neck(self): # 是否含有neck模块
      return hasattr(self, "neck") and self.neck is not None

    @property
    def with_head(self): # 是否含有head模块
      return hasattr(self, "head") and self.head is not None

    @abstractmethod
    def forward_train(self, img, **kwargs): # 模型训练的前向计算部分
      pass

    @abstractmethod
    def simple_test(self, img, **kwargs): # 模型测试的前向计算部分
      pass

    def init_weights(self, pretrained=None): # 模型初始化，pretrained为backbone预训练模型的路径
      if pretrained is not None:
        print_log(f"load model from: {pretrained}", logger="root")

    def forward_test(self, img, **kwargs): # 模型测试的计算部分，相比simple_test，可以多写一些标准化的预处理或后处理
      return self.simple_test(img, **kwargs)

    def forward(self, img, **kwargs): # 模型前向计算
      if self.training:
        losses = self.forward_train(img, **kwargs)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(img))
      else:
        outputs = self.forward_test(img, **kwargs)
      return outputs

    def _parse_losses(self, losses): # 训练损失汇总解析，将dict中含‘loss’字段的损失值进行相加得到最终的loss
      log_vars = OrderedDict()
      for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
          log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, float):
          log_vars[loss_name] = loss_value
        elif isinstance(loss_value, list):
          log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        elif isinstance(loss_value, dict):
          for name, value in loss_value.items():
            log_vars[name] = value
        else:
          raise TypeError(f"{loss_name} is not a tensor or list of tensors")
        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)
        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
          if not isinstance(loss_value, (float, int)):
            if dist.is_available() and dist.is_initialized():
              loss_value = loss_value.data.clone()
              dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
          else:
            log_vars[loss_name] = loss_value
      return loss, log_vars
  ```

## 模型类构建示例
  ```python
  @CLASSIFIERS.register_module() # 将模型类注册到CLASSIFIERS
  class ImageClassifier(BaseClassifier): # 类声明 
    def __init__(self, cfg): # 类初始化
      super(ImageClassifier, self).__init__()
      self.type = cfg.TYPE
      self.backbone = build_backbone(cfg.BACKBONE)

      if len(cfg.NECK) > 0:
        self.neck = build_neck(cfg.NECK)

      if len(cfg.CLS_HEAD) > 0:
        self.head = build_head(cfg.CLS_HEAD)

      self.augmentor = None
      if len(cfg.TRAIN_CFG):
        train_cfg = convert_to_dict(cfg.TRAIN_CFG)
        augments_cfg = train_cfg.get('augmentor', None)
        if augments_cfg is not None:
          self.augmentor = AugmentConstructor(augments_cfg)

      if "PRETRAINED_MODEL_PATH" in cfg:
        if cfg.PRETRAINED_MODEL_PATH != "":
          self.init_weights(pretrained=cfg.PRETRAINED_MODEL_PATH)
        else:
          self.init_weights()
      else:
        self.init_weights()

    def init_weights(self, pretrained=None): # 模型参数初始化
      super(ImageClassifier, self).init_weights(pretrained)
      self.backbone.init_weights(pretrained=pretrained)
      if self.with_neck:
        if isinstance(self.neck, nn.Sequential):
          for m in self.neck:
            m.init_weights()
        else:
          self.neck.init_weights()
      if self.with_head:
        self.head.init_weights()

    def extract_feat(self, img): # 特征提取函数
      x = self.backbone(img)
      if self.with_neck:
        x = self.neck(x)
      return x

    def forward_train(self, img, gt_label, **kwargs): # 重载基类中的forward_train
      if self.augmentor is not None:
        img, gt_label = self.augmentor(img, gt_label)
      x = self.extract_feat(img)
      losses = dict()
      if self.augmentor is not None:
        loss = self.head.forward_train(x, gt_label, return_acc=False)
      else:
        loss = self.head.forward_train(x, gt_label)
      losses.update(loss)
      return losses

    def simple_test(self, img, **kwargs): # 重载基类中的simple_test
      x = self.extract_feat(img)
      return self.head(x)
  ```
  注：新构建模型一般只需要重载forward_train进行前向训练和重载simple_test进行模型推理

## 骨干网络backbones示例
  ```python
  @BACKBONES.register_module() # 将类注册到BACKBONES
  class MobileNetV3(nn.Module): # 类声明
    def __init__(
      self,
      arch="small",
      conv_cfg=None,
      norm_cfg=dict(type="BN"),
      out_indices=(10,),
      frozen_stages=-1,
      norm_eval=False,
      with_cp=False,
    ): # 类初始化
      super(MobileNetV3, self).__init__()        
      self.out_indices = out_indices
      self.frozen_stages = frozen_stages
      self.arch = arch
      self.conv_cfg = conv_cfg
      self.norm_cfg = norm_cfg
      self.out_indices = out_indices
      self.frozen_stages = frozen_stages
      self.norm_eval = norm_eval
      self.with_cp = with_cp
      ......

    def init_weights(self, pretrained=None): # 初始化参数
      if isinstance(pretrained, str):
        logger = logging.getLogger()
        load_checkpoint(self, pretrained, strict=False, logger=logger)
      elif pretrained is None:
        for m in self.modules():
          if isinstance(m, nn.Conv2d):
            kaiming_init(m)
          elif isinstance(m, nn.BatchNorm2d):
            constant_init(m, 1)
      else:
        raise TypeError("pretrained must be a str or None")

    def forward(self, x): # 模型前向计算
      x = self.conv1(x)
      outs = []
      for i, layer_name in enumerate(self.layers):
        layer = getattr(self, layer_name)
        x = layer(x)
        if i in self.out_indices:
          outs.append(x)

      if len(outs) == 1:
        return outs[0]
      else:
        return tuple(outs)

    def _freeze_stages(self): # 冻结模型部分stages
      if self.frozen_stages >= 0:
        for param in self.conv1.parameters():
          param.requires_grad = False
      for i in range(1, self.frozen_stages + 1):
        layer = getattr(self, f"layer{i}")
        layer.eval()
        for param in layer.parameters():
          param.requires_grad = False

    def train(self, mode=True): # 重载模型训练
      super(MobileNetV3, self).train(mode)
      self._freeze_stages()
      if mode and self.norm_eval:
        for m in self.modules():
          if isinstance(m, _BatchNorm):
            m.eval()
  ```
  注：骨干网络backbones的构造与一般的nn.Module类似，一般只需要重载forward和init_weights

## 颈部网络necks示例
  ```python
  @NECKS.register_module() # 将类注册到NECKS
  class GlobalAveragePooling(nn.Module): # 类声明
    def __init__(self): # 类初始化
      super(GlobalAveragePooling, self).__init__()
      self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self): # 模型参数初始化
      pass

    def forward(self, inputs): # 模型前向计算
      if isinstance(inputs, tuple):
        outs = tuple([self.gap(x) for x in inputs])
        outs = tuple([out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
      elif isinstance(inputs, torch.Tensor):
        outs = self.gap(inputs)
        outs = outs.view(inputs.size(0), -1)
      else:
        raise TypeError("neck inputs should be tuple or torch.tensor")
      return outs
  ```
  注：颈部网络necks的构造与一般的nn.Module类似，一般只需要重载forward和init_weights

## 分类头部cls_heads示例
  ```python
  @HEADS.register_module() # 将类注册到HEADS
  class LinearClsHead(BaseClsDenseHead): # 类声明
    def __init__(
      self,
      num_classes,
      in_channels,
    ): # 类初始化
      super(LinearClsHead, self).__init__()
      self.in_channels = in_channels
      self.num_classes = num_classes
      self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self): # 模型参数初始化
      normal_init(self.fc, mean=0, std=0.01, bias=0)

    def forward(self, x): # 模型前向计算
      if isinstance(x, (tuple, list)):
        x = x[-1]
      x = self.fc(x)
      return x
  ```
  注：头部网络的构造与一般的nn.Module类似，一般只需要重载forward和init_weights

## 损失函数模块losses示例
  损失函数定义步骤如下：
  - 通过损失核函数计算元素或样例的损失函数；
  - 获得各元素的损失权值项；
  - 将损失值聚合为一个常量；
  - 获得损失函数的总体权值。

  ```python
  @LOSSES.register_module() # 将损失类注册到LOSSES
  class MSELoss(nn.Module): # 类声明
    def __init__(self, reduction="mean", loss_weight=1.0): # 类初始化
      super().__init__()
      self.reduction = reduction
      self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None): # 模型前向计算
      loss = mse_loss(
        pred, target, weight, reduction=self.reduction, avg_factor=avg_factor
      )
      return self.loss_weight * loss
  ```
  注：损失类同样为nn.Module的派生类，只需定义forward就可以利用nn.Module的自动求导机制
