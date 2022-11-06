# 数据增强类说明
  数据增强操作通过Compose类（mtl/datasets/transforms/compose.py）进行串联拼接而成。
  数据转换类输入中未涉及的字典键值对会传递到输出中。
  数据管道操作可分为：数据加载、预处理、格式化和测试时增强等。
  数据dict中的key说明：
  - 有四个特别申明的数据域，用于Group操作，即img_fields, bbox_fields, mask_fields, seg_fields
  - img_fields在图像生成时导入，其他fields一般在LoadAnnotations导入（mtl/datasets/transforms/loading.py）
  - 元数据key包括：file_name, ori_shape, img_shape, pad_shape, joints_3d, joints_3d_visible, center, scale, rotation, bbox_score, flip_pairs等
  - 标注数据key包括：gt_bboxes, gt_labels, gt_masks, gt_semantic_seg, target, target_weight
  - 输入数据key包括：img, img_k, patch_mask, data_item, data_item_k

## Compose类
  将转换类聚合成一个计算序列。
  - __init__过程中需确保transforms为一个用于构建转换类的dict或一个可调用的转换类对象。
  - __call__依次调用转换类来处理数据，前一个转换类的输出为下一个转换类的输入，该过程中None输入的输出仍未None（跟代码规范可能会有一点冲突）。
  - __repr__用于显示转换类实例对象地址。

## 数据增强类示例
  该示例包括数据增强配置和数据增强类
  ```YAML
  TRAIN_TRANSFORMS: # 训练Pipeline标识
    ImgResize: {size: [256, -1]} # 图像大小转换
    ImgCenterCrop: {size: [224, 224]} # 图像中心裁剪
    ImgRandomFlip: {flip_prob: 0.5, direction: 'horizontal'} # 图像随机水平翻转
    Normalize: {
      mean: [123.675, 116.28, 103.53],
      std: [58.395, 57.12, 57.375],
      to_rgb: False
      } # 图像像素值归一化
    DefaultFormatBundle: {} # 将不同类型的数据分别处理，用于模型计算的数据转换为torch.Tensor类型
    ClsCollect: {
      keys: ['img', 'gt_label']
      } # 收集指定key所对应的数据，并将key和value一起用于模型训练
  ```
  ```python
  @PIPELINES.register_module() # 注册到PIPELINES中
  class GaussianBlur(object): # 数据增强类申明
    def __init__(self, sigma=None): # 初始化
      if sigma is None:
        sigma = [0.1, 5.0]
      self.sigma = sigma

    def __call__(self, results): # 类调用，results为一个字典
      sigma = int(random.uniform(self.sigma[0], self.sigma[1]))
      results["img"] = cv2.GaussianBlur(
        results["img"], (2 * sigma + 1, 2 * sigma + 1), 0
      ) # 更新字典中对应的数据内容
      return results # 返回更新后的字典

    def __repr__(self): # 类描述
      repr_str = f"{self.__class__.__name__}(" f"sigma={self.sigma})"
      return repr_str # 返回描述字段
  ```
