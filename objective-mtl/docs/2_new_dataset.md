# 数据类说明
  数据类包括构建类、采样策略、数据集处理类以及各种数据集接口等
  - 数据类主要由数据构建类（'build_dataset', 'build_dataloader'）来串联，见mtl/datasets/data_builder.py
  - 数据采样策略主要包括DistributedSampler，GroupSampler，DistributedGroupSampler和torch原生的采样器，主要在构造dataloader时调用
  - 通用数据集处理类支持同类数据集合并、数据集重复和类平衡等（'ConcatDataset', 'RepeatDataset'和'ClassBalancedDataset'），见mtl/datasets/data_wrapper.py。
  - 各种数据类注册到DATASETS，这是一个注册类Registry("dataset")，见mtl/datasets/data_wrapper.py, 所有数据类都派生自torch原生的Dataset类

## 数据基类
  数据基类的基本接口仍然是__getitem__函数，通过调用prepare_data函数来获取指定idx的数据。
  prepare_data函数中一般包含两步：获取原始数据getitem_info和数据增强处理pipeline（由get_pipeline_list来获取配置中的pipeline）
  在获取原始数据时：
  - 若数据格式为tfrecord，则通过get_tfrecords、get_record和record_parser进行处理
  - 若数据格式为单文件集合，则通过get_annotations和load_annotations进行处理

## 数据接口定义示例
  ```python
  class ClsNameDataset(DataBaseDataset): # 数据类名称，继承于数据基类
    class_names = None # 类别名称
    def __init__(self, data_cfg, pipeline_cfg, root_path, sel_index=0):
        super(ClsNameDataset, self).__init__(
            data_cfg, pipeline_cfg, root_path, sel_index
        ) # 继承基类的初始化
        self.class_names = self.get_classes() # 获取类别名称列表

    @property
    def class_to_idx(self): # 类别与索引的转换（optional）
        return {_class: i for i, _class in enumerate(self.class_names)}

    def get_gt_labels(self): # 获取dataset的ground truth信息（optional）
        gt_labels = []
        for i in range(len(self)):
            gt_labels.append(self.getitem_info(i, return_img=False)["gt_label"])
        gt_labels = np.array(gt_labels)
        return gt_labels

    def get_cat_ids(self, idx): # 获取类别id信息（optional）
        cat_ids = self.getitem_info(idx)["gt_label"]
        if isinstance(cat_ids, list):
            return np.array(cat_ids).astype(np.int)
        elif isinstance(cat_ids, np.ndarray):
            return cat_ids.astype(np.int)
        return np.asarray([cat_ids]).astype(np.int)

    @classmethod
    def get_classes(cls, classes=None): # 获取类列表（optional）
        if classes is None:
            return cls.class_names
        if isinstance(classes, str):
            # take it as a file path
            class_names = list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f"Unsupported type {type(classes)} of classes.")
        return class_names

    def prepare_data(self, idx): # 根据索引获取数据
        """Prepare data and run pipelines"""        
        results = self.getitem_info(idx)
        if results is None:
            new_idx = np.random.randint(0, len(self))
            return self.prepare_data(new_idx)
        if not self.is_tfrecord:
            results["gt_label"] = self.data_infos[idx]["label"]
        return self.pipeline(results)

    def record_parser(self, feature_list, return_img=True): # 解析tfrecord数据
        for key, feature in feature_list:
            # for image file col
            if key == "image_id" or key == "name":
                file_name = feature.bytes_list.value[0].decode("UTF-8", "strict")
            if key == "image":
                if return_img:
                    image_raw = feature.bytes_list.value[0]
                    pil_img = Image.open(BytesIO(image_raw)).convert("RGB")
                    img = np.array(pil_img).astype(np.float32)
                else:
                    img = None
            elif key == "label":
                gt_label = feature.int64_list.value[0]

        return {
            "file_name": file_name,
            "img": img,
            "gt_label": gt_label,
        }

    def evaluate(self, results, metric="accuracy", metric_options=None, logger=None): # 根据模型运行结果，对数据集指标进行评估
        if metric_options is None:
            metric_options = {"topk": (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = ["accuracy", "precision", "recall", "f1_score", "prcurve"]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        num_imgs = len(results)
        gt_labels = gt_labels[:num_imgs]

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f"metric {metric} is not supported.")
            if metric == "accuracy":
                topk = metric_options.get("topk")
                acc = accuracy(results, gt_labels, topk)
                eval_result = {f"top-{k}": a.item() for k, a in zip(topk, acc)}
            elif metric == "precision":
                precision_value = precision(results, gt_labels)
                eval_result = {"precision": precision_value}
            elif metric == "recall":
                recall_value = recall(results, gt_labels)
                eval_result = {"recall": recall_value}
            elif metric == "f1_score":
                f1_score_value = f1_score(results, gt_labels)
                eval_result = {"f1_score": f1_score_value}
            elif metric == "prcurve":
                prcurve_value = prcurve(results, gt_labels)
                eval_result = {"prcurve": prcurve_value}
            eval_results.update(eval_result)

        return eval_results
  ```

