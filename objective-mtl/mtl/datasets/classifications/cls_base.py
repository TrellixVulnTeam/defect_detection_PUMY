# -*- encoding: utf-8 -*-

###########################################################
# @File    :   cls_base.py
# @Time    :   2021/10/13 20:33:08
# @Author  :   Qian Zhiming
# @Contact :   zhiming.qian@micro-i.com.cn
###########################################################

import os.path as osp
from io import BytesIO
from PIL import Image
import numpy as np

from mtl.utils.metric_util import accuracy, f1_score, precision, recall, prcurve
from mtl.utils.io_util import list_from_file
from ..data_base import DataBaseDataset


class ClsBaseDataset(DataBaseDataset):
    """Base dataset."""

    class_names = None

    def __init__(self, data_cfg, pipeline_cfg, root_path, sel_index=0):
        """Initialization for dataset construction."""
        super(ClsBaseDataset, self).__init__(
            data_cfg, pipeline_cfg, root_path, sel_index
        )

        if "CLASS_NAMES" in data_cfg:
            if isinstance(data_cfg.CLASS_NAMES, str):
                if data_cfg.CLASS_NAMES.endswith('txt'):
                    self.class_names = self.get_classes(
                        osp.join(self.data_root, data_cfg.CLASS_NAMES)
                    )
                else:
                    self.class_names = self.get_classes()
            else:
                self.class_names = self.get_classes(data_cfg.CLASS_NAMES)
        else:
            self.class_names = self.get_classes()

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """
        return {_class: i for i, _class in enumerate(self.class_names)}

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            list[int]: categories for all images.
        """
        gt_labels = []
        for i in range(len(self)):
            gt_labels.append(self.getitem_info(i, return_img=False)["gt_label"])

        gt_labels = np.array(gt_labels)
        return gt_labels

    def get_cat_ids(self, idx):
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            int: Image category of specified index.
        """
        cat_ids = self.getitem_info(idx, return_img=False)["gt_label"]
        if isinstance(cat_ids, list):
            return np.array(cat_ids).astype(np.int)
        elif isinstance(cat_ids, np.ndarray):
            return cat_ids.astype(np.int)
        return np.asarray([cat_ids]).astype(np.int)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default class_names defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the class_names defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
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

    def prepare_data(self, idx):
        """Prepare data and run pipelines"""
        results = self.getitem_info(idx)
        if results is None:
            new_idx = np.random.randint(0, len(self))
            return self.prepare_data(new_idx)
        if not self.is_tfrecord:
            results["gt_label"] = self.data_infos[idx]["label"]

        return self.pipeline(results)

    def record_parser(self, feature_list, return_img=True):
        """Call when is_tfrecord is ture.
        feature_list = [(key, feature), (key, feature)]
        key is your label.txt col name
        feature is oneof bytes_list, int64_list, float_list
        """
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

        return {"file_name": file_name, "img": img, "gt_label": gt_label}

    def evaluate(self, results, metric="accuracy", metric_options=None, logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.

            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict: evaluation results
        """
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

        # assert len(gt_labels) == num_imgs
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
