import copy
import os.path as osp
from io import BytesIO
from PIL import Image
import numpy as np

from mtl.utils.metric_util import multi_label_accuracy, compute_map
from ..data_builder import DATASETS
from .cls_base import ClsBaseDataset


@DATASETS.register_module()
class OIDDataset(ClsBaseDataset):
    """OID classification"""

    def record_parser(self, feature_list, return_img=True):
        """Call when is_tfrecord is ture.

        feature_list = [(key, feature), (key, feature)]
        key is your label.txt col name
        feature is oneof bytes_list, int64_list, float_list
        """
        for key, feature in feature_list:
            # for image file col
            if key == "name":
                file_name = feature.bytes_list.value[0].decode("UTF-8", "strict")
            if key == "image":
                if return_img:
                    image_raw = feature.bytes_list.value[0]
                    pil_img = Image.open(BytesIO(image_raw)).convert("RGB")
                    img = np.array(pil_img).astype(np.float32)
                else:
                    img = None
            elif key == "label":
                gt_label_idx = feature.int64_list.value
            elif key == "confidence":
                confidence = feature.int64_list.value

        # transfer gt_label to 0-1 vector
        gt_label_idx = np.array(gt_label_idx)
        confidence = np.array(confidence)
        gt_idx_filter = gt_label_idx[np.nonzero(gt_label_idx * confidence)]
        gt_label = np.zeros(len(self.class_names))
        gt_label[gt_idx_filter] = 1.0
        if img is None:
            img_shape = (-1, -1)
        else:
            img_shape = img.shape

        return {
            "file_name": file_name,
            "img": img,
            "gt_label": gt_label,
            "ori_shape": img_shape,
        }

    def prepare_data(self, idx):
        """Prepare data and run pipelines"""
        if self.is_tfrecord:
            results = copy.deepcopy(self.getitem_info(idx))
        else:
            if self.data_prefix is not None:
                img_path = osp.join(self.data_prefix, self.data_infos[idx]["file_name"])
            else:
                img_path = osp.join(self.data_root, self.data_infos[idx]["file_name"])

            if not osp.exists(img_path):
                raise ValueError(f"Incorrect image path {img_path}.")

            pil_img = Image.open(img_path).convert("RGB")
            img = np.array(pil_img).astype(np.uint8)

            results = {
                "file_name": self.data_infos[idx]["file_name"],
                "img": img,
                "gt_label": self.data_infos[idx]["label"],
                "ori_shape": img.shape,
            }

        return self.pipeline(results)

    def evaluate(self, results, metric="map", metric_options=None, logger=None):
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
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = ["accuracy", "map"]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f"metric {metric} is not supported.")

            if metric == "accuracy":
                acc_per_cls, nan_index = multi_label_accuracy(results, gt_labels, 0.5)

                # save result and index
                print("NaN index num: ", len(nan_index))
                print("Total classes: ", len(acc_per_cls))
                accuracy_value = acc_per_cls.mean()
                eval_results = {"accuracy": accuracy_value}
                # for i in range(len(self.class_names)):
                #    eval_results['acc_' + self.class_names[i]] = acc_per_cls[i]
            elif metric == "map":
                mean_ap, _ = compute_map(gt_labels, results)
                eval_results = {"map": mean_ap}
            else:
                raise NotImplementedError

            return eval_results
