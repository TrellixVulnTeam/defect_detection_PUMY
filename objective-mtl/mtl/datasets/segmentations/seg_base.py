import os
import os.path as osp
from functools import reduce
import numpy as np
from terminaltables import AsciiTable

from mtl.utils.io_util import list_from_file, imread
from mtl.utils.log_util import print_log
from mtl.cores.eval.common_eval import eval_seg_metrics
from mtl.utils.log_util import get_root_logger
from mtl.utils.misc_util import is_list_of
from mtl.datasets.data_wrapper import DATASETS
from ..data_base import DataBaseDataset


@DATASETS.register_module()
class SegBaseDataset(DataBaseDataset):
    """base dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none
        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val
    """

    class_names = None
    palette = None

    def __init__(self, data_cfg, pipeline_cfg, root_path, sel_index=0):
        """Initialization for dataset construction

        Args:
            data_cfg (cfgNode): dataset info.
            pipeline_cfg (cfgNode): Processing pipeline info.
            root_path (str, optional): Data root for ``ann_file``, ``data_prefix``,
                ``seg_prefix``, ``proposal_file`` if specified.
            sel_index (int): select the annotation file with the index from
                annotation list.
        """
        super(SegBaseDataset, self).__init__(
            data_cfg, pipeline_cfg, root_path, sel_index
        )

        if "CLASS_NAMES" in data_cfg and "PALETTE" in data_cfg:
            if isinstance(data_cfg.CLASS_NAMES, str):
                if data_cfg.CLASS_NAMES.endswith('txt'):
                    self.class_names, self.palette = self.get_classes_and_palette(
                        osp.join(self.data_root, data_cfg.CLASS_NAMES), data_cfg.PALETTE
                    )
                else:
                    self.class_names, self.palette = self.get_classes_and_palette()
            else:
                self.class_names, self.palette = self.get_classes_and_palette(
                    data_cfg.CLASS_NAMES, data_cfg.PALETTE
                )
        else:
            self.class_names, self.palette = self.get_classes_and_palette()

        if "IGNORE_INDEX" in data_cfg:
            self.ignore_index = data_cfg.IGNORE_INDEX
        self.reduce_zero_label = data_cfg.REDUCE_ZERO_LABEL
        self.label_map = None

        if self.is_tfrecord:
            self.proposals = None
            self.anno_prefix = None
            self.seg_prefix = None

    def get_annotations(self, data_cfg, sel_index):
        if "DATA_PREFIX" in data_cfg and isinstance(data_cfg.DATA_PREFIX, list):
            self.data_prefix = data_cfg.DATA_PREFIX[sel_index]
        else:
            self.data_prefix = None
        if "ANNO_PREFIX" in data_cfg and isinstance(data_cfg.ANNO_PREFIX, list):
            self.anno_prefix = data_cfg.ANNO_PREFIX[sel_index]
        else:
            self.anno_prefix = None
        if "SEG_PREFIX" in data_cfg and isinstance(data_cfg.SEG_PREFIX, list):
            self.seg_prefix = data_cfg.SEG_PREFIX[sel_index]
        else:
            self.seg_prefix = None
        if not (self.data_prefix is None or osp.isabs(self.data_prefix)):
            self.data_prefix = osp.join(self.data_root, self.data_prefix)
        if not (self.anno_prefix is None or osp.isabs(self.anno_prefix)):
            self.anno_prefix = osp.join(self.data_root, self.anno_prefix)
        if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
            self.seg_prefix = osp.join(self.data_root, self.seg_prefix)

        self.ann_file = self.ann_file[0]
        if self.ann_file == "":
            self.ann_file = None
        else:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
        self.img_suffix = ".jpg"
        self.anno_suffix = ".png"
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            anno_dir (str|None): Path to annotation directory.
            anno_file (str|None): Annotation file. If it is specified, only file
                with suffix in the file will be loaded. Otherwise, all images
                in anno_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """
        data_infos = []
        if self.ann_file is not None:
            img_ids = list_from_file(self.ann_file)

            for img_ind in img_ids:
                data_info = dict(file_name=img_ind + self.img_suffix)
                # if self.anno_prefix is not None:
                seg_file = img_ind + self.anno_suffix
                data_info["ann"] = dict(seg_map=seg_file)
                data_infos.append(data_info)
        else:
            len_img_suffix = len(self.img_suffix)
            for img_file in os.listdir(self.data_prefix):
                if img_file.endswith(self.img_suffix):
                    img_ind = img_file[:-len_img_suffix]
                    data_info = dict(file_name=img_ind + self.img_suffix)
                    # if self.anno_prefix is not None:
                    seg_file = img_ind + self.anno_suffix
                    data_info["ann"] = dict(seg_map=seg_file)
                    data_infos.append(data_info)

        print_log(f"Loaded {len(data_infos)} images", logger=get_root_logger())
        return data_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        return self.getitem_info(idx)["ann"]

    def getitem_info(self, index, return_img=True):
        datainfo = super(SegBaseDataset, self).getitem_info(index, return_img)
        if not self.is_tfrecord:
            if "img" in datainfo:
                datainfo["height"] = datainfo["img"].shape[0]
                datainfo["width"] = datainfo["img"].shape[1]
            else:
                datainfo["height"] = 100
                datainfo["width"] = 100
        return datainfo

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results["img_prefix"] = self.data_prefix
        results["seg_prefix"] = self.seg_prefix
        if self.label_map is not None:
            results["label_map"] = self.label_map

    def prepare_data(self, idx):
        if self.test_mode:
            results = self.prepare_test_img(idx)
        else:
            results = self.prepare_train_img(idx)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        results = self.getitem_info(idx)
        if results is None:
            new_idx = np.random.randint(0, len(self))
            return self.prepare_train_img(new_idx)
        results["img_shape"] = results["img"].shape
        results["ori_shape"] = results["img"].shape
        results["img_fields"] = ["img"]
        if "gt_semantic_seg" in results:
            results["seg_fields"] = ["gt_semantic_seg"]
        else:
            results["seg_fields"] = []
        return results

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """
        results = self.getitem_info(idx)
        if results is None:
            new_idx = np.random.randint(0, len(self))
            return self.prepare_test_img(new_idx)
        results["img_shape"] = results["img"].shape
        results["ori_shape"] = results["img"].shape
        results["img_fields"] = ["img"]
        return results

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        if self.is_tfrecord:
            for idx in range(self.samples):
                item_info = self.getitem_info(idx)
                gt_seg_maps.append(item_info["ann"]["seg_map"])
        else:
            for data_info in self.data_infos:
                seg_map = osp.join(self.seg_prefix, data_info["ann"]["seg_map"])
                gt_seg_map = imread(seg_map, flag="unchanged", backend="pillow")
                gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default class_names defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the class_names defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.class_names, self.palette

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f"Unsupported type {type(classes)} of classes.")

        if self.class_names:
            if not set(class_names).issubset(self.class_names):
                raise ValueError("classes is not a subset of class_names.")

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.class_names):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):
        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.palette[old_id])
            palette = type(self.palette)(palette)

        elif palette is None:
            if self.palette is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.palette

        return palette

    def evaluate(self, results, metric="mIoU", logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ["mIoU", "mDice"]
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError("metric {} is not supported".format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps()
        if self.class_names is None:
            num_classes = len(reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.class_names)
        ret_metrics = eval_seg_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label,
        )
        class_table_data = [["Class"] + [m[1:] for m in metric] + ["Acc"]]
        if self.class_names is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.class_names
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes):
            class_table_data.append(
                [class_names[i]]
                + [m[i] for m in ret_metrics_round[2:]]
                + [ret_metrics_round[1][i]]
            )
        summary_table_data = [
            ["Scope"] + ["m" + head for head in class_table_data[0][1:]] + ["aAcc"]
        ]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2) for ret_metric in ret_metrics
        ]
        summary_table_data.append(
            ["global"]
            + ret_metrics_mean[2:]
            + [ret_metrics_mean[1]]
            + [ret_metrics_mean[0]]
        )
        print_log("per class results:", logger)
        table = AsciiTable(class_table_data)
        print_log("\n" + table.table, logger=logger)
        print_log("Summary:", logger)
        table = AsciiTable(summary_table_data)
        print_log("\n" + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0][i]] = summary_table_data[1][i] / 100.0
        if is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results
