import os.path as osp
import numpy as np
from PIL import Image
from io import BytesIO

from mtl.cores.eval.common_eval import eval_map, eval_recalls
from mtl.datasets.data_wrapper import DATASETS
from mtl.utils.io_util import file_load, list_from_file
from ..data_base import DataBaseDataset


@DATASETS.register_module()
class DetBaseDataset(DataBaseDataset):
    """Base dataset for detection."""

    class_names = None

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
        super(DetBaseDataset, self).__init__(
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

        if "FILTER_EMPTY_GT" in data_cfg:
            self.filter_empty_gt = data_cfg.FILTER_EMPTY_GT
        else:
            self.filter_empty_gt = True

        if self.is_tfrecord:
            self.proposals = None
            self.anno_prefix = None
            self.seg_prefix = None
            self.proposal_file = None

    def get_annotations(self, data_cfg, sel_index):
        if "ANNO_PREFIX" in data_cfg and isinstance(data_cfg.ANNO_PREFIX, list):
            self.anno_prefix = data_cfg.ANNO_PREFIX[sel_index]
        else:
            self.anno_prefix = None
        if "SEG_PREFIX" in data_cfg and isinstance(data_cfg.SEG_PREFIX, list):
            self.seg_prefix = data_cfg.SEG_PREFIX[sel_index]
        else:
            self.seg_prefix = None
        if "PROPOSAL_FILE" in data_cfg and isinstance(data_cfg.PROPOSAL_FILE, list):
            self.proposal_file = data_cfg.PROPOSAL_FILE[sel_index]
        else:
            self.proposal_file = None
        if not (self.anno_prefix is None or osp.isabs(self.anno_prefix)):
            self.anno_prefix = osp.join(self.data_root, self.anno_prefix)
        if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
            self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
        if not (self.proposal_file is None or osp.isabs(self.proposal_file)):
            self.proposal_file = osp.join(self.data_root, self.proposal_file)
        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
        super(DetBaseDataset, self).get_annotations(data_cfg, sel_index)

        # filter images too small and containing no annotations
        if not self.test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        return file_load(ann_file)

    def load_proposals(self, proposal_file):
        """Load proposal from proposal file."""
        return file_load(proposal_file)

    def record_parser(self, feature_list, return_img=True):
        """Call when is_tfrecord is ture.

        feature_list = [(key, feature), (key, feature)]
        key is your label.txt col name
        feature is oneof bytes_list, int64_list, float_list
        """
        for key, feature in feature_list:
            # for image file col
            if key == "name" or key == "image_name":
                file_name = feature.bytes_list.value[0].decode("UTF-8", "strict")
            elif key == "image":
                if return_img:
                    image_raw = feature.bytes_list.value[0]
                    pil_img = Image.open(BytesIO(image_raw)).convert("RGB")
                    img = np.array(pil_img).astype(np.float32)
                else:
                    img = None
            elif key == "bbox/class":
                obj_cls = feature.int64_list.value
            elif key == "bbox/xmin":
                obj_xmin = feature.int64_list.value
            elif key == "bbox/ymin":
                obj_ymin = feature.int64_list.value
            elif key == "bbox/xmax":
                obj_xmax = feature.int64_list.value
            elif key == "bbox/ymax":
                obj_ymax = feature.int64_list.value

        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []

        for i in range(len(obj_cls)):
            label = obj_cls[i]
            bbox = [
                int(float(obj_xmin[i]) + 0.5),
                int(float(obj_ymin[i]) + 0.5),
                int(float(obj_xmax[i]) + 0.5),
                int(float(obj_ymax[i]) + 0.5),
            ]

            ignore = False
            if hasattr(self, "min_size"):
                if self.min_size is not None:
                    assert not self.test_mode
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w < self.min_size or h < self.min_size:
                        ignore = True

            if ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)

        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64),
        )
        if img is None:
            img_shape = (-1, -1)
        else:
            img_shape = img.shape

        return {
            "file_name": file_name + ".jpg",
            "width": img_shape[1],
            "height": img_shape[0],
            "img": img,
            "ann": ann,
        }

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        return self.getitem_info(idx)["ann"]

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        return self.get_ann_info(idx)["labels"].astype(np.int).tolist()

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results["img_prefix"] = self.data_prefix
        results["seg_prefix"] = self.seg_prefix
        results["proposal_file"] = self.proposal_file
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["dataset"] = self

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info["width"], img_info["height"]) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def batch_rand_others(self, idx, batch):
        """Get a batch of random index from the same group as the given
        index."""
        mask = self.flag == self.flag[idx]
        mask[idx] = False
        pool = np.where(mask)[0]
        if len(pool) == 0:
            return np.array([idx] * batch)
        if len(pool) < batch:
            return np.random.choice(pool, size=batch, replace=True)
        return np.random.choice(pool, size=batch, replace=False)

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
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        results = self.getitem_info(idx)
        if results is None:
            new_idx = self._rand_another(idx)
            return self.prepare_train_img(new_idx)
        results["img_fields"] = ["img"]
        results["_idx"] = idx
        results["img_shape"] = results["img"].shape
        results["ori_shape"] = results["img"].shape
        if self.proposals is not None:
            results["proposals"] = self.proposals[idx]
        return results

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by \
                piepline.
        """
        results = self.getitem_info(idx)
        if results is None:
            new_idx = self._rand_another(idx)
            return self.prepare_test_img(new_idx)
        results["img_fields"] = ["img"]
        results["_idx"] = idx
        results["img_shape"] = results["img"].shape
        results["ori_shape"] = results["img"].shape
        if self.proposals is not None:
            results["proposals"] = self.proposals[idx]
        return results

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
            class_names = list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f"Unsupported type {type(classes)} of classes.")

        return class_names

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def evaluate(
        self,
        results,
        metric="map",
        logger=None,
        proposal_nums=(100, 300, 1000),
        iou_thrs=None,
        scale_ranges=None,
    ):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating map, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating map.
                Default: None.
        """
        if iou_thrs is None:
            iou_thrs = np.linspace(
                0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
            )
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ["map", "recall"]
        if metric not in allowed_metrics:
            raise KeyError(f"metric {metric} is not supported")
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == "map":
            if isinstance(iou_thrs, float):
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thrs,
                    dataset=self.class_names,
                    logger=logger,
                )
                eval_results["map"] = mean_ap
            else:
                # assert isinstance(iou_thr, list)
                all_mean_ap = 0
                for iou_single in iou_thrs:
                    # print(iou_single)
                    mean_ap, _ = eval_map(
                        results,
                        annotations,
                        scale_ranges=scale_ranges,
                        iou_thr=iou_single,
                        dataset=self.class_names,
                        logger=logger,
                        print_summary=False,
                    )
                    if abs(iou_single - 0.5) < 1e-3:
                        eval_results["ap@50"] = mean_ap
                    elif abs(iou_single - 0.75) < 1e-3:
                        eval_results["ap@75"] = mean_ap
                    all_mean_ap += mean_ap
                all_mean_ap = all_mean_ap / len(iou_thrs)
                eval_results["map"] = all_mean_ap

        elif metric == "recall":
            gt_bboxes = [ann["bboxes"] for ann in annotations]
            if isinstance(iou_thrs, float):
                iou_thrs = [iou_thrs]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thrs, logger=logger
            )
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thrs):
                    eval_results[f"recall@{num}@{iou}"] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f"AR@{num}"] = ar[i]

        return eval_results
