import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

from mtl.cores.eval.common_eval import eval_map, eval_recalls
from mtl.datasets.data_wrapper import DATASETS
from mtl.utils.io_util import list_from_file
from .det_base import DetBaseDataset


@DATASETS.register_module()
class VOCDataset(DetBaseDataset):
    """Data interface for the VOC dataset"""

    class_names = (
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    def __init__(self, data_cfg, pipeline_cfg, root_path, sel_index=0):
        """Same as base detection dataset"""
        super(VOCDataset, self).__init__(data_cfg, pipeline_cfg, root_path, sel_index)
        self.cat2label = {cat: i for i, cat in enumerate(self.class_names)}
        if "MIN_SIZE" in data_cfg:
            self.min_size = data_cfg.MIN_SIZE
        else:
            self.min_size = None
        if not self.is_tfrecord:
            if "VOC2007" in self.data_prefix:
                self.year = 2007
            elif "VOC2012" in self.data_prefix:
                self.year = 2012
            else:
                raise ValueError("Cannot infer dataset year from data_prefix")

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        data_infos = []
        img_inds = list_from_file(ann_file)
        for file_name in img_inds:
            filename = f"JPEGImages/{file_name}.jpg"
            xml_path = osp.join(self.data_prefix, "Annotations", f"{file_name}.xml")
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            width = 0
            height = 0
            if size is not None:
                width = int(size.find("width").text)
                height = int(size.find("height").text)
            else:
                img_path = osp.join(self.data_prefix, filename)
                img = Image.open(img_path).convert("RGB")
                width, height = img.size
            data_infos.append(dict(file_name=filename, width=width, height=height))

        return data_infos

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without annotation."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info["width"], img_info["height"]) < min_size:
                continue
            if self.filter_empty_gt:
                file_name = img_info["file_name"]
                xml_path = osp.join(self.data_prefix, "Annotations", f"{file_name}.xml")
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    name = obj.find("name").text
                    if name in self.class_names:
                        valid_inds.append(i)
                        break
            else:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        if self.is_tfrecord:
            return self.getitem_info(idx)["ann"]

        file_name = self.data_infos[idx]["file_name"]
        xml_path = osp.join(self.data_prefix, "Annotations", f"{file_name}.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in self.class_names:
                continue
            label = self.cat2label[name]
            difficult = int(obj.find("difficult").text)
            bnd_box = obj.find("bndbox")
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find("xmin").text)),
                int(float(bnd_box.find("ymin").text)),
                int(float(bnd_box.find("xmax").text)),
                int(float(bnd_box.find("ymax").text)),
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64),
        )
        return ann

    def get_cat_ids(self, idx):
        """Get category ids in XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        cat_ids = []
        file_name = self.data_infos[idx]["file_name"]
        xml_path = osp.join(self.data_prefix, "Annotations", f"{file_name}.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in self.class_names:
                continue
            label = self.cat2label[name]
            cat_ids.append(label)

        return cat_ids

    def evaluate(
        self,
        results,
        metric="map",
        logger=None,
        proposal_nums=(100, 300, 1000),
        iou_thrs=None,
    ):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'map', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating map, and can be a list when evaluating recall.
                Default: 0.5.

        Returns:
            dict[str, float]: AP/recall metrics.
        """
        if self.is_tfrecord:
            return super(VOCDataset, self).evaluate(
                results, metric, logger, proposal_nums, iou_thrs
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
            if self.year == 2007:
                ds_name = "voc07"
            else:
                ds_name = self.class_names
            if isinstance(iou_thrs, float):
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thrs,
                    dataset=ds_name,
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
                        scale_ranges=None,
                        iou_thr=iou_single,
                        dataset=ds_name,
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
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger
            )
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f"recall@{num}@{iou}"] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f"AR@{num}"] = ar[i]

        return eval_results
