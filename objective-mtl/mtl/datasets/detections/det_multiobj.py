import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

from mtl.datasets.data_wrapper import DATASETS
from mtl.utils.io_util import list_from_file
from .det_base import DetBaseDataset


@DATASETS.register_module()
class MultiObjectDataset(DetBaseDataset):
    """Data interface for the VOC dataset"""

    class_names = (
        "person",
        "cartoon-person",
        "game-role",
        "cat",
        "dog",
        "snake",
        "bird",
        "fish",
        "rabbit",
        "monkey",
        "horse",
        "chicken",
        "pig",
        "cow",
        "sheep",
        "bicycle",
        "tricycle",
        "motorbike",
        "tractor",
        "car",
        "bus",
        "truck",
        "excavator",
        "crane",
        "train",
        "plane",
        "tank",
        "ship",
        "villa",
        "pavilion",
        "tower",
        "temple",
        "palace",
        "chair",
        "bed",
        "table",
        "sofa",
        "bench",
        "vase",
        "potted-plant",
        "bag",
        "umbrella",
        "computer",
        "television",
        "lamp",
        "mouse",
        "keyboard",
        "cell-phone",
        "dish",
        "bowl",
        "spoon",
        "bottle",
        "cup",
        "fork",
        "pot",
        "knife",
        "basketball",
        "skateboard",
        "book",
        "banana",
        "apple",
        "orange",
        "watermelon",
        "pizza",
        "cake",
    )

    def __init__(self, data_cfg, pipeline_cfg, root_path, sel_index=0):
        """Same as base detection dataset"""
        super(MultiObjectDataset, self).__init__(
            data_cfg, pipeline_cfg, root_path, sel_index
        )

        self.cat2label = {cat: i for i, cat in enumerate(self.class_names)}

        if "MIN_SIZE" in data_cfg:
            self.min_size = data_cfg.MIN_SIZE
        else:
            self.min_size = None

    def load_annotations(self):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        data_infos = []
        img_ids = list_from_file(self.ann_file)

        for file_name in img_ids:
            filename = f"{file_name}.jpg"
            xml_path = osp.join(self.anno_prefix, f"{file_name}.xml")
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
                img = Image.open(img_path)
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
                xml_path = osp.join(self.anno_prefix, f"{file_name[:-4]}.xml")
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
        if not self.is_tfrecord:
            file_name = self.data_infos[idx]["file_name"]
            width = self.data_infos[idx]["width"]
            height = self.data_infos[idx]["height"]
            xml_path = osp.join(self.anno_prefix, f"{file_name[:-4]}.xml")
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
                # truncated = int(obj.find('truncated').text)
                # difficult = int(obj.find('difficult').text)
                bnd_box = obj.find("bndbox")
                # TODO: check whether it is necessary to use int
                # Coordinates may be float type
                bbox = [
                    int(float(bnd_box.find("xmin").text)),
                    int(float(bnd_box.find("ymin").text)),
                    int(float(bnd_box.find("xmax").text)),
                    int(float(bnd_box.find("ymax").text)),
                ]

                if bbox[0] > bbox[2]:
                    tmp = bbox[0]
                    bbox[0] = bbox[2]
                    bbox[2] = tmp
                if bbox[1] > bbox[3]:
                    tmp = bbox[1]
                    bbox[1] = bbox[3]
                    bbox[3] = tmp

                if bbox[0] < 0:
                    bbox[0] = 0
                if bbox[0] >= width:
                    bbox[0] = width - 1
                if bbox[2] < 0:
                    bbox[2] = 0
                if bbox[2] >= width:
                    bbox[2] = width - 1
                if bbox[1] < 0:
                    bbox[1] = 0
                if bbox[1] >= height:
                    bbox[1] = height - 1
                if bbox[3] < 0:
                    bbox[3] = 0
                if bbox[3] >= height:
                    bbox[3] = height - 1

                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w <= 0 or h <= 0:
                    continue
                ignore = False
                if self.min_size:
                    assert not self.test_mode
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
        else:
            return self.getitem_info(idx)["ann"]
