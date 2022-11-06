import numpy as np
from PIL import Image
from io import BytesIO

from mtl.datasets.data_wrapper import DATASETS
from .det_base import DetBaseDataset


@DATASETS.register_module()
class OneObjectDataset(DetBaseDataset):
    """Data interface for the VOC dataset"""

    class_names = ("object",)

    def __init__(self, data_cfg, pipeline_cfg, root_path, sel_index=0):
        """Same as base detection dataset"""
        super(OneObjectDataset, self).__init__(
            data_cfg, pipeline_cfg, root_path, sel_index
        )

        self.cat2label = {cat: i for i, cat in enumerate(self.class_names)}

        if "MIN_SIZE" in data_cfg:
            self.min_size = data_cfg.MIN_SIZE
        else:
            self.min_size = None

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

        for i in range(len(obj_xmin)):
            label = 0
            bbox = [
                int(float(obj_xmin[i]) + 0.5),
                int(float(obj_ymin[i]) + 0.5),
                int(float(obj_xmax[i]) + 0.5),
                int(float(obj_ymax[i]) + 0.5),
            ]

            ignore = False
            if self.min_size:
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
