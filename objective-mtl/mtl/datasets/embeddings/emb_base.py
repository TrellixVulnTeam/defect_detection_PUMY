# -*- encoding: utf-8 -*-

###########################################################
# @File    :   emb_base.py
# @Time    :   2021/10/13 22:24:42
# @Author  :   Qian Zhiming
# @Contact :   zhiming.qian@micro-i.com.cn
###########################################################

from io import BytesIO
from PIL import Image
import numpy as np

from ..data_base import DataBaseDataset
from ..data_builder import DATASETS


@DATASETS.register_module()
class EmbBaseDataset(DataBaseDataset):
    """Base dataset."""

    def __init__(self, data_cfg, pipeline_cfg, root_path, sel_index=0):
        """Initialization for dataset construction.

        Args:
            data_prefix (str): the prefix of data path
            pipeline (list): a list of dict, where each element represents
                a operation defined in transforms.
            ann_file (str | None): the annotation file. When ann_file is str,
                the subclass is expected to read from the ann_file. When ann_file
                is None, the subclass is expected to read according to data_prefix
            test_mode (bool): in train mode or test mode
        """

        super(EmbBaseDataset, self).__init__(
            data_cfg, pipeline_cfg, root_path, sel_index
        )

    def record_parser(self, feature_list, return_img=True):
        """Call when is_tfrecord is ture.

        feature_list = [(key, feature), (key, feature)]
        key is your label.txt col name
        feature is oneof bytes_list, int64_list, float_list
        """

        for key, feature in feature_list:
            # print(key)
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
        if img is None:
            img_shape = (-1, -1)
        else:
            img_shape = img.shape

        # print(file_name, img.shape)
        return {"file_name": file_name, "img": img, "ori_shape": img_shape}

    def prepare_data(self, idx):
        """Prepare data and run pipelines"""
        results = self.getitem_info(idx)
        if results is None:
            new_idx = np.random.randint(0, len(self))
            return self.prepare_data(new_idx)

        return self.pipeline(results)
