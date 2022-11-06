import os.path as osp
from PIL import Image
import numpy as np
from io import BytesIO

from ..data_builder import DATASETS
from .emb_base import EmbBaseDataset


@DATASETS.register_module()
class SslPretrainDataset(EmbBaseDataset):
    """SSL pretraining dataset"""

    def load_annotations(self):
        """Load data_infos"""
        data_infos = []
        with open(self.ann_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            tmp_str = line.strip()
            str_list = tmp_str.split()
            if len(str_list) >= 4:
                file_name = str_list[0]
                height = int(str_list[1])
                width = int(str_list[2])
                data_infos.append(dict(file_name=file_name, ori_shape=(height, width)))

        return data_infos

    def record_parser(self, feature_list):
        """Call when is_tfrecord is ture.

        feature_list = [(key, feature), (key, feature)]
        key is your label.txt col name
        feature is oneof bytes_list, int64_list, float_list
        """
        for key, feature in feature_list:
            # print(key)
            # for image file col
            if key == "image_name":
                file_name = feature.bytes_list.value[0].decode("UTF-8", "strict")
            elif key == "image":
                image_raw = feature.bytes_list.value[0]
                pil_img = Image.open(BytesIO(image_raw)).convert("RGB")
                img = np.array(pil_img).astype(np.float32)

        # print(file_name, img.shape)
        return {
            "file_name": file_name,
            "img": img,
            "ori_shape": (img.shape[0], img.shape[1]),
        }

    def prepare_data(self, idx):
        """Prepare data and run pipelines"""
        if self.is_tfrecord:
            results = self.getitem_info(idx)
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
                "ori_shape": self.data_infos[idx]["ori_shape"],
            }

        return self.pipeline(results)
