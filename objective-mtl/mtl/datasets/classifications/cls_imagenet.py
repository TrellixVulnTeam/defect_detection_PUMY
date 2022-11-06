import os
from io import BytesIO
from PIL import Image
import numpy as np

from ..data_builder import DATASETS
from .cls_base import ClsBaseDataset


@DATASETS.register_module()
class ImageNetDataset(ClsBaseDataset):
    """Cls imagenet classification"""

    class_names = None

    def load_annotations(self):
        """Load data_infos"""
        label_dict = {}
        with open(self.ann_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            tmp_str = line.strip()
            str_list = tmp_str.split()
            if len(str_list) >= 2:
                dir_name = str_list[0]
                label = int(str_list[1])
                label_dict[dir_name] = label

        data_infos = []
        for sub_dir in os.listdir(self.data_prefix):
            if sub_dir.startswith("."):
                continue
            label = label_dict[sub_dir]
            for f_img in os.listdir(os.path.join(self.data_prefix, sub_dir)):
                if f_img.endswith(".jpg"):
                    data_infos.append(
                        {"file_name": os.path.join(sub_dir, f_img), "label": label}
                    )

        return data_infos

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
                    img = np.array(pil_img)
                else:
                    img = None
            elif key == "label":
                gt_label = feature.int64_list.value[0]

        return {"file_name": file_name, "img": img, "gt_label": gt_label}

    def __getitem__(self, idx):
        if self.sample_rate < 0.999:
            index = self.convert_idx[idx]
            img_data = self.prepare_data(index)
        else:
            img_data = self.prepare_data(idx)
        img_data["idx"] = idx
        return img_data
