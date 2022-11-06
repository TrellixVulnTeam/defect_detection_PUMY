from io import BytesIO
from PIL import Image
import numpy as np

from ..classifications.cls_base import ClsBaseDataset
from ..data_builder import DATASETS


@DATASETS.register_module()
class SslClsPretrainDataset(ClsBaseDataset):
    """SSL pretraining dataset"""

    def record_parser(self, feature_list, return_img=True):
        """Call when is_tfrecord is ture.

        feature_list = [(key, feature), (key, feature)]
        key is your label.txt col name
        feature is oneof bytes_list, int64_list, float_list
        """

        for key, feature in feature_list:
            # print(key)
            # for image file col
            if key == "name":
                file_name = feature.bytes_list.value[0].decode("UTF-8", "strict")
            elif key == "image":
                if return_img:
                    image_raw = feature.bytes_list.value[0]
                    pil_img = Image.open(BytesIO(image_raw)).convert("RGB")
                    img = np.array(pil_img).astype(np.float32)
                else:
                    img = None
            elif key == "label":
                gt_label = feature.int64_list.value[0]

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

        results = self.getitem_info(idx, return_img=True)
        if not self.is_tfrecord:
            results["gt_label"] = self.data_infos[idx]["label"]

        return self.pipeline(results)
