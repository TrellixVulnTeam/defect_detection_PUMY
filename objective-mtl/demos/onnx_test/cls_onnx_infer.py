# -*- encoding: utf-8 -*-

###########################################################
# @File    :   cls_onnx_online_test.py
# @Time    :   2021/10/14 15:20:56
# @Content :   Inference for multilabel with onnx model
# @Author  :   Qian Zhiming
# @Contact :   zhiming.qian@micro-i.com.cn
###########################################################

import os
import io
import onnx
import onnxruntime
import cv2
import time
from PIL import Image
import numpy as np
from torchvision import transforms

from mtl.datasets.transforms.common_transforms import imnormalize

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class OnnxImageClassifier:
    def __init__(self, onnx_model_path):
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)

        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [node.name for node in onnx_model.graph.initializer]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert len(net_feed_input) == 1
        self.input_layer_name = net_feed_input[0]

        sess = onnxruntime.InferenceSession(onnx_model_path)

        self.sess = sess

    def _preprocess_bk(self, input_image):
        if isinstance(input_image, str):
            img = Image.open(input_image)
        elif isinstance(input_image, bytes):
            img = Image.open(io.BytesIO(input_image))
        if not img.mode == "RGB":
            img = img.convert("RGB")

        # w, h = img.shape[:2]
        h, w = img.size[:2]
        long_side = 384
        if w < h:
            height = long_side
            width = int(long_side * w / h)
        else:
            width = long_side
            height = int(long_side * h / w)

        w_pad = (384 - width) // 2
        h_pad = (384 - height) // 2
        left, top, right, bottom = (
            w_pad,
            h_pad,
            384 - width - w_pad,
            384 - height - h_pad,
        )
        padding = (left, top, right, bottom)

        transform = transforms.Compose(
            [
                transforms.Resize([height, width]),
                transforms.Pad(padding=padding, fill=-1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.491, 0.482, 0.4465], std=[0.2023, 0.1994, 0.201]
                ),
            ]
        )
        # pil_img = transforms.functional.to_pil_image(np.array(img))
        # pil_img.save("test.jpg")

        img = transform(img).unsqueeze(0)
        return img

    def _preprocess(self, input_image, normalize_cfg=None, resize=384):
        """Prepare an example input image for ``generate_inputs_and_wrap_model``."""
        if isinstance(input_image, str):
            img = Image.open(input_image)
        elif isinstance(input_image, bytes):
            img = Image.open(io.BytesIO(input_image))
        if not img.mode == "RGB":
            img = img.convert("RGB")
        img = np.array(img).astype(np.float32)
        h, w = img.shape[:2]
        long_side = resize
        if w < h:
            height = long_side
            width = int(long_side * w / h)
        else:
            width = long_side
            height = int(long_side * h / w)

        left, top, right, bottom = (0, 0, resize - width, resize - height)
        img = cv2.resize(img, (width, height), dst=None, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=-1
        )
        img = imnormalize(
            img,
            np.array(normalize_cfg["mean"], dtype=np.float32),
            np.array(normalize_cfg["std"], dtype=np.float32),
            True,
        )

        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img).unsqueeze(0)
        return img

    def predict_raw(self, input_image):
        normalize_cfg = {
            "mean": [125.307, 122.961, 113.8575],
            "std": [51.5865, 50.847, 51.255],
        }
        input_data = self._preprocess(input_image, normalize_cfg)
        out = self.sess.run(None, {self.input_layer_name: input_data.detach().numpy()})
        return out


if __name__ == "__main__":
    ONNX_MODEL_PATH = "meta/onnx_models/xxx.onnx"
    model = OnnxImageClassifier(ONNX_MODEL_PATH)
    test_images = ["meta/test_data/1.jpg"]
    for test_image in test_images:
        start_time = time.time()
        for _ in range(100):
            ret = model.predict_raw(test_image)
        print(time.time() - start_time)
