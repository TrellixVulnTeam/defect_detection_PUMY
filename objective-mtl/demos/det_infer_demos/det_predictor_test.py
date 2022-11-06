# -*- coding: utf-8 -*-
# @Time    : 2020/3/16 18:00
# @Author  : zhiming.qian
# @Email   : zhiming.qian@micro-i.com.cn

import os

from configs import cfg
from mtl.utils.config_util import get_task_cfg
from mtl.engines.predictor import get_predictor, inference_predictor
from mtl.engines.predictor import show_predictor_result


if __name__ == "__main__":
    """Test demo for detection models"""

    print("Infer the detection results from an image.")
    image_file_name = "000001.jpg"
    image_dir = "meta/test_data"
    image_path = os.path.join(image_dir, image_file_name)
    save_dir = "meta/test_res"
    save_path = os.path.join(save_dir, image_file_name)  # None for no saving

    task_config_path = "tasks/detections/det_coco/det_yolov7_coco.yaml"
    checkpoint_path = "meta/models/yolov7_mtl.pth"

    with_show = False
    show_score_thr = 0.2

    # get config
    get_task_cfg(cfg, task_config_path)

    # get model
    model = get_predictor(cfg, checkpoint_path, device="cpu")

    # get result
    result = inference_predictor(cfg, model, image_path)

    # show and save result
    show_predictor_result(
        model, image_path, result[0], show_score_thr, with_show, save_path
    )
