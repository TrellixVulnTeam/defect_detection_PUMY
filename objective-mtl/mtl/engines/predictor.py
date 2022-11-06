import os
import numpy as np
import torch
from yacs.config import CfgNode

from mtl.utils.runtime_util import collate
from mtl.utils.checkpoint_util import load_checkpoint
from mtl.utils.io_util import imread
from mtl.utils.data_util import get_classes
from mtl.models.model_builder import build_model
from mtl.datasets.transforms import Compose


def get_pipeline_list(pipeline_cfg):
    """Get the list of configures for constructing pipelines

    Note:
        self.pipeline is a CfgNode

    Returns:
        list[dict]: list of dicts with types and parameters for constructing pipelines.
    """
    pipeline_list = []
    for k_t, v_t in pipeline_cfg.items():
        pipeline_item = {}
        if len(v_t) > 0:
            if not isinstance(v_t, CfgNode):
                raise TypeError("pipeline items must be a CfgNode")
        pipeline_item["type"] = k_t

        for k_a, v_a in v_t.items():
            if isinstance(v_a, CfgNode):
                pipeline_item[k_a] = []
                for sub_kt, sub_vt in v_a.items():
                    sub_item = {}
                    if len(sub_vt) > 0:
                        if not isinstance(sub_vt, CfgNode):
                            raise TypeError("transform items must be a CfgNode")
                    sub_item["type"] = sub_kt
                    for sub_ka, sub_va in sub_vt.items():
                        if isinstance(sub_va, CfgNode):
                            raise TypeError("Only support two built-in layers")
                        sub_item[sub_ka] = sub_va
                    pipeline_item[k_a].append(sub_item)
            else:
                pipeline_item[k_a] = v_a
        pipeline_list.append(pipeline_item)

    return pipeline_list


def get_predictor(config, checkpoint=None, with_name=True, device="cuda:0"):
    """Initialize a detector from config file.

    Args:
        config (CfgNode): cfg
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    config.MODEL.PRETRAINED_MODEL_PATH = ""
    model = build_model(config.MODEL)
    if checkpoint is not None:
        map_loc = "cpu" if device == "cpu" else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)

        if with_name:
            if config.MODEL.TYPE != "emb":
                if hasattr(checkpoint, "meta"):
                    if "class_names" in checkpoint["meta"]:
                        model.class_names = checkpoint["meta"]["class_names"]
                else:
                    cls_names = config.DATA.TRAIN_DATA.CLASS_NAMES
                    if cls_names.endswith('.txt'):
                        model.class_names = get_classes(
                            os.path.join(config.DATA.ROOT_PATH, cls_names)
                        )
                    else:
                        model.class_names = get_classes(cls_names)

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    return model


def inference_predictor(config, model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    device = next(model.parameters()).device  # model device
    # prepare data
    if isinstance(img, np.ndarray):
        img_data = dict(img=img)
    else:
        img_data = dict(img=imread(img, channel_order="rgb", backend="pillow"))
    pipeline_list = get_pipeline_list(config.DATA.TEST_TRANSFORMS)

    # build the data pipeline
    test_pipeline = Compose(pipeline_list)
    data = test_pipeline(img_data)
    data = collate([data])

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data["img"] = data["img"].float().to(device)
    else:
        # just get the actual data
        data["img_metas"] = data["img_metas"]

    # forward the model
    with torch.no_grad():
        result = model(rescale=True, **data)
    return result


def show_predictor_result(
    model, img, result, score_thr=0.3, with_show=True, save_path=None
):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, "module"):
        model = model.module
    img = model.show_result(
        img, result, score_thr=score_thr, show=with_show, out_file=save_path
    )
