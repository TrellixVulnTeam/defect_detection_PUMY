# -*- coding: utf-8 -*-
# @Time    : 2020/12/02 22:00
# @Author  : zhiming.qian
# @Email   : zhiming.qian@micro-i.com.cn
# @File    : block_builder.py

from mtl.utils.reg_util import Registry, build_model_from_cfg, build_module_from_dict
from mtl.utils.checkpoint_util import load_checkpoint

# --------------------------------------------------------------------------- #
# Registries for blocks
# --------------------------------------------------------------------------- #
CLASSIFIERS = Registry("classifier")
DETECTORS = Registry("detector")
SEGMENTORS = Registry("segmentor")
REGRESSORS = Registry("regressor")
POSERS = Registry("poser")
EMBEDDERS = Registry("embedder")
MULTITASKS = Registry("multitask")
AUGMENTORS = Registry("augmentor")


def build_model(cfg):
    """Build model."""
    if "TYPE" not in cfg:
        raise KeyError("cfg must have key 'TYPE' to define model type")

    model_type = cfg.TYPE

    if model_type == "cls":
        return build_model_from_cfg(cfg, CLASSIFIERS)
    elif model_type == "det":
        return build_model_from_cfg(cfg, DETECTORS)
    elif model_type == "seg":
        return build_model_from_cfg(cfg, SEGMENTORS)
    elif model_type == "reg":
        return build_model_from_cfg(cfg, REGRESSORS)
    elif model_type == "pos":
        return build_model_from_cfg(cfg, POSERS)
    elif model_type == "emb":
        return build_model_from_cfg(cfg, EMBEDDERS)
    elif model_type == "mtl":
        return build_model_from_cfg(cfg, MULTITASKS)
    else:
        raise TypeError(f"No type for the task {model_type}")


def build_augmentor(cfg_dict):
    return build_module_from_dict(cfg_dict, AUGMENTORS)


def get_initialized_model(config, checkpoint=None, device="cuda:0"):
    """Initialize a pose model from config file.

    Args:
        config (str or :obj:`CfgNode`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    config.MODEL.PRETRAINED_MODEL_PATH = ""
    model = build_model(config.MODEL)

    # load model checkpoint
    if checkpoint is not None:
        map_loc = "cpu" if device == "cpu" else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)

    model.to(device)
    model.eval()
    return model
