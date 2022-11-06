import sys
import torch

from configs import cfg
from mtl.utils.config_util import get_task_cfg
from mtl.models.model_builder import build_model


def show_checkpoint(snapshot):
    checkpoint = torch.load(snapshot, map_location=torch.device("cpu"))
    if "state_dict" in checkpoint:
        print("=> state_dict")
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        print("=> model")
        state_dict = checkpoint["model"]
    else:
        print("=> raw")
        state_dict = checkpoint
    for k, v in state_dict.items():
        print(k, ": ", v.shape)


if __name__ == "__main__":
    arg_str = sys.argv[1]
    if arg_str.endswith("yaml"):
        get_task_cfg(cfg, arg_str)
        model = build_model(cfg.MODEL)
        state_dict = model.state_dict()
        for k, v in state_dict.items():
            print(k, ": ", v.shape)
    else:
        show_checkpoint(sys.argv[1])
