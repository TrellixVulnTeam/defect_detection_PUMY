import torch

from configs import cfg
from mtl.utils.config_util import get_task_cfg
from mtl.models.model_builder import build_model


if __name__ == "__main__":
    pth_path = "meta/models/yolov7_static.pth"
    checkpoint = torch.load(pth_path, map_location=torch.device("cpu"))
    pth_state_dict = checkpoint["state_dict"]
    pth_key_list = []
    for key, _ in pth_state_dict.items():
        pth_key_list.append(key)

    config_path = "tasks/detections/det_coco/det_yolov7_coco.yaml"
    get_task_cfg(cfg, config_path)
    model = build_model(cfg.MODEL)
    model_state_dict = model.state_dict()
    model_key_list = []
    for key, _ in model_state_dict.items():
        model_key_list.append(key)

    for i in range(len(pth_key_list)):
        if pth_key_list[i].startswith("model.105.anchor"):
            continue
        elif pth_key_list[i].startswith("model.105.m"):
            assert (
                model_state_dict[model_key_list[i - 1]].shape
                == pth_state_dict[pth_key_list[i]].shape
            ), f"{model_key_list[i-1]}_{model_state_dict[model_key_list[i-1]].shape} vs \
                    {pth_key_list[i]}_{pth_state_dict[pth_key_list[i]].shape}"
            model_state_dict[model_key_list[i - 1]] = pth_state_dict[pth_key_list[i]]
        else:
            assert (
                model_state_dict[model_key_list[i + 1]].shape
                == pth_state_dict[pth_key_list[i]].shape
            ), f"{model_key_list[i+1]}_{model_state_dict[model_key_list[i+1]].shape} vs \
                    {pth_key_list[i]}_{pth_state_dict[pth_key_list[i]].shape}"
            model_state_dict[model_key_list[i + 1]] = pth_state_dict[pth_key_list[i]]

    save_path = "meta/models/yolov7_mtl.pth"
    torch.save({"state_dict": model_state_dict}, save_path)
    print("Succeed!")
