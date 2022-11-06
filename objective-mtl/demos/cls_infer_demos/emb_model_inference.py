import os
import argparse

from configs import cfg
from mtl.utils.config_util import get_task_cfg
from mtl.engines.predictor import get_predictor, inference_predictor
from mtl.utils.io_util import imread


def parse_args():
    parser = argparse.ArgumentParser(description="infer a model")
    parser.add_argument("task_config", help="test config file")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--img_dir", type=str, help="the dir of images to be inferred")
    parser.add_argument(
        "--device", type=str, default="cpu", help="the dir of images to be inferred"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # get config
    get_task_cfg(cfg, args.task_config)
    # get model
    model = get_predictor(cfg, args.checkpoint, device=args.device)

    for img_file in os.listdir(args.img_dir):
        img_path = os.path.join(args.img_dir, img_file)
        img_data = imread(img_path, backend="pillow")
        result = inference_predictor(cfg, model, img_data)[0]

        print(img_file)
        print(result)
        print("\n")
