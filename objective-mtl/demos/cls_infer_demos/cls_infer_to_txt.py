import argparse
import glob
import os
import os.path as osp
import numpy as np
from PIL import Image
import tqdm

from configs import cfg
from mtl.utils.config_util import get_task_cfg
from mtl.engines.predictor import get_predictor, inference_predictor


def parse_args():
    parser = argparse.ArgumentParser(description="infer a model")
    parser.add_argument("task_config", help="test config file")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--img_dir", type=str, help="the dir of images to be inferred")
    parser.add_argument("--out_file", type=str)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    get_task_cfg(cfg, args.task_config)

    model = get_predictor(cfg, args.checkpoint)

    out_dir = osp.dirname(args.out_file)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.out_file, "w") as f:
        for img_file in tqdm.tqdm(glob.glob(f"{args.img_dir}/*/*/*.jpg")):
            try:
                img = Image.open(img_file).convert("RGB")
            except:
                continue
            img = np.array(img).astype(np.float32)
            result = inference_predictor(cfg, model, img)[0]
            f.write(f"{img_file} {result[1]}\n")
            f.flush()


if __name__ == "__main__":
    main()
