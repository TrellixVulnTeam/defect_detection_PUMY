import argparse
import numpy as np
import onnx
import onnxruntime as rt
import torch
import os
import os.path as osp
from functools import partial

from configs import cfg
from mtl.engines.predictor import get_predictor
from mtl.utils.config_util import get_task_cfg
from mtl.utils.io_util import imread
from mtl.utils.geometric_util import imresize, imcrop_center
from mtl.utils.photometric_util import imnormalize


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def preprocess_example_input(input_config):
    """Prepare an example input image for ``generate_inputs_and_wrap_model``.

    Args:
        input_config (dict): customized config describing the example input.

    Returns:
        tuple: (one_img, one_meta), tensor of the example input image and
            meta information for the example input image.
    """
    input_path = input_config["input_path"]
    input_resize_shape = input_config["input_resize_shape"]
    input_crop_shape = input_config.get("input_crop_shape", None)
    one_img = imread(input_path)
    if "normalize_cfg" in input_config.keys():
        normalize_cfg = input_config["normalize_cfg"]
        mean = np.array(normalize_cfg["mean"], dtype=np.float32)
        std = np.array(normalize_cfg["std"], dtype=np.float32)
        one_img = imnormalize(one_img, mean, std)
    if input_crop_shape is None:
        one_img = imresize(one_img, input_resize_shape[2:][::-1]).transpose(2, 0, 1)
        input_shape = input_resize_shape
    else:
        one_img = imcrop_center(
            imresize(one_img, input_resize_shape[2:][::-1]), input_crop_shape[2:]
        ).transpose(2, 0, 1)
        input_shape = input_crop_shape
    one_img = torch.from_numpy(one_img).unsqueeze(0).float().requires_grad_(True)
    (_, C, H, W) = input_shape
    one_meta = {
        "img_shape": (H, W, C),
        "ori_shape": (H, W, C),
        "file_name": "<demo>.png",
        "scale_factor": 1.0,
        "flip": False,
    }

    return one_img, one_meta


def cls_pth2onnx(
    config_path,
    checkpoint_path,
    input_img,
    input_resize_shape,
    input_crop_shape=None,
    opset_version=11,
    show=False,
    output_file="tmp.onnx",
    verify=False,
    normalize_cfg=None,
):
    input_config = {
        "input_resize_shape": input_resize_shape,
        "input_crop_shape": input_crop_shape,
        "input_path": input_img,
        "normalize_cfg": normalize_cfg,
    }

    # prepare original model and meta for verifying the onnx model
    # get config
    get_task_cfg(cfg, config_path)

    cfg.MODEL.TRAIN_CFG = ""

    model = get_predictor(cfg, checkpoint_path, with_name=False, device="cpu")
    one_img, _ = preprocess_example_input(input_config)

    # new onnx version has fixed this problem
    # def roll(feat_input, shifts, dims):
    #     if isinstance(shifts, int):
    #         shifts = [shifts]
    #     if isinstance(dims, int):
    #         dims = [dims]
    #     assert len(shifts) == len(dims)
    #     for shift, dim in zip(shifts, dims):
    #         dim_len = feat_input.shape[dim]
    #         shift = torch.tensor(shift)
    #         if shift > 0:
    #             shift = dim_len - shift % dim_len
    #         else:
    #             shift = -shift
    #         inds = (torch.arange(dim_len) + shift) % dim_len
    #         feat_input = torch.index_select(feat_input, dim, inds)
    #     return feat_input

    # torch.roll = roll

    torch.onnx.export(
        model,
        one_img,
        output_file,
        training=False,
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=show,
        opset_version=opset_version,
    )

    print(f"Successfully exported ONNX model: {output_file}")

    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        pytorch_result = model(one_img)

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [node.name for node in onnx_model.graph.initializer]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert len(net_feed_input) == 1
        sess = rt.InferenceSession(output_file)

        onnx_results = sess.run(None, {net_feed_input[0]: one_img.detach().numpy()})

        print(pytorch_result)
        print(onnx_results)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert MTL models to ONNX")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--input-img", type=str, help="Images for input")
    parser.add_argument("--show", action="store_true", help="show onnx graph")
    parser.add_argument("--output-file", type=str, default="tmp.onnx")
    parser.add_argument("--opset-version", type=int, default=11)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify the onnx model output against pytorch output",
    )
    parser.add_argument(
        "--resize_shape",
        type=int,
        nargs="+",
        default=[224, 224],
        help="resized image size",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs="+",
        default=[123.675, 116.28, 103.53],
        help="mean value used for preprocess input data",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        default=[58.395, 57.12, 57.375],
        help="variance value used for preprocess input data",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # assert args.opset_version == 11, 'Only support opset 11 now'
    if not args.input_img:
        args.input_img = osp.join(osp.dirname(__file__), "../meta/test_data/1.jpg")

    if len(args.resize_shape) == 1:
        input_resize_shape = (1, 3, args.resize_shape[0], args.resize_shape[0])
    elif len(args.resize_shape) == 2:
        input_resize_shape = (1, 3) + tuple(args.resize_shape)
    else:
        raise ValueError("invalid input resize shape")

    assert len(args.mean) == 3
    assert len(args.std) == 3

    # print(args.shape, args.mean, args.std)
    normalize_cfg = {"mean": args.mean, "std": args.std}

    # convert model to onnx file
    cls_pth2onnx(
        args.config,
        args.checkpoint,
        args.input_img,
        input_resize_shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        normalize_cfg=normalize_cfg,
    )
