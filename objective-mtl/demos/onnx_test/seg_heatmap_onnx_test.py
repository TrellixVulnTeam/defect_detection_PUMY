import argparse
import os
import numpy as np
import onnx
import onnxruntime as rt
import torch

from mtl.utils.io_util import imread, imwrite
from mtl.utils.vis_util import imshow
from mtl.utils.geometric_util import imresize
from mtl.utils.photometric_util import imnormalize


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def preprocess_example_input(input_config):
    """Prepare an example input image for ``generate_inputs_and_wrap_model``.

    Args:
        input_config (dict): customized config describing the example input.

    Returns:
        tuple: (one_img, one_meta), tensor of the example input image and \
            meta information for the example input image.
    """
    input_path = input_config["input_path"]
    input_shape = input_config["input_shape"]
    one_img = imread(input_path)
    if "normalize_cfg" in input_config.keys():
        normalize_cfg = input_config["normalize_cfg"]
        mean = np.array(normalize_cfg["mean"], dtype=np.float32)
        std = np.array(normalize_cfg["std"], dtype=np.float32)
        one_img = imnormalize(one_img, mean, std)

    # 得到原图片宽高比例
    ori_shape = one_img.shape
    # print("ori_shape：", ori_shape)
    # print("new_shape：", new_shape)

    # print(ori_shape[0:2][::-1])
    # print(input_shape[2:][::-1])

    one_img = imresize(one_img, input_shape[2:][::-1]).transpose(2, 0, 1)

    # print("one_img imresize success!!!")
    # print(one_img.shape)

    one_img = torch.from_numpy(one_img).unsqueeze(0).float().requires_grad_(True)

    print("one_img torch from_numpy success!!!")

    (_, C, H, W) = input_shape
    one_meta = {
        "img_shape": (H, W, C),
        "ori_shape": ori_shape,
        "pad_shape": (H, W, C),
        "file_name": "<demo>.png",
        "scale_factor": 1.0,
        "flip": False,
    }

    return one_img, one_meta


def concat_line_det(
    seg_heatmap, ori_shape, line_type=0, line_avg_score=0.2, min_interval=20
):
    h = seg_heatmap.shape[0]
    w = seg_heatmap.shape[1]

    ori_h = ori_shape[0]
    ori_w = ori_shape[1]

    cmp_line_num = -1
    cmp_avg_score = 0
    concat_lines = []
    if line_type == 0:
        for i in range(10, h - 10):
            if cmp_line_num > 0 and i - cmp_line_num > min_interval:
                concat_lines.append(int(cmp_line_num * ori_h / h + 0.5))
                cmp_line_num = -1
                cmp_avg_score = 0

            tmp_avg_score = np.sum(seg_heatmap[i, :]) / h
            if cmp_avg_score > 1e-3:
                if tmp_avg_score >= cmp_avg_score:
                    cmp_line_num = i
                    cmp_avg_score = tmp_avg_score
            else:
                if tmp_avg_score >= line_avg_score:
                    cmp_line_num = i
                    cmp_avg_score = tmp_avg_score

        if cmp_avg_score >= line_avg_score:
            concat_lines.append(int(cmp_line_num * ori_h / h + 0.5))
    else:
        for j in range(10, w - 10):
            if cmp_line_num > 0 and j - cmp_line_num > min_interval:
                concat_lines.append(int(cmp_line_num * ori_w / w + 0.5))
                cmp_line_num = -1
                cmp_avg_score = 0

            tmp_avg_score = np.sum(seg_heatmap[:, j]) / w
            if cmp_avg_score > 1e-3:
                if tmp_avg_score >= cmp_avg_score:
                    cmp_line_num = j
                    cmp_avg_score = tmp_avg_score
            else:
                if tmp_avg_score >= line_avg_score:
                    cmp_line_num = j
                    cmp_avg_score = tmp_avg_score

        if cmp_avg_score >= line_avg_score:
            concat_lines.append(int(cmp_line_num * ori_w / w + 0.5))

    return concat_lines


def show_img_with_lines(
    img, concat_lines, line_type, show=True, mask_color=[255, 0, 0], out_file=None
):
    h = img.shape[0]
    w = img.shape[1]
    color_seg = np.zeros((h, w, 3), dtype=np.uint8)
    if line_type == 0:
        for idx in concat_lines:
            for i in range(w):
                color_seg[idx, i] = mask_color
                color_seg[idx - 1, i] = mask_color
                color_seg[idx + 1, i] = mask_color
    else:
        for idx in concat_lines:
            for j in range(h):
                color_seg[j, idx] = mask_color
                color_seg[j, idx - 1] = mask_color
                color_seg[j, idx + 1] = mask_color

    out_img = img * 0.5 + color_seg * 0.5
    out_img = out_img.astype(np.uint8)
    if show:
        imshow(out_img)
    if out_file is not None:
        imwrite(out_img, out_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test heatmap segmentation ONNX models"
    )
    parser.add_argument("--input-img", type=str, help="Images for input")
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--opset-version", type=int, default=11)
    parser.add_argument(
        "--shape", type=int, nargs="+", default=[512, 512], help="input image size"
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
    # 指定opset version
    assert args.opset_version == 11, "Only support opset 11 now"
    # 指定input img
    if not args.input_img:
        args.input_img = "meta/test_data/WechatIMG22.jpeg"
    # 指定input shape
    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")
    # 指定normalize_cfg
    assert len(args.mean) == 3
    assert len(args.std) == 3
    normalize_cfg = {"mean": args.mean, "std": args.std}
    # onnx模型
    onnx_file = "meta/onnx_models/seg_hrnet_heatmap_concat.onnx"
    # load onnx
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)

    # get onnx output
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert len(net_feed_input) == 1
    sess = rt.InferenceSession(onnx_file)

    # 预处理
    input_config = {
        "input_shape": input_shape,
        "input_path": args.input_img,
        "normalize_cfg": normalize_cfg,
    }
    one_img, one_meta = preprocess_example_input(input_config)

    # print("one_meta:", one_meta)

    # from mtl.cores.bbox import bbox2result
    onnx_results = sess.run(None, {net_feed_input[0]: one_img.detach().numpy()})

    seg_v = onnx_results[0][0, 0, :, :]
    seg_h = onnx_results[0][0, 1, :, :]

    ori_img = imread(args.input_img)
    if np.sum(seg_v) > np.sum(seg_h):
        # segment vertically
        line_type = 0
        concat_lines = concat_line_det(seg_v, one_meta["ori_shape"], line_type)
        show_img_with_lines(ori_img, concat_lines, line_type, True, [255, 0, 0])
    else:
        line_type = 1
        concat_lines = concat_line_det(seg_h, one_meta["ori_shape"], line_type)
        show_img_with_lines(ori_img, concat_lines, line_type, True, [0, 0, 255])
