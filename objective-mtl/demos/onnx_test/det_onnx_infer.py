import argparse
from enum import Enum
import numpy as np
import onnx
import onnxruntime as rt
import torch
import os
import os.path as osp
from pathlib import Path
import cv2
from PIL import Image
from torchvision.ops import nms


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class_names = ['label_'+str(i) for i in range(20)]

cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def _pillow2array(img, flag="color", channel_order="bgr"):
    """Convert a pillow image to numpy array.
    Args:
        img (:obj:`PIL.Image.Image`): The image loaded using PIL
        flag (str): Flags specifying the color type of a loaded image,
            candidates are 'color', 'grayscale' and 'unchanged'.
            Default to 'color'.
        channel_order (str): The channel order of the output image array,
            candidates are 'bgr' and 'rgb'. Default to 'bgr'.
    Returns:
        np.ndarray: The converted numpy array
    """
    channel_order = channel_order.lower()
    if channel_order not in ["rgb", "bgr"]:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == "unchanged":
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if img.mode != "RGB":
            if img.mode != "LA":
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert("RGB")
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert("RGBA")
                img = Image.new("RGB", img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        if flag == "color":
            array = np.array(img)
            if channel_order != "rgb":
                array = array[:, :, ::-1]  # RGB to BGR
        elif flag == "grayscale":
            img = img.convert("L")
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale" or "unchanged", ' f"but got {flag}"
            )
    return array


def imread(img_or_path, flag="color", channel_order="bgr"):
    """Read an image.

    Args:
        img_or_path (ndarray or str or Path): Either a numpy array or str or
            pathlib.Path. If it is a numpy array (loaded image), then
            it will be returned as is.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
            Note that the `turbojpeg` backened does not support `unchanged`.
        channel_order (str): Order of channel, candidates are `bgr` and `rgb`.

    Returns:
        ndarray: Loaded image array.
    """
    if isinstance(img_or_path, Path):
        img_or_path = str(img_or_path)

    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif isinstance(img_or_path, str):
        check_file_exist(img_or_path, f"img file does not exist: {img_or_path}")

        img = Image.open(img_or_path)
        img = _pillow2array(img, flag, channel_order)
        return img
    else:
        raise TypeError(
            '"img" must be a numpy array or a str or ' "a pathlib.Path object"
        )


def imresize(
    img, size, return_scale=False, interpolation="bilinear"
):
    """Resize image to a given size.
    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = img.shape[:2]
    if min(size) == -1:
        scale = float(max(size)) / min(h, w)
        size = int(w * scale), int(h * scale)

    resized_img = cv2.resize(
        img, tuple(size), interpolation=cv2_interp_codes[interpolation]
    )

    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def imnormalize_(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std.
    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.
    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def imnormalize(img, mean, std, to_rgb=True):
    """Normalize an image with mean and std.
    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.
    Returns:
        ndarray: The normalized image.
    """
    img = img.copy().astype(np.float32)
    return imnormalize_(img, mean, std, to_rgb)


class Color(Enum):
    """An enum that defines common colors.
    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


def color_val(color):
    """Convert various input to color tuples.
    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.
    """
    if isinstance(color, str):
        return Color[color].value
    elif isinstance(color, Color):
        return color.value
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    elif isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError(f"Invalid type for color: {type(color)}")


def imshow(img, win_name="", wait_time=0):
    """Show an image.
    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    if isinstance(img, np.ndarray):
        cv2.imshow(win_name, img)
    elif isinstance(img, str):
        cv2.imshow(win_name, imread(img))

    print("showing ...")

    if wait_time == 0:  # prevent from hangning if windows was closed
        while True:
            ret = cv2.waitKey(0)
            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)


def imshow_det_bboxes(
    img,
    bboxes,
    labels,
    class_names=None,
    score_thr=0,
    bbox_color="green",
    text_color="green",
    thickness=1,
    font_scale=0.5,
    show=True,
    win_name="",
    wait_time=0,
):
    """Draw bboxes and class labels (with scores) on an image.
    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    img = np.ascontiguousarray(img)
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[label] if class_names is not None else f"cls {label}"
        if len(bbox) > 4:
            label_text += f"|{bbox[-1]:.02f}"
        cv2.putText(
            img,
            label_text,
            (bbox_int[0], bbox_int[1] - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            text_color,
        )

    if show:
        imshow(img, win_name, wait_time)

    return img


def multiclass_nms(
    multi_bboxes,
    multi_scores,
    score_thr,
    iou_threshold,
    max_num=-1,
    score_factors=None,
    has_background=True,
):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_threshold (float): IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    # exclude background category
    if has_background:
        num_classes = multi_scores.size(1) - 1
        scores = multi_scores[:, :-1]
    else:
        num_classes = multi_scores.size(1)
        scores = multi_scores

    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # as ONNX does not support repeat now,
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes, torch.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1)
    ).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                "[ONNX Error] Can not record NMS "
                "as it has not been executed this time"
            )
        return bboxes, labels

    # dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
    max_coordinate = bboxes.max()
    offsets = labels.to(bboxes) * (max_coordinate + torch.tensor(1).to(bboxes))
    boxes_for_nms = bboxes + offsets[:, None]

    if boxes_for_nms.shape[0] < 10000:
        inds = nms(boxes_for_nms, scores, iou_threshold)
        dets = torch.cat((boxes_for_nms[inds], scores[inds].reshape(-1, 1)), dim=1)
        bboxes = bboxes[inds]
        scores = dets[:, -1]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(labels):
            mask = (labels == id).nonzero(as_tuple=False).view(-1)
            inds = nms(boxes_for_nms[mask], scores[mask], iou_threshold)
            dets = torch.cat((boxes_for_nms[mask][inds], scores[mask][inds].reshape(-1, 1)), dim=1)
            total_mask[mask[inds]] = True

        inds = total_mask.nonzero(as_tuple=False).view(-1)
        inds = inds[scores[inds].argsort(descending=True)]
        bboxes = bboxes[inds]
        scores = scores[inds]

    dets = torch.cat([bboxes, scores[:, None]], -1)
    keep = inds

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]


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
    new_shape = input_shape
    print("ori_shape: ", ori_shape)
    print("new_shape: ", new_shape)

    one_img = imresize(one_img, input_shape[2:][::-1]).transpose(2, 0, 1)

    print("one_img imresize success!!!")
    print(one_img.shape)

    one_img = torch.from_numpy(one_img).unsqueeze(0).float().requires_grad_(True)

    print("one_img torch from_numpy success!!!")

    _, C, H, W = input_shape
    one_meta = {
        "img_shape": (H, W, C),
        "ori_shape": ori_shape,
        "pad_shape": (H, W, C),
        "file_name": "<demo>.png",
        "scale_factor": 1.0,
        "flip": False,
    }

    return one_img, one_meta


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


def parse_args():
    parser = argparse.ArgumentParser(description="Test detection ONNX models")
    parser.add_argument("--input-img", type=str, help="Images for input")
    parser.add_argument("--onnx-model", type=str, help="Images for input")
    parser.add_argument("--opset-version", type=int, default=11)
    parser.add_argument(
        "--shape", type=int, nargs="+", default=[416, 416], help="input image size"
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs="+",
        default=[0, 0, 0],
        help="mean value used for preprocess input data",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        default=[255, 255, 255],
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
        args.input_img = osp.join(
            osp.dirname(__file__), "../../meta/test_data/000001.jpg"
        )
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
    output_file = args.onnx_model

    # 预处理
    input_config = {
        "input_shape": input_shape,
        "input_path": args.input_img,
        "normalize_cfg": normalize_cfg,
    }
    one_img, one_meta = preprocess_example_input(input_config)

    # print("one_meta:", one_meta)

    # load onnx
    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model)

    # get onnx output
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert len(net_feed_input) == 1
    # sess = rt.InferenceSession(output_file) for cpu
    sess = rt.InferenceSession(
        output_file,
        providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

    # from mtl.cores.bbox import bbox2result
    output_res = sess.run(None, {net_feed_input[0]: one_img.detach().numpy()})
    ml_bboxes = output_res[0][0, :, :4]
    ml_conf_scores = output_res[0][0, :, 4]
    ml_cls_scores = output_res[0][0, :, 5:]

    # # only compare a part of result
    conf_thr = 0.005
    conf_inds = np.where(ml_conf_scores > conf_thr)
    ml_bboxes = ml_bboxes[conf_inds]
    ml_cls_scores = ml_cls_scores[conf_inds]
    ml_conf_scores = ml_conf_scores[conf_inds]

    iou_threshold = 0.45

    det_bboxes, det_labels = multiclass_nms(
        torch.from_numpy(ml_bboxes),
        torch.from_numpy(ml_cls_scores),
        0.05,
        iou_threshold,
        100,
        score_factors=ml_conf_scores,
    )
    # only compare a part of result
    bbox_results = bbox2result(det_bboxes, det_labels, len(class_names))
    # print('bbox_results:', bbox_results)

    bboxes = np.vstack(bbox_results)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_results)
    ]
    labels = np.concatenate(labels)

    score_thr = 0.3
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    h_scale = one_meta["ori_shape"][0] / one_meta["img_shape"][0]
    w_scale = one_meta["ori_shape"][1] / one_meta["img_shape"][1]

    bboxes[:, 0] = bboxes[:, 0] * w_scale
    bboxes[:, 1] = bboxes[:, 1] * h_scale
    bboxes[:, 2] = bboxes[:, 2] * w_scale
    bboxes[:, 3] = bboxes[:, 3] * h_scale

    print("bboxes:", bboxes)
    print("labels:", labels)

    imshow_det_bboxes(args.input_img, bboxes, labels, class_names)
