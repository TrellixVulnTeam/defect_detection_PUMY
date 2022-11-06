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
from mtl.utils.geometric_util import imresize
from mtl.utils.photometric_util import imnormalize
from mtl.cores.ops import multiclass_nms
from mtl.cores.bbox import bbox2result


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def generate_inputs_and_wrap_model(cfg_node, checkpoint_path, one_meta):
    """Prepare sample input and wrap model for ONNX export.

    The ONNX export API only accept args, and all inputs should be
    torch.Tensor or corresponding types (such as tuple of tensor).
    So we should call this function before exporting. This function will:

    1. generate corresponding inputs which are used to execute the model.
    2. Wrap the model's forward function.

    Args:
        config_path (str): the config for the model we want to export to ONNX
        checkpoint_path (str): Path to the corresponding checkpoint
        input_config (dict): the exactly data in this dict depends on the
            framework. For Det, we may pass the real img path, or the NMS will
            return None as there is no legal bbox.

    Returns:
        tuple: (model) wrapped model which can be called by
            model(*tensor_data) and a list of inputs which are used to execute
            the model while exporting.
    """
    model = get_predictor(cfg_node, checkpoint_path, device="cpu")
    model.forward = partial(model.forward, img_metas=[one_meta])

    return model


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
    one_img = imresize(one_img, input_shape[2:][::-1]).transpose(2, 0, 1)
    one_img = torch.from_numpy(one_img).unsqueeze(0).float().requires_grad_(True)
    (_, C, H, W) = input_shape
    one_meta = {
        "img_shape": (H, W, C),
        "ori_shape": (H, W, C),
        "pad_shape": (H, W, C),
        "filename": "<demo>.png",
        "scale_factor": 1.0,
        "flip": False,
    }

    return one_img, one_meta


def det_onestage_pth2onnx(
    config_path,
    checkpoint_path,
    input_img,
    input_shape,
    opset_version=11,
    show=False,
    output_file="tmp.onnx",
    verify=False,
    normalize_cfg=None,
    is_dynamic=False,
    simplify=False
):
    input_config = {
        "input_shape": input_shape,
        "input_path": input_img,
        "normalize_cfg": normalize_cfg,
    }

    # prepare original model and meta for verifying the onnx model
    # get config
    get_task_cfg(cfg, config_path)
    num_classes = cfg.MODEL.BBOX_HEAD["num_classes"]  # two stage models are not support currently

    cfg.MODEL.TRAIN_CFG = ""

    orig_model = get_predictor(cfg, checkpoint_path, device="cpu")
    one_img, one_meta = preprocess_example_input(input_config)

    model = generate_inputs_and_wrap_model(
        cfg, checkpoint_path, one_meta
    )

    # Troch新版本解决了该问题
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

    if is_dynamic:
        torch.onnx.export(
            model,
            one_img,
            output_file,
            opset_version=opset_version,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=['INPUT__0'],
            output_names=['OUTPUT__0'],
            dynamic_axes={
                'INPUT__0': {0: 'batch', 2: 'height', 3: 'width'},
                'OUTPUT__0': {0: 'batch', 1: 'anchors'}
            }
        )
    else:
        torch.onnx.export(
            model,
            one_img,
            output_file,
            export_params=True,
            keep_initializers_as_inputs=True,
            verbose=show,
            opset_version=opset_version,
        )
    # Simplify
    if simplify:
        try:
            # check_requirements(('onnx-simplifier',))
            import onnxsim

            print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=is_dynamic,
                input_shapes={'INPUT__0': list(one_img["img_shape"])} if is_dynamic else None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, output_file)
        except Exception as e:
            print(f'simplifier failure: {e}')

    model.forward = orig_model.forward
    print(f"Successfully exported ONNX model: {output_file}")

    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        pytorch_result = model(one_img, img_metas=[one_meta])

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [node.name for node in onnx_model.graph.initializer]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert len(net_feed_input) == 1
        sess = rt.InferenceSession(output_file)

        output_res = sess.run(None, {net_feed_input[0]: one_img.detach().numpy()})
        # for output in output_res:
        #     print(output.shape)
        # det_bboxes, det_labels = out_list
        ml_bboxes = output_res[0][0, :, :4]
        ml_conf_scores = output_res[0][0, :, 4]
        ml_cls_scores = output_res[0][0, :, 5:]

        # # only compare a part of result
        conf_thr = cfg.MODEL.TEST_CFG.get("conf_thr", -1)
        conf_inds = np.where(ml_conf_scores > conf_thr)
        ml_bboxes = ml_bboxes[conf_inds]
        ml_cls_scores = ml_cls_scores[conf_inds]
        ml_conf_scores = ml_conf_scores[conf_inds]

        det_bboxes, det_labels = multiclass_nms(
            torch.from_numpy(ml_bboxes),
            torch.from_numpy(ml_cls_scores),
            cfg.MODEL.TEST_CFG["score_thr"],
            cfg.MODEL.TEST_CFG["nms"],
            cfg.MODEL.TEST_CFG["max_per_img"],
            score_factors=ml_conf_scores,
        )

        onnx_results = bbox2result(det_bboxes, det_labels, num_classes)
        # assert np.allclose(
        #     pytorch_result[0][38][0][:4], onnx_results[38][0]
        #     [:4]), 'The outputs are different between Pytorch and ONNX'
        # print('The numerical values are the same between Pytorch and ONNX')

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
        "--is-dynamic",
        action="store_true",
        help="whether using dynamic input and output",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="whether using onnx simplify",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify the onnx model output against pytorch output",
    )
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
    # assert args.opset_version == 11, 'Only support opset 11 now'
    if not args.input_img:
        args.input_img = osp.join(
            osp.dirname(__file__), "../meta/test_data/a0519qvbyom_001.jpg"
        )

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")

    assert len(args.mean) == 3
    assert len(args.std) == 3

    # print(args.shape, args.mean, args.std)

    normalize_cfg = {"mean": args.mean, "std": args.std}

    # convert model to onnx file
    det_onestage_pth2onnx(
        args.config,
        args.checkpoint,
        args.input_img,
        input_shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        normalize_cfg=normalize_cfg,
        is_dynamic=args.is_dynamic,
        simplify=args.simplify
    )
