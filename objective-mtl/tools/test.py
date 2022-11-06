# -*- coding: utf-8 -*-
# @Time    : 2020/12/1 21:00
# @Author  : zhiming.qian
# @Email   : zhiming.qian@micro-i.com.cn

import argparse
import os
import torch
from torch.nn.parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from configs import cfg
from mtl.cores.ops import fuse_conv_bn
from mtl.utils.parallel_util import init_dist
from mtl.utils.misc_util import get_dist_info
from mtl.utils.checkpoint_util import load_checkpoint
from mtl.engines.evaluator import multi_device_test, single_device_test
from mtl.datasets.data_builder import build_dataloader, build_dataset
from mtl.models.model_builder import build_model
from mtl.utils.config_util import get_task_cfg, get_dataset_global_args
from mtl.utils.config_util import get_dict_from_list, convert_to_dict
from mtl.utils.io_util import obj_dump


def parse_args():
    parser = argparse.ArgumentParser(description="test (and eval) a model")
    parser.add_argument("task_config", help="test config file")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "map", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument(
        "--show-dir", help="directory where painted images will be saved"
    )
    parser.add_argument("--save-txt-path", help="txt path where results will be saved")
    parser.add_argument(
        "--show-score-thr",
        type=float,
        default=0.3,
        help="score threshold (default: 0.3)",
    )
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def eval_dataset(args, outputs, dataset):
    if args.out:
        print(f"\nwriting results to {args.out}")
        obj_dump(outputs, args.out)
    kwargs = {}
    if args.format_only:
        dataset.format_results(outputs, **kwargs)
    if args.eval:
        if len(cfg.RUNTIME.EVALUATION) > 0:
            eval_kwargs = cfg.RUNTIME.EVALUATION[0]
        else:
            eval_kwargs = {}
        # hard-code way to remove EvalHook args
        for key in ["interval", "tmpdir", "start", "gpu_collect"]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        print("Evaluation results: ", dataset.evaluate(outputs, **eval_kwargs))


def args_assert(args):
    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")


def init_distribute(args):
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        if "DIST_PARAMS" in cfg.RUNTIME:
            if isinstance(cfg.RUNTIME.DIST_PARAMS, list):
                init_dist(args.launcher, **get_dict_from_list(cfg.RUNTIME.DIST_PARAMS))
        else:
            init_dist(args.launcher)
    return distributed


def model_reset(args, model, dataset):
    print("Loading checkpoint from: ", args.checkpoint)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    # this walkaround is for backward compatibility
    if hasattr(checkpoint, "meta"):
        if "class_names" in checkpoint["meta"]:
            model.class_names = checkpoint["meta"]["class_names"]
    else:
        if hasattr(dataset, "class_names"):
            model.class_names = dataset.class_names
    return model


def main():
    args = parse_args()
    args_assert(args)

    get_task_cfg(cfg, args.task_config)
    distributed = init_distribute(args)

    # build the dataloader
    dataset_args = get_dataset_global_args(cfg.DATA)
    dataset = build_dataset(cfg.DATA.TEST_DATA, cfg.DATA.TEST_TRANSFORMS, dataset_args)
    data_loader = build_dataloader(
        dataset,
        samples_per_device=cfg.DATA.TEST_DATA.SAMPLES_PER_DEVICE,
        workers_per_device=cfg.DATA.TEST_DATA.WORKERS_PER_DEVICE,
        dist=distributed,
        shuffle=False,
        drop_last=False,
    )

    # build the model and load checkpoint
    model = build_model(cfg.MODEL)
    model = model_reset(args, model, dataset)

    if cfg.MODEL.TYPE == "mtl" and cfg.MODEL.NAME == "MTLMultiModalVideoClassifier":
        cfg.MODEL.TYPE = "vcls"

    if not distributed:
        model = DataParallel(model, device_ids=[0])
        outputs = single_device_test(
            model,
            data_loader,
            model_type=cfg.MODEL.TYPE,
            show=args.show,
            out_dir=args.show_dir,
            show_score_thr=args.show_score_thr,
            save_txt_path=args.save_txt_path,
        )
    else:
        model = DistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_device_test(
            model,
            data_loader,
            model_type=cfg.MODEL.TYPE,
            tmpdir=args.tmpdir,
            gpu_collect=args.gpu_collect,
        )

        # save to .npy

    rank, _ = get_dist_info()
    if rank == 0:
        eval_dataset(args, outputs, dataset)


if __name__ == "__main__":
    main()
