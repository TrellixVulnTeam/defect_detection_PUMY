import argparse
import torch

from mtl.models.model_builder import build_model
from mtl.utils.config_util import get_task_cfg
from mtl.utils.fps_stat_util import get_model_complexity_info


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--shape", type=int, nargs="+", default=[1280, 800], help="input image size"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")

    cfg = get_task_cfg(args.config)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mtl.utils.misc_util import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    model = build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, "forward_dummy"):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            "FLOPs counter is currently not currently supported with {}".format(
                model.__class__.__name__
            )
        )

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = "=" * 30
    print(
        f"{split_line}\nInput shape: {input_shape}\n"
        f"Flops: {flops}\nParams: {params}\n{split_line}"
    )
    print(
        "!!!Please be cautious if you use the results in papers. "
        "You may need to check if all ops are supported and verify that the "
        "flops computation is correct."
    )


if __name__ == "__main__":
    main()
