# -*- coding: utf-8 -*-
# @Time    : 2021/8/5 21:00
# @Author  : zhiming.qian
# @Email   : zhiming.qian@micro-i.com.cn

import os
import argparse
import torch
from torch import nn as nn
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel

from configs import cfg
from mtl.utils.parallel_util import init_dist
from mtl.datasets.data_builder import build_dataloader, build_dataset
from mtl.models.model_builder import build_model
from mtl.utils.config_util import get_task_cfg, get_dataset_global_args
from mtl.utils.config_util import get_dict_from_list


def extract_feature_pipeline(
    model, train_dataloader, test_dataloader, use_cuda=True, dump_path=None
):
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features, train_labels = extract_features(model, train_dataloader, use_cuda)
    print("Extracting features for val set...")
    test_features, test_labels = extract_features(model, test_dataloader, use_cuda)

    if dist.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    # save features and labels
    if dump_path is not None and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(dump_path, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(dump_path, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(dump_path, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(dump_path, "testlabels.pth"))

    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True):
    features = None
    all_labels = None
    for i, img_data in enumerate(data_loader):
        if i % 1000 == 0:
            print(f"{i} items have been proceed.")

        if use_cuda:
            samples = img_data["img"].cuda(non_blocking=True)
            index = img_data["idx"].cuda(non_blocking=True)
            labels = img_data["gt_label"].cuda(non_blocking=True)
        feats = model(samples)
        if isinstance(feats, tuple):
            feats = feats[-1]

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            all_labels = torch.zeros(len(data_loader.dataset), dtype=labels.dtype)
            if use_cuda:
                features = features.cuda(non_blocking=True)
                all_labels = all_labels.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(
            dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device
        )
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_f = list(feats_all.unbind(0))
        output_f_reduce = torch.distributed.all_gather(output_f, feats, async_op=True)
        output_f_reduce.wait()

        labels_all = torch.empty(
            dist.get_world_size(),
            labels.size(0),
            dtype=labels.dtype,
            device=labels.device,
        )
        output_l = list(labels_all.unbind(0))
        output_l_reduce = torch.distributed.all_gather(output_l, labels, async_op=True)
        output_l_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            features.index_copy_(0, index_all, torch.cat(output_f))
            all_labels.index_copy_(0, index_all, torch.cat(output_l))

    return features, all_labels


@torch.no_grad()
def knn_classifier(
    train_features,
    train_labels,
    test_features,
    test_labels,
    num_k,
    temperature,
    num_classes=1000,
):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(num_k, num_classes).cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[idx : min((idx + imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(num_k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * num_k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(temperature).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = (
            top5 + correct.narrow(1, 0, min(5, num_k)).sum().item()
        )  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


def parse_args():
    parser = argparse.ArgumentParser(description="test (and eval) a model")
    parser.add_argument("task_config", help="test config file")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--nb_knn",
        default=[10, 20, 100, 200],
        nargs="+",
        type=int,
        help="Number of NN to use. 20 is usually working the best.",
    )
    parser.add_argument(
        "--temperature",
        default=0.07,
        type=float,
        help="Temperature used in the voting coefficient",
    )
    parser.add_argument(
        "--dump_path",
        default=None,
        help="Path where to save computed features, empty for no saving",
    )
    parser.add_argument(
        "--load_feat_path",
        default=None,
        help="""If the features have
            already been computed, where to find them.""",
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


def main():
    args = parse_args()

    get_task_cfg(cfg, args.task_config)

    # init distributed env first, since logger depends on the dist info.

    if args.load_feat_path is not None:
        train_features = torch.load(os.path.join(args.load_feat_path, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.load_feat_path, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_feat_path, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(args.load_feat_path, "testlabels.pth"))
        if torch.cuda.is_available():
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()
    else:
        # need to extract features!
        # build the dataloader
        assert args.launcher != "none", "only support distributed evaluation"

        distributed = True
        if "DIST_PARAMS" in cfg.RUNTIME:
            if isinstance(cfg.RUNTIME.DIST_PARAMS, list):
                init_dist(args.launcher, **get_dict_from_list(cfg.RUNTIME.DIST_PARAMS))
        else:
            init_dist(args.launcher)

        dataset_args = get_dataset_global_args(cfg.DATA)
        print("build train dataset")
        train_dataset = build_dataset(
            cfg.DATA.TRAIN_DATA, cfg.DATA.TRAIN_TRANSFORMS, dataset_args
        )
        print("build test dataset")
        test_dataset = build_dataset(
            cfg.DATA.TEST_DATA, cfg.DATA.TRAIN_TRANSFORMS, dataset_args
        )
        # cfg.RUNTIME.GPU_IDS = list(range(args.gpus))

        train_dataloader = build_dataloader(
            train_dataset,
            cfg.DATA.TEST_DATA.SAMPLES_PER_DEVICE,
            cfg.DATA.TEST_DATA.WORKERS_PER_DEVICE,
            None,
            dist=distributed,
            shuffle=False,
            seed=cfg.RUNTIME.SEED,
            pin_memory=cfg.RUNTIME.PIN_MEMORY,
            drop_last=False,
        )
        test_dataloader = build_dataloader(
            test_dataset,
            cfg.DATA.TEST_DATA.SAMPLES_PER_DEVICE,
            cfg.DATA.TEST_DATA.WORKERS_PER_DEVICE,
            None,
            dist=distributed,
            shuffle=False,
            seed=cfg.RUNTIME.SEED,
            pin_memory=cfg.RUNTIME.PIN_MEMORY,
            drop_last=False,
        )

        # build the model and load checkpoint
        model = build_model(cfg.MODEL)

        model = DistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        (
            train_features,
            test_features,
            train_labels,
            test_labels,
        ) = extract_feature_pipeline(
            model,
            train_dataloader,
            test_dataloader,
            use_cuda=True,
            dump_path=args.dump_path,
        )

    print("Features are ready!\nStart the k-NN classification.")
    if dist.get_rank() == 0:
        for k in args.nb_knn:
            top1, top5 = knn_classifier(
                train_features,
                train_labels,
                test_features,
                test_labels,
                k,
                args.temperature,
            )
            print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")


if __name__ == "__main__":
    main()
