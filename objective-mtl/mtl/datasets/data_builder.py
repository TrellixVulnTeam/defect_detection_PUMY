# -*- coding: utf-8 -*-
# @Time    : 2020/12/03 16:00
# @Author  : zhiming.qian

import os
import glob
from functools import partial
from torch.utils.data import DataLoader, RandomSampler

from mtl.utils.reg_util import build_data_from_cfg
from mtl.utils.runtime_util import worker_init_fn, collate
from mtl.utils.misc_util import get_dist_info
from .data_sampler import (
    DistributedSampler,
    InfiniteBatchSampler,
    InfiniteGroupBatchSampler,
)
from .data_wrapper import ConcatDataset, RepeatDataset
from .data_wrapper import ClassBalancedDataset, DATASETS


# --------------------------------------------------------------------------- #
# Avoid resource limit
# --------------------------------------------------------------------------- #

# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# hard_limit = rlimit[1]
# soft_limit = min(4096, hard_limit)
# resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def build_dataset(data_cfg, pipeline_cfg, default_args, sel_index=0):
    if isinstance(data_cfg, (list, tuple)):
        dataset = ConcatDataset(
            [build_dataset(c, pipeline_cfg, default_args, sel_index) for c in data_cfg]
        )
    elif data_cfg.TYPE == "Concat":
        cfg_opt = data_cfg.clone()
        cfg_opt.defrost()
        cfg_opt.TYPE = "Normal"
        dataset = ConcatDataset(
            [
                build_dataset(cfg_opt, pipeline_cfg, default_args, sel_index=i)
                for i in range(len(data_cfg.DATA_INFO))
            ],
            data_cfg.FLAG,
        )
    elif data_cfg.TYPE == "Repeat":
        cfg_opt = data_cfg.clone()
        cfg_opt.defrost()
        cfg_opt.TYPE = "Concat"
        dataset = RepeatDataset(
            build_dataset(cfg_opt, pipeline_cfg, default_args, sel_index), data_cfg.FLAG
        )
    elif data_cfg.TYPE == "Balanced":
        cfg_opt = data_cfg.clone()
        cfg_opt.defrost()
        cfg_opt.TYPE = "Concat"
        dataset = ClassBalancedDataset(
            build_dataset(cfg_opt, pipeline_cfg, default_args, sel_index), data_cfg.FLAG
        )
    else:
        relative_path_list = []
        for relative_path in data_cfg.DATA_INFO[sel_index]:
            if "*" in relative_path:
                extend_list = glob.glob(
                    os.path.join(default_args["root_path"], relative_path)
                )
                len_root_path = len(default_args["root_path"])
                for extend_global_path in extend_list:
                    re_path = extend_global_path[len_root_path:]
                    if re_path.startswith("/"):
                        re_path = re_path[1:]
                    relative_path_list.append(re_path)
            else:
                relative_path_list.append(relative_path)
        data_cfg.DATA_INFO[sel_index] = relative_path_list
        dataset = build_data_from_cfg(
            data_cfg, pipeline_cfg, default_args, DATASETS, sel_index
        )

    return dataset


def build_dataloader(
    dataset,
    samples_per_device,
    workers_per_device,
    gpu_ids=None,
    dist=False,
    shuffle=True,
    seed=None,
    runner_type="EpochBasedRunner",
    pin_memory=True,
    drop_last=True,
    **kwargs
):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_device (int): Number of training samples on each device,
            i.e., batch size of each device.
        workers_per_device (int): How many subprocesses to use for data
            loading for each device.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        batch_size = samples_per_device
        num_workers = workers_per_device
    else:
        if gpu_ids is not None and gpu_ids != "":
            num_gpus = len(gpu_ids)
            batch_size = num_gpus * samples_per_device
            num_workers = num_gpus * workers_per_device
        else:
            batch_size = samples_per_device
            num_workers = workers_per_device

    if runner_type == "IterBasedRunner":
        # this is a batch sampler, which can yield
        # a mini-batch indices each time.
        # it can be used in both `DataParallel` and
        # `DistributedDataParallel`
        if shuffle:
            batch_sampler = InfiniteGroupBatchSampler(
                dataset, batch_size, world_size, rank, seed=seed
            )
        else:
            batch_sampler = InfiniteBatchSampler(
                dataset, batch_size, world_size, rank, seed=seed, shuffle=False
            )
        batch_size = 1
        sampler = None
    else:
        if dist:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=shuffle, seed=seed
            )  # if dataset size is large enough, shuffle can be set as False
        else:
            sampler = RandomSampler(dataset) if shuffle else None
        batch_sampler = None

    init_fn = (
        partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
        if seed is not None
        else None
    )

    if batch_sampler is not None:
        drop_last = False

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collate,
        pin_memory=pin_memory,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs
    )

    return data_loader
