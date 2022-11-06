# -*- coding: utf-8 -*-
# @Time    : 2020/12/02 17:00
# @Author  : zhiming.qian
# @Email   : zhiming.qian@micro-i.com.cn
# @File    : runtime_util.py

import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from collections.abc import Mapping, Sequence
import os.path as osp
import subprocess
import sys
from collections import defaultdict
import cv2
from torch.utils.cpp_extension import CUDA_HOME
from getpass import getuser
from socket import gethostname


def collect_env():
    """Collect the information of the running environments.

    Returns:
        dict: The environment information. The following fields are contained.
            - sys.platform: The variable of ``sys.platform``.
            - Python: Python version.
            - CUDA available: Bool, indicating if CUDA is available.
            - GPU devices: Device type of each GPU.
            - CUDA_HOME (optional): The env var ``CUDA_HOME``.
            - NVCC (optional): NVCC version.
            - GCC: GCC version, "n/a" if GCC is not installed.
            - PyTorch: PyTorch version.
            - PyTorch compiling details: The output of \
                ``torch.__config__.show()``.
            - TorchVision (optional): TorchVision version.
            - OpenCV: OpenCV version.
    """
    env_info = {}
    env_info["sys.platform"] = sys.platform
    env_info["Python"] = sys.version.replace("\n", "")

    cuda_available = torch.cuda.is_available()
    env_info["CUDA available"] = cuda_available

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info["GPU " + ",".join(device_ids)] = name

        env_info["CUDA_HOME"] = CUDA_HOME

        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, "bin/nvcc")
                nvcc = subprocess.check_output(f'"{nvcc}" -V | tail -n1', shell=True)
                nvcc = nvcc.decode("utf-8").strip()
            except subprocess.SubprocessError:
                nvcc = "Not Available"
            env_info["NVCC"] = nvcc

    try:
        gcc = subprocess.check_output("gcc --version | head -n1", shell=True)
        gcc = gcc.decode("utf-8").strip()
        env_info["GCC"] = gcc
    except subprocess.CalledProcessError:  # gcc is unavailable
        env_info["GCC"] = "n/a"

    env_info["PyTorch"] = torch.__version__
    env_info["PyTorch compiling details"] = torch.__config__.show()

    try:
        import torchvision

        env_info["TorchVision"] = torchvision.__version__
    except ModuleNotFoundError:
        pass

    env_info["OpenCV"] = cv2.__version__

    return env_info


def get_host_info():
    return f"{getuser()}@{gethostname()}"


def collate(batch):
    """Puts each data field into a tensor with outer dimension
    batch size.
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        result = {}
        for key in batch[0]:
            if key == "imgs":
                result[key] = []
                for j in range(0, len(batch[0]["imgs"])):
                    slice_imgs = []
                    for i in range(0, len(batch)):
                        slice_imgs.append(batch[i][key][j])
                    result[key].append(torch.stack(slice_imgs))
            elif (
                key == "img_metas"
                or key == "gt_bboxes"
                or key == "gt_labels"
                or key == "gt_masks"
            ):
                result[key] = []
                for i in range(0, len(batch)):
                    result[key].append(batch[i][key])
            elif key == "txt_id" or key == "txt_mask":
                max_shape = batch[0][key].size(-1)
                for i in range(0, len(batch)):
                    max_shape = max(max_shape, batch[i][key].size(-1))
                padded_samples = []
                for i in range(0, len(batch)):
                    len_pad = max_shape - batch[i][key].size(-1)
                    if len_pad > 0:
                        pad_value = torch.zeros([len_pad], dtype=torch.long)
                        padded_samples.append(
                            torch.cat([batch[i][key], pad_value], dim=-1)
                        )
                    else:
                        padded_samples.append(batch[i][key])
                result[key] = torch.stack(padded_samples)
            else:
                result[key] = collate([d[key] for d in batch])
        return result
    else:
        if isinstance(batch[0], torch.Tensor) and batch[0].dim() == 3:
            # image data, should convert to the same size
            max_shape = [batch[0].size(-1), batch[0].size(-2)]
            for sample in batch:
                max_shape[0] = max(max_shape[0], sample.size(-1))
                max_shape[1] = max(max_shape[1], sample.size(-2))

            padded_samples = []
            for sample in batch:
                pad = [
                    0,
                    max_shape[0] - sample.size(-1),
                    0,
                    max_shape[1] - sample.size(-2),
                ]
                padded_samples.append(F.pad(sample, pad, value=-1))
            return default_collate(padded_samples)

        return default_collate(batch)


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, num_workers, rank, seed):
    """The seed of each worker equals to
    num_worker * rank + worker_id + user_seed
    """
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_time_str():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def ensure_rng(rng=None):
    """Simple version of the ``kwarray.ensure_rng``

    Args:
        rng (int | numpy.random.RandomState | None):
            if None, then defaults to the global rng. Otherwise this can be an
            integer or a RandomState class
    Returns:
        (numpy.random.RandomState) : rng -
            a numpy random number generator
    """
    if rng is None:
        rng = np.random.mtrand._rand
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)
    else:
        rng = rng
    return rng
