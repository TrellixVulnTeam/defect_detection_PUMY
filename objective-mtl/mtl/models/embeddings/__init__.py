# -*- encoding: utf-8 -*-

###########################################################
# @File    :   __init__.py
# @Time    :   2021/10/13 22:51:49
# @Author  :   Qian Zhiming
# @Contact :   zhiming.qian@micro-i.com.cn
###########################################################

from .moco_embedder import MocoEmbedder
from .moby_embedder import MobyEmbedder
from .dino_embedder import DINOEmbedder
from .dino_cls_embedder import DINOCLSEmbedder
from .mae_embedder import MaeEmbedder
from .knn_embedder import KNNEmbedder
from .hvpmtl_embedder import HVPMTLEmbedder
from .distill_embedder import DistillEmbedder

__all__ = [
    "MocoEmbedder",
    "MobyEmbedder",
    "DINOEmbedder",
    "DINOCLSEmbedder",
    "MaeEmbedder",
    "KNNEmbedder",
    "HVPMTLEmbedder",
    "DistillEmbedder",
]
