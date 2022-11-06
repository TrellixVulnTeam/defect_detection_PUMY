import torch.nn as nn

from mtl.utils.init_util import normal_init
from ..block_builder import HEADS
from .base_cls_head import BaseClsDenseHead


@HEADS.register_module()
class LinearClsHead(BaseClsDenseHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
    """

    def __init__(self, num_classes, in_channels, dropout_ratio=0.0):
        super(LinearClsHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.dropout = nn.Dropout(dropout_ratio)

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[-1]
        x = self.fc(x)
        x = self.dropout(x)
        return x
