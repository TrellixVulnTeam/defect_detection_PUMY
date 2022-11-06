from collections import OrderedDict
import torch.nn as nn

from mtl.utils.init_util import normal_init, constant_init
from mtl.blocks.backbones.swin_transformer import BatchNorm1dNoBias
from ..block_builder import HEADS
from .base_emb_head import BaseEmbHead


@HEADS.register_module()
class FCMlpHead(BaseEmbHead):
    """fc mlp embedder head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """

    def __init__(self, num_classes, in_channels, use_mlp=True):
        super(FCMlpHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_mlp = use_mlp

        if use_mlp:
            self.fc = nn.Sequential(
                OrderedDict(
                    [
                        ("fc1", nn.Linear(in_channels, in_channels, bias=False)),
                        ("bn1", nn.BatchNorm1d(in_channels, eps=1e-4)),
                        ("relu1", nn.ReLU()),
                        ("fc2", nn.Linear(in_channels, in_channels, bias=False)),
                        ("bn2", nn.BatchNorm1d(in_channels, eps=1e-4)),
                        ("relu2", nn.ReLU()),
                        ("fc3", nn.Linear(in_channels, num_classes, bias=False)),
                        ("bn3", BatchNorm1dNoBias(num_classes)),
                    ]
                )
            )
        else:
            self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)
            elif isinstance(m, nn.BatchNorm1d):
                constant_init(m, 1)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[-1]
        x = self.fc(x)
        return x
