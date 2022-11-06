import torch
from torch import nn
from collections import namedtuple

from mtl.cores.layer_ops.inception_layer import InceptionGAux
from ..block_builder import HEADS
from .base_cls_head import BaseClsDenseHead


_GoogLeNetOuputs = namedtuple(
    "GoogLeNetOuputs", ["logits", "aux_logits2", "aux_logits1"]
)


@HEADS.register_module()
class GoogLeNetClsHead(BaseClsDenseHead):
    """classification aux logits head."""

    def __init__(self, num_classes=1000, aux_logits=True):
        super(GoogLeNetClsHead, self).__init__()
        self.aux_logits = aux_logits
        if aux_logits:
            self.aux1 = InceptionGAux(512, num_classes)
            self.aux2 = InceptionGAux(528, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats

                stddev = m.stddev if hasattr(m, "stddev") else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                m.weight.data.copy_(values.reshape(m.weight.shape))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.training and self.aux_logits:
            aux1 = self.aux1(x[0])
            aux2 = self.aux2(x[1])

        x = self.avgpool(x[2])
        # N x 1024 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x num_classes

        if self.training and self.aux_logits:
            return _GoogLeNetOuputs(x, aux2, aux1)
        return x
