import torch
from torch import nn
from collections import OrderedDict

from mtl.cores.layer_ops.inception_layer import InceptionAux
from ..block_builder import HEADS
from .base_cls_head import BaseClsDenseHead


@HEADS.register_module()
class InceptionClsHead(BaseClsDenseHead):
    """classification aux logits head."""

    def __init__(self, num_classes=1000, aux_logits=True):
        super(InceptionClsHead, self).__init__()
        self.aux_logits = aux_logits

        if aux_logits:
            self.inception_aux = InceptionAux(768, num_classes)
        self.fc = nn.Sequential(OrderedDict([("fc", nn.Linear(2048, num_classes))]))

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
        x = self.fc(x[1])
        if self.training and self.aux_logits:
            aux = self.inception_aux(x[0])
            return x, aux
        else:
            return x
