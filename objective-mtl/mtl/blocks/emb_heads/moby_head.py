import torch.nn as nn

from mtl.utils.init_util import trunc_normal_
from ..block_builder import HEADS
from .base_emb_head import BaseEmbHead


@HEADS.register_module()
class MoBYHead(BaseEmbHead):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(MoBYHead, self).__init__()

        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = (
            nn.Linear(in_dim if num_layers == 1 else inner_dim, out_dim)
            if num_layers >= 1
            else nn.Identity()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[-1]
        x = self.linear_hidden(x)
        x = self.linear_out(x)

        return x
