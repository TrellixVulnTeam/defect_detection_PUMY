import torch
import torch.nn as nn

from mtl.utils.init_util import trunc_normal_
from ..block_builder import HEADS
from .base_emb_head import BaseEmbHead


@HEADS.register_module()
class ProtoTypeHead(BaseEmbHead):
    def __init__(
        self, in_dim=768, inner_dim=1536, out_dim=3000, num_layers=3, num_prototypes=1
    ):
        super(ProtoTypeHead, self).__init__()
        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.nmb_heads = num_prototypes
        head_in_dim = in_dim if num_layers == 1 else inner_dim
        for i in range(self.nmb_heads):
            self.add_module(
                "prototypes" + str(i), nn.Linear(head_in_dim, out_dim, bias=False)
            )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        with torch.no_grad():
            for i in range(self.nmb_heads):
                w = torch.abs(getattr(self, "prototypes" + str(i)).weight.data.clone())
                w = nn.functional.normalize(w, dim=1, p=2)
                getattr(self, "prototypes" + str(i)).weight.copy_(w)
        if isinstance(x, (tuple, list)):
            x = x[-1]
        x = self.linear_hidden(x)
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out
