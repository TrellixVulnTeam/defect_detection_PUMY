import math
import torch
import torch.nn.functional as F
from torch import nn

from mtl.cores.layer_ops.q2l_transformer import Q2LTransformer
from .base_cls_head import BaseClsDenseHead
from ..block_builder import HEADS


class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.w = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.w.size(2))
        for i in range(self.num_class):
            self.w[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.w * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


@HEADS.register_module()
class Q2LTransformerClsHead(BaseClsDenseHead):
    def __init__(
        self,
        num_classes,
        in_channels,
        d_model=1536,
        num_encoder_layers=1,
        num_decoder_layers=2,
    ):
        super(Q2LTransformerClsHead, self).__init__()
        self.transformer = self.get_transformer(
            d_model, num_encoder_layers, num_decoder_layers
        )
        self.in_channels = in_channels
        self.num_classes = num_classes
        hidden_dim = self.transformer.d_model
        self.input_proj = nn.Conv2d(self.in_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_classes, hidden_dim)
        self.fc = GroupWiseLinear(num_classes, hidden_dim, bias=True)

    def get_transformer(self, d_model=1536, num_encoder_layers=1, num_decoder_layers=2):
        return Q2LTransformer(
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            normalize_before=False,
            return_intermediate_dec=False,
        )

    def init_weights(self):
        pass

    def forward(self, src, pos):
        query_input = self.query_embed.weight
        hs = self.transformer(self.input_proj(src), query_input, pos)[0]  # B,K,d
        out = self.fc(hs[-1])
        return out
