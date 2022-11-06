import torch.nn as nn

from mtl.utils.init_util import trunc_normal_
from ..block_builder import HEADS
from .base_emb_head import BaseEmbHead


@HEADS.register_module()
class MIMDecoderHead(BaseEmbHead):
    """Decoder head for MIM."""

    def __init__(self, in_dim, encoder_stride):
        super(MIMDecoderHead, self).__init__()
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_dim,
                out_channels=self.encoder_stride ** 2 * 3,
                kernel_size=1,
            ),
            nn.PixelShuffle(self.encoder_stride),
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
        x = self.decoder(x)
        return x
