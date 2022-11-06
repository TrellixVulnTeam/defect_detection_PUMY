import math
import torch
from torch import nn
from torch.functional import Tensor

from ..block_builder import BACKBONES


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding,
    very similar to the one used by the Attention is all you need paper,
    generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats=64,
        temperature=10000,
        normalize=False,
        scale=None,
        max_h=30,
        max_w=30,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.max_h = max_h
        self.max_w = max_w
        pe = self._gen_pos_buffer()
        self.register_buffer("pe", pe)

    def _gen_pos_buffer(self):
        _eyes = torch.ones((1, self.max_h, self.max_w))
        y_embed = _eyes.cumsum(1, dtype=torch.float32)
        x_embed = _eyes.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, input: Tensor):
        x = input
        return self.pe.repeat((x.size(0), 1, 1, 1))


@BACKBONES.register_module()
class PositionEmbedding(nn.Module):
    """Position Embeddign for Q2L model.
        A Pytorch implementation adapted from orginal Q2L code:
        https://github.com/SlongLiu/query2labels
    Args:
        img_size (int): Input image size, default 224.
        downsample_ratio (int): for 4 stage swin-t, the patch_embed resolution
            downsample 2^3 times,  i.e., 4 (patch size) x 2^3 = 32
        hidden_dim (int):
    """

    def __init__(self, img_size=224, downsample_ratio=32, hidden_dim=1536):
        super().__init__()

        self.img_size = img_size
        assert self.img_size % 32 == 0, "img_size ({}) % 32 != 0".format(self.img_size)

        self.downsample_ratio = downsample_ratio

        self.hidden_dim = hidden_dim
        n_steps = self.hidden_dim // 2  # 768 = 96 x 8

        self.position_embedding = PositionEmbeddingSine(
            n_steps,
            normalize=True,
            max_h=self.img_size // self.downsample_ratio,  # 224 / 32 = 7
            max_w=self.img_size // self.downsample_ratio,  # 224 / 32 = 7
        )

    def forward(self, x):
        return self.position_embedding(x)
