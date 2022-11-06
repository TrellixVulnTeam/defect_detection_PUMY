import torch.nn as nn
import torch.nn.functional as F

from mtl.utils.checkpoint_util import load_checkpoint
from mtl.utils.log_util import get_root_logger
from mtl.cores.ops.ops_builder import build_activation_layer, build_norm_layer
from mtl.utils.init_util import normal_init
from mtl.cores.layer_ops.se_layer import DyReLU
from ..block_builder import NECKS


class DyNeckBlock(nn.Module):
    """DyNeck Block with three types of attention.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        zero_init_offset (bool, optional): Whether to use zero init for
            `spatial_conv_offset`. Default: True.
        act_cfg (dict, optional): Config dict for the last activation layer of
            scale-aware attention. Default: dict(type='HSigmoid', bias=3.0,
            divisor=6.0).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        act_cfg=dict(type="HSigmoid", bias=3.0, divisor=6.0),
        norm_cfg=dict(type="GN", num_groups=16, requires_grad=True),
    ):
        super().__init__()
        self.spatial_conv_high = nn.Conv2d(
            in_channels, out_channels, 3, padding=1, bias=False
        )
        self.norm_high = build_norm_layer(norm_cfg, out_channels)[1]
        self.spatial_conv_mid = nn.Conv2d(
            in_channels, out_channels, 3, padding=1, bias=False
        )
        self.norm_mid = build_norm_layer(norm_cfg, out_channels)[1]
        self.spatial_conv_low = nn.Conv2d(
            in_channels, out_channels, 3, stride=2, padding=1, bias=False
        )
        self.norm_low = build_norm_layer(norm_cfg, out_channels)[1]

        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, 1, 1),
            nn.ReLU(inplace=True),
            build_activation_layer(act_cfg),
        )
        self.task_attn_module = DyReLU(out_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)

    def forward(self, x):
        """Forward function."""
        outs = []
        for level in range(len(x)):

            mid_feat = self.spatial_conv_mid(x[level])
            mid_feat = self.norm_mid(mid_feat)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat = self.spatial_conv_low(x[level - 1])
                low_feat = self.norm_low(low_feat)
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                high_feat = self.spatial_conv_high(x[level + 1])
                high_feat = self.norm_high(high_feat)
                high_feat = F.interpolate(
                    high_feat,
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            outs.append(self.task_attn_module(sum_feat / summed_levels))

        return outs


@NECKS.register_module()
class DyNeck(nn.Module):
    """Dynamic neck consisting of multiple DyNeck Blocks.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_blocks (int, optional): Number of DyHead Blocks. Default: 6.
        zero_init_offset (bool, optional): Whether to use zero init for
            `spatial_conv_offset`. Default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self, in_channels, out_channels, num_blocks=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        dyneck_blocks = []
        for i in range(num_blocks):
            in_channels = self.in_channels if i == 0 else self.out_channels
            dyneck_blocks.append(DyNeckBlock(in_channels, self.out_channels))
        self.dyneck_blocks = nn.Sequential(*dyneck_blocks)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, inputs):
        """Forward function."""
        assert isinstance(inputs, (tuple, list))
        outs = self.dyneck_blocks(inputs)
        return tuple(outs)
