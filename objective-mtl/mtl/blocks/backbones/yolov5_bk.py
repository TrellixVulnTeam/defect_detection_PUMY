import torch.nn as nn
import math
import functools
from torch.nn.modules.batchnorm import _BatchNorm

from mtl.cores.layer_ops import brick as vn_layer
from mtl.utils.init_util import constant_init, kaiming_init
from ..block_builder import BACKBONES


def _make_divisible(x, divisor, width_multiple):
    return math.ceil(x * width_multiple / divisor) * divisor


def _make_round(x, depth_multiple=1.0):
    return max(round(x * depth_multiple), 1) if x > 1 else x


def make_divisible(divisor, width_multiple=1.0):
    return functools.partial(
        _make_divisible, divisor=divisor, width_multiple=width_multiple
    )


def make_round(depth_multiple=1.0):
    return functools.partial(_make_round, depth_multiple=depth_multiple)


@BACKBONES.register_module()
class YOLOV5BKNet(nn.Module):
    def __init__(self, input_channel=3, depth_multiple=1.0, width_multiple=1.0):
        # yolov5s: depth_multiple=0.33,width_multiple=0.5
        super(YOLOV5BKNet, self).__init__()
        self.depth_multiple = depth_multiple  # layer
        self.width_multiple = width_multiple  # channel
        self.return_index = [4, 6, 9]
        make_div8_fun = make_divisible(8, self.width_multiple)
        make_round_fun = make_round(self.depth_multiple)

        model = []

        focal = vn_layer.Focus(input_channel, make_div8_fun(64), k=3)
        model.append(focal)
        conv1 = vn_layer.Conv(make_div8_fun(64), make_div8_fun(128), k=3, s=2)
        model.append(conv1)
        bsp1 = vn_layer.BottleneckCSP(
            make_div8_fun(128), make_div8_fun(128), make_round_fun(3)
        )
        model.append(bsp1)
        conv2 = vn_layer.Conv(make_div8_fun(128), make_div8_fun(256), k=3, s=2)
        model.append(conv2)
        bsp2 = vn_layer.BottleneckCSP(
            make_div8_fun(256), make_div8_fun(256), make_round_fun(9)
        )
        model.append(bsp2)  # out 1
        conv3 = vn_layer.Conv(make_div8_fun(256), make_div8_fun(512), k=3, s=2)
        model.append(conv3)
        bsp3 = vn_layer.BottleneckCSP(
            make_div8_fun(512), make_div8_fun(512), make_round_fun(9)
        )
        model.append(bsp3)  # out 2
        conv4 = vn_layer.Conv(make_div8_fun(512), make_div8_fun(1024), k=3, s=2)
        model.append(conv4)
        spp1 = vn_layer.SPP(make_div8_fun(1024), make_div8_fun(1024))
        model.append(spp1)
        bsp4 = vn_layer.BottleneckCSP(
            make_div8_fun(1024), make_div8_fun(1024), make_round_fun(3), shortcut=False
        )
        model.append(bsp4)  # out 3
        self.backbone = nn.Sequential(*model)
        self.init_weights()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            pass
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x):
        out = []
        for i, m in enumerate(self.backbone):
            x = m(x)
            if i in self.return_index:
                out.append(x)
        return out
