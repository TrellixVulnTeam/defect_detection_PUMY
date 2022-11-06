import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mtl.cores.layer_ops.yolo_layer import YoloConv, YoloConcat
from ..block_builder import BACKBONES


class YoloV7StemBlock(nn.Sequential):
    """Begin block for yolov7, size to be 1/4"""

    def __init__(self, in_channels=3, basic_channels=32):
        layers = [
            YoloConv(in_channels, basic_channels, k=3, s=1),
            YoloConv(basic_channels, 2 * basic_channels, k=3, s=2),
            YoloConv(2 * basic_channels, 2 * basic_channels, k=3, s=1),
            YoloConv(2 * basic_channels, 4 * basic_channels, k=3, s=2),
        ]
        super(YoloV7StemBlock, self).__init__(*layers)

    def forward(self, input):
        for module in self:
            input = module(input)
        return input


class YoloV7BasicBlock(nn.Module):
    """Basic convolutions for block feature extration"""

    def __init__(
        self,
        in_channels=128,
        basic_channels=64,
        out_channels=256,
        num_convs=6,
        concat_list=[-1, -3, -5, -6],
    ):
        super(YoloV7BasicBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            if i < 2:
                layers.append(YoloConv(in_channels, basic_channels, k=1, s=1))
            else:
                layers.append(YoloConv(basic_channels, basic_channels, k=3, s=1))
        self.convs = nn.ModuleList(layers)

        self.concat = YoloConcat(dimension=1)
        self.conv_out = YoloConv(
            basic_channels * len(concat_list), out_channels, k=1, s=1
        )
        self.num_convs = num_convs
        self.concat_list = concat_list

    def forward(self, x):
        mid_list = []
        for i, conv in enumerate(self.convs):
            if i - self.num_convs in self.concat_list:
                mid_list.append(conv(x))
            if i > 0:
                x = conv(x)
        x = self.concat(mid_list[::-1])
        x = self.conv_out(x)

        return x


class YoloV7MaxPoolingBlock(nn.Module):
    """Basic convolutions for block feature extration"""

    def __init__(self, in_channels=256, basic_channels=128, stride=2):
        super(YoloV7MaxPoolingBlock, self).__init__()
        self.mp = nn.MaxPool2d(kernel_size=stride, stride=stride)
        self.conv_mp = YoloConv(in_channels, basic_channels, k=1, s=1)

        self.conv1 = YoloConv(in_channels, basic_channels, k=1, s=1)
        self.conv2 = YoloConv(basic_channels, basic_channels, k=3, s=stride)
        self.concat = YoloConcat(dimension=1)

    def forward(self, x):
        x1 = self.mp(x)
        x1 = self.conv_mp(x1)

        x2 = self.conv1(x)
        x2 = self.conv2(x2)
        x = self.concat([x2, x1])

        return x


@BACKBONES.register_module()
class YOLOV7BKNet(nn.Module):
    """Backbone for yolov7"""

    def __init__(
        self,
        in_channels=3,
        basic_channels=32,
        bottle_channels=64,
        num_convs=6,
        concat_list=[-1, -3, -5, -6],
        out_indices=(0, 1, 2, 3),
    ):
        super(YOLOV7BKNet, self).__init__()
        num_concat = len(concat_list)
        self.stage1 = nn.Sequential(
            YoloV7StemBlock(in_channels, basic_channels),
            YoloV7BasicBlock(
                in_channels=4 * basic_channels,
                basic_channels=bottle_channels,
                out_channels=num_concat * bottle_channels,
                num_convs=num_convs,
                concat_list=concat_list,
            ),
        )  # 11

        self.stage2 = nn.Sequential(
            YoloV7MaxPoolingBlock(
                in_channels=num_concat * bottle_channels,
                basic_channels=num_concat * bottle_channels // 2,
            ),
            YoloV7BasicBlock(
                in_channels=num_concat * bottle_channels,
                basic_channels=bottle_channels * 2,
                out_channels=num_concat * bottle_channels * 2,
                num_convs=num_convs,
                concat_list=concat_list,
            ),
        )  # 24

        self.stage3 = nn.Sequential(
            YoloV7MaxPoolingBlock(
                in_channels=num_concat * bottle_channels * 2,
                basic_channels=num_concat * bottle_channels,
            ),
            YoloV7BasicBlock(
                in_channels=num_concat * bottle_channels * 2,
                basic_channels=bottle_channels * 4,
                out_channels=num_concat * bottle_channels * 4,
                num_convs=num_convs,
                concat_list=concat_list,
            ),
        )  # 37

        self.stage4 = nn.Sequential(
            YoloV7MaxPoolingBlock(
                in_channels=num_concat * bottle_channels * 4,
                basic_channels=num_concat * bottle_channels * 2,
            ),
            YoloV7BasicBlock(
                in_channels=num_concat * bottle_channels * 4,
                basic_channels=bottle_channels * 4,
                out_channels=num_concat * bottle_channels * 4,
                num_convs=num_convs,
                concat_list=concat_list,
            ),
        )  # 50
        self.out_indices = out_indices

        self.init_weights()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            pass
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
                elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                    m.inplace = True
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x):
        out1 = x = self.stage1(x)
        out2 = x = self.stage2(x)
        out3 = x = self.stage3(x)
        out4 = x = self.stage4(x)
        all_outs = (out1, out2, out3, out4)
        outs = []
        for i in self.out_indices:
            outs.append(all_outs[i])
        return outs
