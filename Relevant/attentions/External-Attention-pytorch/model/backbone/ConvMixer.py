import torch.nn as nn
from torch.nn.modules.activation import GELU
import torch
from torch.nn.modules.pooling import AdaptiveAvgPool2d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


# def ConvMixer(dim, depth, channel, kernel_size=9, patch_size=1, num_classes=1000):
#     return nn.Sequential(
#         nn.Conv2d(channel, dim, kernel_size=patch_size, stride=patch_size),
#         nn.GELU(),
#         nn.BatchNorm2d(dim),
#         *[nn.Sequential(
#             Residual(nn.Sequential(
#                 nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding=kernel_size // 2),
#                 nn.GELU(),
#                 nn.BatchNorm2d(dim)
#             )),
#             nn.Conv2d(dim, dim, kernel_size=1),
#             nn.GELU(),
#             nn.BatchNorm2d(dim)
#         ) for _ in range(depth)]
#         # nn.AdaptiveAvgPool2d(1),
#         # nn.Flatten(),
#         # nn.Linear(dim, num_classes)
#     )

def ConvMixer_(channel, dim, kernel_size=9,
               patch_size=1, depth=12):
    return nn.Sequential(
        nn.Conv2d(channel, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding=kernel_size // 2),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        ) for _ in range(depth)]
        # nn.AdaptiveAvgPool2d(1),
        # nn.Flatten(),
        # nn.Linear(dim, num_classes)
    )


class ConvMixer(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, in_channel, out_channel,
                 kernel,
                 patch_size
                 ):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv_mixer = ConvMixer_(channel=in_channel,
                                     dim=out_channel,
                                     kernel_size=kernel,
                                     patch_size=patch_size
                                     )

    def forward(self, x):
        return self.conv_mixer(x)


if __name__ == '__main__':
    size = 64
    x = torch.randn(1, size, 640, 640)
    convmixer = ConvMixer(in_channel=size,
                          out_channel=size,
                          kernel=3,
                          patch_size=2)
    out = convmixer(x)
    print(out.shape)  # [1, 1000]
