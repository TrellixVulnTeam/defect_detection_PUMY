from torch import nn

from mtl.utils.checkpoint_util import load_checkpoint
from mtl.utils.log_util import get_root_logger
from mtl.utils.init_util import kaiming_init
from ..block_builder import NECKS


@NECKS.register_module()
class ChannelAlginNeck(nn.Module):
    """The non-linear neck in ChannelAlginNeck.
    Single and dense in parallel: conv-bn-relu-conv
    """

    def __init__(self, in_channels, hid_channels, out_channels):
        super(ChannelAlginNeck, self).__init__()

        self.conv_neck = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1),
        )

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
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    kaiming_init(m, mode="fan_in", nonlinearity="relu")
                elif isinstance(
                    m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)
                ):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[-1]
        x = self.conv_neck(x)  # sxs: bxdxsxs
        return x
