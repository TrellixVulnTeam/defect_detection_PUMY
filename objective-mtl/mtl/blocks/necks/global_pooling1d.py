import torch
import torch.nn as nn

from ..block_builder import NECKS


@NECKS.register_module()
class GlobalAveragePooling1D(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling.
    We do not use `squeeze` as it will also remove the batch dimension
    when the tensor has a batch dimension of size 1, which can lead to
    unexpected errors.
    """

    def __init__(self):
        super(GlobalAveragePooling1D, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x.transpose(1, 2)) for x in inputs])
            outs = tuple([out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs.transpose(1, 2))
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError("neck inputs should be tuple or torch.tensor")
        return outs
