from abc import ABCMeta, abstractmethod
from torch import nn


class BaseEmbHead(nn.Module, metaclass=ABCMeta):
    """Linear regressor head.
    Args:
        num_dim (int): Number of regression dimensions.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """

    @abstractmethod
    def init_weights(self):
        """Initialize weights."""
        pass
