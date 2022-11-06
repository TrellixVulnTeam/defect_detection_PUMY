from abc import ABCMeta, abstractmethod
import torch.nn as nn


class BaseClsDenseHead(nn.Module, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    @abstractmethod
    def init_weights(self):
        """Initialize weights."""
        pass

    @abstractmethod
    def forward(self, x):
        """Placeholder of forward function."""
        pass
