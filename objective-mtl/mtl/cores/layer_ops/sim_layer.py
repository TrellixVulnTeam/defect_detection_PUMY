import torch
from torch import nn


class SimLayer(nn.Module):
    """Similarity Module for training a background embedding.

    Args:
        channels (int): The input (and output) channels of the similarity layer
    """

    def __init__(self, channels):
        super(SimLayer, self).__init__()
        self.bg_embedding = nn.Parameter(torch.FloatTensor(channels))

    def forward(self, x):
        x = (
            torch.sum(x * self.bg_embedding)
            / torch.norm(x)
            * torch.norm(self.bg_embedding)
        )
        return x
