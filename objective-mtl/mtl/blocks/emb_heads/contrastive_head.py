import torch

from ..block_builder import HEADS
from .base_emb_head import BaseEmbHead


@HEADS.register_module()
class ContrastiveHead(BaseEmbHead):
    def __init__(self, temperature=0.1):
        super(ContrastiveHead, self).__init__()
        self.temperature = temperature

    def init_weights(self):
        pass

    def forward(self, pos, neg):
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature

        return logits
