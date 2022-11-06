from torch import nn

from ..block_builder import HEADS
from .base_emb_head import BaseEmbHead


@HEADS.register_module()
class MultiLevelPoolHead(BaseEmbHead):
    def __init__(self):
        super(MultiLevelPoolHead, self).__init__()
        self.pure_pool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        pass

    def forward(self, multilevel_feats):
        output_feats = []
        for i in range(len(multilevel_feats)):
            layer_feat = self.pure_pool(multilevel_feats[i])
            layer_feat = layer_feat.view(layer_feat.shape[0], -1)
            output_feats.append(layer_feat)

        return output_feats
