from torch import nn

from ..block_builder import HEADS
from .base_emb_head import BaseEmbHead


@HEADS.register_module()
class MultiLevelMlpHead(BaseEmbHead):
    def __init__(self, in_channels, hid_channels, out_channels):
        super(MultiLevelMlpHead, self).__init__()
        self.num_layers = len(in_channels)
        for i in range(self.num_layers):
            setattr(
                self,
                f"conv_pool_{i+1}",
                nn.Sequential(
                    nn.Conv2d(in_channels[i], hid_channels[i], 1),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.ReLU(inplace=True),
                ),
            )
            setattr(self, f"mlp_{i+1}", nn.Linear(hid_channels[i], out_channels[i], 1))

    def init_weights(self):
        pass

    def forward(self, multilevel_feats):
        output_feats = []
        for i in range(self.num_layers):
            conv_pool = getattr(self, f"conv_pool_{i+1}")
            mlp = getattr(self, f"mlp_{i+1}")
            layer_feat = conv_pool(multilevel_feats[i])
            layer_feat = layer_feat.view(layer_feat.shape[0], -1)
            layer_feat = mlp(layer_feat)
            output_feats.append(layer_feat)

        return output_feats
