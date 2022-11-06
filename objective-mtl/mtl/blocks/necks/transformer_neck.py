import torch
import torch.nn as nn
import torch.nn.functional as F

from mtl.utils.checkpoint_util import load_checkpoint
from mtl.utils.log_util import get_root_logger
from mtl.cores.layer_ops.transformer_layer import build_positional_encoding
from mtl.cores.layer_ops.transformer_ops import build_transformer
from ..block_builder import NECKS


@NECKS.register_module()
class TransformerNeck(nn.Module):
    def __init__(
        self,
        in_channels,
        num_query=100,
        transformer=None,
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True
        ),
    ):
        super(TransformerNeck, self).__init__()
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        self.input_proj = nn.Conv2d(in_channels, self.embed_dims, kernel_size=1)
        self.query_embedding = nn.Embedding(num_query, self.embed_dims)

        assert "num_feats" in positional_encoding
        num_feats = positional_encoding["num_feats"]
        assert num_feats * 2 == self.embed_dims, (
            "embed_dims should"
            f" be exactly 2 times of num_feats. Found {self.embed_dims}"
            f" and {num_feats}."
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
            self.transformer.init_weights()
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x, img_metas):
        """Forward function."""
        num_levels = len(x)
        batch_size = x[0].size(0)
        input_img_h, input_img_w = img_metas[0]["batch_input_shape"]
        srcs = []
        masks = []
        pos_embeds = []
        for i in range(num_levels):
            src, mask, pos = self._forward_single(
                x[i], img_metas, batch_size, input_img_h, input_img_w
            )
            srcs.append(src)
            masks.append(mask)
            pos_embeds.append(pos)

        # list[outs_dec]: [[nb_dec, bs, num_query, embed_dim]]
        outs_decs, _ = self.transformer(
            srcs, masks, self.query_embedding.weight, pos_embeds
        )

        return outs_decs[-1]

    def _forward_single(self, x, img_metas, batch_size, input_img_h, input_img_w):
        mask = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]["img_shape"]
            mask[img_id, :img_h, :img_w] = 0

        x = self.input_proj(x)
        # interpolate mask to have the same spatial shape with x
        mask = (
            F.interpolate(mask.unsqueeze(1), size=x.shape[-2:])
            .to(torch.bool)
            .squeeze(1)
        )
        # position encoding
        pos_embed = self.positional_encoding(mask)  # [bs, embed_dim, h, w]

        return x, mask, pos_embed


@NECKS.register_module()
class MultiscaleTransformerNeck(nn.Module):
    def __init__(
        self,
        in_channels,
        num_query=300,
        transformer=None,
        positional_encoding=dict(
            type="SinePositionalEncoding",
            num_feats=128,
            temperature=10000,
            normalize=True,
        ),
    ):
        super(MultiscaleTransformerNeck, self).__init__()
        if not isinstance(in_channels, list):
            in_channels = [in_channels]
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        self.input_proj = nn.ModuleList(
            [
                nn.Conv2d(in_channel, self.embed_dims, kernel_size=1)
                for in_channel in in_channels
            ]
        )

        assert "num_feats" in positional_encoding
        num_feats = positional_encoding["num_feats"]
        assert num_feats * 2 == self.embed_dims, (
            "embed_dims should"
            f" be exactly 2 times of num_feats. Found {self.embed_dims}"
            f" and {num_feats}."
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
            self.transformer.init_weights()
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x, img_metas, gt_bboxes=None):
        """Forward function."""
        num_levels = len(x)
        batch_size = x[0].size(0)
        input_img_h, input_img_w = img_metas[0]["batch_input_shape"]
        srcs = []
        masks = []
        pos_embeds = []

        for i in range(num_levels):
            src = self.input_proj[i](x[i])
            src, mask, pos = self._forward_single(
                src, img_metas, batch_size, input_img_h, input_img_w
            )
            srcs.append(src)
            masks.append(mask)
            pos_embeds.append(pos)
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        if gt_bboxes is None:
            outs_dec = self.transformer(srcs, masks, pos_embeds)
        else:
            outs_dec = self.transformer(srcs, masks, pos_embeds, img_metas, gt_bboxes)

        return outs_dec

    def _forward_single(self, src, img_metas, batch_size, input_img_h, input_img_w):
        mask = src.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]["img_shape"]
            mask[img_id, :img_h, :img_w] = 0

        # interpolate masks to have the same spatial shape with x
        mask = (
            F.interpolate(mask.unsqueeze(1), size=src.shape[-2:])
            .to(torch.bool)
            .squeeze(1)
        )
        # position encoding
        pos_embed = self.positional_encoding(mask)  # [bs, embed_dim, h, w]

        return src, mask, pos_embed
