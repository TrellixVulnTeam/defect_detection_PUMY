import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from mtl.blocks.block_builder import build_backbone, build_head
from mtl.blocks.block_builder import build_loss
from mtl.blocks.backbones.encoder_vit import get_sinusoid_encoding_table
from ..model_builder import EMBEDDERS
from .base_emb import BaseEmbedder


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


@EMBEDDERS.register_module()
class MaeEmbedder(BaseEmbedder):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(self, cfg):
        super(MaeEmbedder, self).__init__()

        self.backbone = build_backbone(cfg.BACKBONE)
        self.head = build_head(cfg.HEAD)

        self.encoder_to_decoder = nn.Linear(
            cfg.BACKBONE.embed_dim, cfg.HEAD.embed_dim, bias=False
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.HEAD.embed_dim))
        self.pos_embed = get_sinusoid_encoding_table(
            cfg.HEAD.num_patches, cfg.HEAD.embed_dim
        )
        self.compute_loss = build_loss(cfg.LOSS)

        if "PRETRAINED_MODEL_PATH" in cfg:
            if cfg.PRETRAINED_MODEL_PATH != "":
                self.init_weights(pretrained=cfg.PRETRAINED_MODEL_PATH)
            else:
                self.init_weights()
        else:
            self.init_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        super(MaeEmbedder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.head.init_weights()
        self.apply(self._init_weights)
        trunc_normal_(self.mask_token, std=0.02)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "mask_token"}

    def forward_with_module(self, img, mask):
        x_vis = self.backbone(img, mask)  # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]

        B, _, C = x_vis.shape
        # # TODO: raise the in-place error
        # x_full = self.mask_token.repeat(B, N, 1)
        # x_full[mask] += x_mask

        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = (
            self.pos_embed.expand(B, -1, -1)
            .type_as(img)
            .to(img.device)
            .clone()
            .detach()
        )
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)

        x = self.head(x_full, pos_emd_mask.shape[1])  # [B, N_mask, 3 * 16 * 16]

        return x

    def forward_train(self, img, patch_labels, patch_mask, **kwargs):
        """Forward computation during training."""
        x = self.forward_with_module(img, patch_mask)
        losses = dict()
        # compute loss
        loss = self.compute_loss(x, patch_labels)
        losses["loss"] = loss
        return losses

    def simple_test(self, img, patch_mask, **kwargs):
        return self.forward_with_module(img, patch_mask)
