import torch
import torch.nn as nn

from mtl.blocks.block_builder import build_backbone, build_neck
from ..model_builder import EMBEDDERS
from .base_emb import BaseEmbedder


@EMBEDDERS.register_module()
class KNNEmbedder(BaseEmbedder):
    def __init__(self, cfg):
        super(KNNEmbedder, self).__init__()
        self.type = cfg.TYPE
        self.backbone = build_backbone(cfg.BACKBONE)

        if len(cfg.NECK) > 0:
            self.neck = build_neck(cfg.NECK)

        if "PRETRAINED_MODEL_PATH" in cfg:
            if cfg.PRETRAINED_MODEL_PATH != "":
                self.init_weights(pretrained=cfg.PRETRAINED_MODEL_PATH)
            else:
                self.init_weights()
        else:
            self.init_weights()

    def init_weights(self, pretrained=None):
        super(KNNEmbedder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck"""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): of shape (N, 1) encoding the ground-truth label
                of input images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()
        loss = torch.Tensor([0]).to(x.device).sum()
        losses.update(loss)

        return losses

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img)
        return x
