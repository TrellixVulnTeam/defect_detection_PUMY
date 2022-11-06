import torch

from mtl.blocks.block_builder import build_backbone, build_head, build_loss
from ..model_builder import EMBEDDERS
from .base_emb import BaseEmbedder


@EMBEDDERS.register_module()
class DistillEmbedder(BaseEmbedder):
    """Distillation"""

    def __init__(self, cfg):
        super(DistillEmbedder, self).__init__()
        self.backbone = build_backbone(cfg.BACKBONE)
        self.backbone_t = build_backbone(cfg.TEACHER_BACKBONE)
        self.head = build_head(cfg.EMB_HEAD)
        self.head_t = build_head(cfg.EXTEND_HEAD)

        print("load teacher net params")
        self.backbone_t.init_weights(pretrained=cfg.EXTEND_MODEL_PATH)
        for param_t in self.backbone_t.parameters():
            param_t.requires_grad = False

        if "PRETRAINED_MODEL_PATH" in cfg:
            if cfg.PRETRAINED_MODEL_PATH != "":
                self.init_weights(pretrained=cfg.PRETRAINED_MODEL_PATH)
            else:
                self.init_weights()
        else:
            self.init_weights()

        self.compute_loss = build_loss(cfg.LOSS)

    def init_weights(self, pretrained=None):
        super(DistillEmbedder, self).init_weights(pretrained)

    @property
    def with_neck(self):
        return False

    @property
    def with_head(self):
        return True

    def forward_train(self, img, **kwargs):
        x = self.backbone(img)
        x = self.head(x)

        with torch.no_grad():
            x_t = self.backbone_t(img)
            x_t = self.head_t(x_t)

        losses = dict()
        loss = self.compute_loss(x, x_t)

        losses["emb_loss"] = loss

        return losses

    def simple_test(self, img, **kwargs):
        x = self.backbone(img)
        x = self.head(x)
        return x
