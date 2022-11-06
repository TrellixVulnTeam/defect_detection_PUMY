# -*- encoding: utf-8 -*-

# --------------------------------------------------------
# @File    :   img_kd_q2l_cls.py
# @Time    :   2021/10/15 20:47:40
# @Content :   Head network for multilabel cls with kd
# @Author  :   Qian Zhiming
# @Contact :   zhiming.qian@micro-i.com.cn
# --------------------------------------------------------
import torch

from mtl.blocks.block_builder import build_backbone, build_neck, build_head, build_loss
from ..model_builder import CLASSIFIERS
from .img_q2l_cls import ImageQ2LClassifier


@CLASSIFIERS.register_module()
class ImageQ2LKDClassifier(ImageQ2LClassifier):
    """Multilabel classifier with Q2L and knowledge distillation"""

    def __init__(self, cfg):
        super(ImageQ2LKDClassifier, self).__init__(cfg)

        self.backbone_t = build_backbone(cfg.TEACHER_BACKBONE)
        print("load teacher net params")
        self.backbone_t.init_weights(pretrained=cfg.TEACHER_MODEL_PATH)
        for param_t in self.backbone_t.parameters():
            param_t.requires_grad = False

        self.neck_emb = build_neck(cfg.EXTEND_NECK)
        self.head_emb = build_head(cfg.EMB_HEAD)

        # kd loss
        self.compute_kd_loss = build_loss(cfg.EXTEND_LOSS)

    def init_weights(self, pretrained=None):
        super(ImageQ2LKDClassifier, self).init_weights(pretrained)

        # other heads can calls its own reset_param func to init weighs
        self.head_emb.init_weights()
        self.head_mcls.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)  # B x 768 x 7 x 7
        if isinstance(x, (list, tuple)):
            x = x[0]
        pos = self.position_embedding(x).to(x.dtype)
        out = self.head_mcls(x, pos)

        emb_x = self.neck_emb(x)
        emb_x = self.head_emb(emb_x)
        if isinstance(emb_x, tuple):
            emb_x = emb_x[-1]

        with torch.no_grad():
            emb_t = self.backbone_t(img)
            if isinstance(emb_t, tuple):
                emb_t = emb_t[-1]
        return out, emb_x, emb_t

    def forward_train(self, img, gt_label, **kwargs):
        out, emb_x, emb_t = self.extract_feat(img)

        losses = dict()
        cls_loss = self.compute_loss(out, gt_label)
        losses.update({"cls_loss": cls_loss})

        kd_loss = self.compute_kd_loss(emb_x, emb_t)
        losses.update({"kd_loss": kd_loss})
        return losses
