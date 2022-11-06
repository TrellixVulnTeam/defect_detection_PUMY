# -*- encoding: utf-8 -*-

# --------------------------------------------------------
# @File    :   img_q2l_cls.py
# @Time    :   2021/10/15 20:46:49
# @Content :   Head network for multilabel classification
# @Author  :   Qian Zhiming
# @Contact :   zhiming.qian@micro-i.com.cn
# --------------------------------------------------------

from mtl.blocks.block_builder import build_backbone, build_head, build_loss
from ..model_builder import CLASSIFIERS
from .base_cls import BaseClassifier


@CLASSIFIERS.register_module()
class ImageQ2LClassifier(BaseClassifier):
    """Multilabel classifier with Q2L and knowledge distillation"""

    def __init__(self, cfg):
        super(ImageQ2LClassifier, self).__init__()
        self.type = cfg.TYPE

        # backbone part
        self.backbone = build_backbone(cfg.BACKBONE)
        self.position_embedding = build_backbone(cfg.EXTEND_BACKBONE)

        # transformer part
        self.head_mcls = build_head(cfg.CLS_HEAD)

        # classification loss compute part
        self.compute_loss = build_loss(cfg.LOSS)

        if "PRETRAINED_MODEL_PATH" in cfg:
            if cfg.PRETRAINED_MODEL_PATH != "":
                self.init_weights(pretrained=cfg.PRETRAINED_MODEL_PATH)
            else:
                self.init_weights()
        else:
            self.init_weights()

    def init_weights(self, pretrained=None):
        super(ImageQ2LClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)

        # in transformer head, the transformer calls its own reset_param func
        # to init weighs self.transformer_head.init_weights()

    def extract_feat(self, img):
        """Get the outputs of mcls module

        Args:
            img (torch.Tensor): input image

        Returns:
            torch.Tensor (B x 63): features for calculate class probilities
        """
        x = self.backbone(img)  # B x 768 x 7 x 7
        if isinstance(x, (list, tuple)):
            x = x[-1]
        pos = self.position_embedding(x).to(x.dtype)
        out = self.head_mcls(x, pos)
        return out

    def forward_train(self, img, gt_label, **kwargs):
        out = self.extract_feat(img)

        losses = dict()
        cls_loss = self.compute_loss(out, gt_label)

        losses.update({"cls_loss": cls_loss})
        # print(losses)

        return losses

    def simple_test(self, img, **kwargs):
        """Test for inference
        Args:
            img (torch.Tensor): input images

        Returns:
            torch.Tensor(B x 63): class probilities of input images
        """
        x = self.backbone(img)  # B x 768 x 7 x 7
        if isinstance(x, (list, tuple)):
            x = x[0]
        pos = self.position_embedding(x).to(x.dtype)

        cls_score = self(x, pos)
        pred = F.sigmoid(cls_score) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
