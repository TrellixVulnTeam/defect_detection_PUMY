import torch

from mtl.blocks.block_builder import build_head
from ..model_builder import EMBEDDERS
from .dino_embedder import DINOEmbedder


@EMBEDDERS.register_module()
class DINOCLSEmbedder(DINOEmbedder):
    def __init__(self, cfg):
        super(DINOCLSEmbedder, self).__init__(cfg)

        self.head_cls = build_head(cfg.CLS_HEAD)
        self.head_cls.init_weights()

    def init_weights(self, pretrained=None):
        super(DINOCLSEmbedder, self).init_weights(pretrained)

    def forward_train(self, img, gt_label, img_k, img_s=None, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        imgs = [img, img_k]
        if img_s is not None:
            # img_s = img_s.chunk(self.ncrops)
            imgs.extend(img_s)

        student_output, out_feats = self.forward_with_module(
            imgs, self.backbone_s, self.head_s, self.neck_s, return_feats=True
        )

        # compute key features
        with torch.no_grad():  # no gradient to keys
            teacher_output = self.forward_with_module(
                imgs[:2], self.backbone_t, self.head_t, self.neck_t
            )  # only the 2 global views pass through the teacher

        losses = dict()
        # compute loss
        loss_cls = self.head_cls.forward_train(out_feats, gt_label)
        loss_ssl = self.compute_loss(student_output, teacher_output, kwargs["epoch"])
        losses.update(loss_cls)
        losses["loss"] += loss_ssl

        return losses

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""

        _, img_feats = self.forward_with_module(
            [img], self.backbone_s, self.head_s, self.neck_s, return_feats=True
        )
        return self.head_cls.simple_test(img_feats)
