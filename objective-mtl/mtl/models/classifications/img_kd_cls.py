import torch

from mtl.blocks.block_builder import build_backbone, build_loss
from ..model_builder import CLASSIFIERS
from .img_cls import ImageClassifier


@CLASSIFIERS.register_module()
class ImageKDClassifier(ImageClassifier):
    """Knowledge distill for image classifier"""

    def __init__(self, cfg):
        super(ImageKDClassifier, self).__init__(cfg)

        self.backbone_t = build_backbone(cfg.EXTEND_BACKBONE)
        self.compute_kd_loss = build_loss(cfg.EXTEND_LOSS)

        print("load teacher net params")
        self.backbone_t.init_weights(pretrained=cfg.EXTEND_MODEL_PATH)
        for param_t in self.backbone_t.parameters():
            param_t.requires_grad = False

    def extract_feat(self, img, with_t=False):
        """Directly extract features from the backbone + neck"""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        if with_t:
            with torch.no_grad():
                t_embedding = self.backbone_t(img)
            return x, t_embedding
        else:
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
        if self.augmentor is not None:
            img, gt_label = self.augmentor(img, gt_label)

        x, t_embedding = self.extract_feat(img, with_t=True)
        out = self.head(x)

        losses = dict()
        if self.augmentor is not None:
            loss_cls = self.get_losses(out, gt_label, return_acc=False)
        else:
            loss_cls = self.get_losses(out, gt_label)
        if self.with_rdrop:
            out_tmp = self.head(x)
            loss_tmp = self.get_losses(out_tmp, gt_label)
            loss_cls = self._merge_loss(loss_cls, loss_tmp)
            loss_cls["loss_rdrop"] = self.compute_rdrop_loss(out, out_tmp)
        losses.update(loss_cls)

        if isinstance(x, (tuple, list)):
            kd_loss = self.compute_kd_loss(x[-1], t_embedding[-1])
        else:
            kd_loss = self.compute_kd_loss(x, t_embedding)
        losses.update({"loss_kd": kd_loss})

        return losses
