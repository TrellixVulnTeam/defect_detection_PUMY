import torch
import torch.nn.functional as F

from ..model_builder import CLASSIFIERS
from .img_cls import ImageClassifier


@CLASSIFIERS.register_module()
class ImageFlexibleClassifier(ImageClassifier):
    """Image classifier with flexible input"""

    def extract_feat(self, img, **kwargs):
        """Directly extract features from the backbone + neck"""
        x = self.backbone(img, **kwargs)
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
        x = self.extract_feat(img, **kwargs)
        x = self.head(x)

        losses = dict()
        loss = self.get_losses(x, gt_label, **kwargs)
        losses.update(loss)

        return losses

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img, **kwargs)
        if isinstance(x, list):
            x = sum(x) / float(len(x))
        pred = F.softmax(x, dim=1) if x is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
