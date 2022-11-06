from torch import nn

from ..model_builder import DETECTORS
from .deformable_detr import DeformableDETR


@DETECTORS.register_module()
class SalientDeformableDETR(DeformableDETR):
    """Deformable detr with salient points"""

    def extract_feat(self, img, img_metas, gt_bboxes=None):
        """Extract features from images."""
        x = self.backbone(img)
        assert self.with_neck is True
        if isinstance(self.neck, nn.Sequential):
            for module in self.neck:
                if gt_bboxes is not None:
                    x = module(x, img_metas, gt_bboxes)
                else:
                    x = module(x, img_metas)
        else:
            if gt_bboxes is not None:
                x = self.neck(x, img_metas, gt_bboxes)
            else:
                x = self.neck(x, img_metas)
        return x

    def forward_train(
        self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs
    ):
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta["batch_input_shape"] = batch_input_shape

        outs = self.extract_feat(img, img_metas, gt_bboxes)
        losses = self.bbox_head.forward_train(
            *outs, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore
        )
        return losses
