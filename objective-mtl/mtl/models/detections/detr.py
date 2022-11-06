import torch
from torch import nn

from mtl.cores.bbox import bbox2result
from ..model_builder import DETECTORS
from .base_detectors import SingleStageDetector


@DETECTORS.register_module()
class DETR(SingleStageDetector):
    r"""Implementation of `DETR: End-to-End Object Detection with
    Transformers <https://arxiv.org/pdf/2005.12872>`_"""

    def __init__(self, cfg):
        super(DETR, self).__init__(cfg)

    def extract_feat(self, img, img_metas):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for module in self.neck:
                    x = module(x, img_metas)
            else:
                x = self.neck(x, img_metas)
        return x

    def extract_feats(self, imgs, img_metas):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [
            self.extract_feat(img, img_meta) for img, img_meta in zip(imgs, img_metas)
        ]

    def forward_dummy(self, img):
        """Used for computing network flops."""
        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(batch_input_shape=(height, width), img_shape=(height, width, 3))
            for _ in range(batch_size)
        ]
        x = self.extract_feat(img, dummy_img_metas)
        outs = self.bbox_head(x)
        return outs

    def forward_train(
        self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs
    ):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img, img_metas)
        losses = self.bbox_head.forward_train(
            x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore
        )
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.
        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta["batch_input_shape"] = batch_input_shape
        x = self.extract_feat(img, img_metas)
        outs = self.bbox_head(x)

        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            mlres_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
            return mlres_list
        else:
            bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
            return bbox_results
