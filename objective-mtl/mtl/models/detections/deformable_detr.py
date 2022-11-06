import torch

from mtl.cores.bbox import bbox2result
from ..model_builder import DETECTORS
from .detr import DETR


@DETECTORS.register_module()
class DeformableDETR(DETR):
    def __init__(self, cfg):
        super(DeformableDETR, self).__init__(cfg)

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
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta["batch_input_shape"] = batch_input_shape
        outs = self.extract_feat(img, img_metas)
        losses = self.bbox_head.forward_train(
            *outs, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore
        )
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.
        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether toe information. rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta["batch_input_shape"] = batch_input_shape
        feat_outs = self.extract_feat(img, img_metas)
        x = feat_outs[0]
        bbox_predicts = feat_outs[1]
        class_scores = self.bbox_head(x)

        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            mlres_list = self.bbox_head.get_bboxes(
                class_scores, bbox_predicts, img_metas, rescale=rescale
            )
            return mlres_list
        else:
            bbox_list = self.bbox_head.get_bboxes(
                class_scores, bbox_predicts, img_metas, rescale=rescale
            )
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
            return bbox_results
