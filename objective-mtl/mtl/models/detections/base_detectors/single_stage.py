import torch
import torch.nn as nn

from mtl.cores.bbox import bbox2result
from mtl.blocks.block_builder import build_backbone, build_head, build_neck
from mtl.utils.config_util import convert_to_dict
from ...model_builder import DETECTORS
from .base_detector import BaseDetector


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.
    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self, cfg):
        super(SingleStageDetector, self).__init__()
        if len(cfg.BACKBONE) <= 0:
            raise AttributeError("No BACKBONE definition in cfg!")
        self.type = cfg.TYPE
        self.backbone = build_backbone(cfg.BACKBONE)

        if len(cfg.NECK) > 0:
            self.neck = build_neck(cfg.NECK)

        if len(cfg.BBOX_HEAD) <= 0:
            raise AttributeError("No BBOX_HEAD definition in cfg!")

        self.train_cfg = convert_to_dict(cfg.TRAIN_CFG)
        self.test_cfg = convert_to_dict(cfg.TEST_CFG)
        default_args = {"train_cfg": self.train_cfg, "test_cfg": self.test_cfg}
        self.bbox_head = build_head(cfg.BBOX_HEAD, default_args)

        if "PRETRAINED_MODEL_PATH" in cfg:
            if cfg.PRETRAINED_MODEL_PATH != "":
                self.init_weights(pretrained=cfg.PRETRAINED_MODEL_PATH)
            else:
                self.init_weights()
        else:
            self.init_weights()

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def forward_dummy(self, img):
        """Used for computing network flops."""
        x = self.extract_feat(img)
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
        x = self.extract_feat(img)
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
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            mlres_list = self.bbox_head.get_bboxes(
                *outs, img_metas, rescale=rescale, with_nms=False
            )
            # return mlres_list
            tmp_list = []
            for mlres in mlres_list:
                tmp_res = torch.cat([mlres[0], mlres[2][:, None], mlres[1]], dim=1)
                tmp_list.append(tmp_res)

            onnx_res = torch.stack(tmp_list, dim=0)
            return onnx_res
        else:
            bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)

            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
            return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, "aug_test"), (
            f"{self.bbox_head.__class__.__name__}"
            " does not support test-time augmentation"
        )

        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]
