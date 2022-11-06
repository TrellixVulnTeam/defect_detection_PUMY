import torch
import torch.nn as nn
from yacs.config import CfgNode

from mtl.blocks.block_builder import build_backbone, build_head, build_neck
from mtl.utils.config_util import convert_to_dict
from ...model_builder import DETECTORS
from .base_detector import BaseDetector


@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.
    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self, cfg):
        super(TwoStageDetector, self).__init__()
        self.type = cfg.TYPE
        if "BACKBONE" not in cfg:
            raise AttributeError("No BACKBONE definition in cfg!")
        backbone_dict = convert_to_dict(cfg.BACKBONE)
        self.backbone = build_backbone(backbone_dict)

        self.train_cfg = convert_to_dict(cfg.TRAIN_CFG)
        self.test_cfg = convert_to_dict(cfg.TEST_CFG)

        if "NECK" in cfg:
            neck_dict = convert_to_dict(cfg.NECK)
            self.neck = build_neck(neck_dict)

        if "RPN_HEAD" in cfg:
            rpn_train_cfg = (
                self.train_cfg["rpn"] if self.train_cfg is not None else None
            )
            rpn_test_cfg = self.test_cfg["rpn"] if self.test_cfg is not None else None
            rpn_head_ = convert_to_dict(cfg.RPN_HEAD)
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=rpn_test_cfg)
            self.rpn_head = build_head(rpn_head_)

        if "ROI_HEAD" in cfg:
            # update train and test cfg here
            rcnn_train_cfg = (
                self.train_cfg["rcnn"] if self.train_cfg is not None else None
            )
            rcnn_test_cfg = self.test_cfg["rcnn"] if self.test_cfg is not None else None
            roi_head_ = convert_to_dict(cfg.ROI_HEAD)
            roi_head_.update(train_cfg=rcnn_train_cfg, test_cfg=rcnn_test_cfg)
            self.roi_head = build_head(roi_head_)

        if "PRETRAINED_MODEL_PATH" in cfg:
            if cfg.PRETRAINED_MODEL_PATH != "":
                self.init_weights(pretrained=cfg.PRETRAINED_MODEL_PATH)
            else:
                self.init_weights()
        else:
            self.init_weights()

    @property
    def with_rpn(self):
        """Whether the detector has RPN"""
        return hasattr(self, "rpn_head") and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """Whether the detector has a RoI head"""
        return hasattr(self, "roi_head") and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def forward_dummy(self, img):
        """Used for computing network flops."""
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        proposals=None,
        **kwargs
    ):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see `Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            if isinstance(self.test_cfg, CfgNode):
                test_cfg_dict = convert_to_dict(self.test_cfg)
            else:
                test_cfg_dict = self.test_cfg

            proposal_cfg = self.train_cfg.get("rpn_proposal", test_cfg_dict["rpn"])

            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
            )
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            x,
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            gt_masks,
            **kwargs
        )
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(x, proposal_list, img_metas, rescale=rescale)
