import torch
from yacs.config import CfgNode

from mtl.blocks.block_builder import build_backbone, build_neck, build_loss
from mtl.utils.config_util import convert_to_dict
from ..model_builder import DETECTORS
from .base_detectors import TwoStageDetector


@DETECTORS.register_module()
class FasterRCNNKD(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self, cfg):
        super(FasterRCNNKD, self).__init__(cfg)
        self.backbone_t = build_backbone(cfg.TEACHER_BACKBONE)
        self.neck_ext = build_neck(cfg.EXTEND_NECK)
        self.compute_loss = build_loss(cfg.LOSS)

        print("load teacher net params")
        self.backbone_t.init_weights(pretrained=cfg.TEACHER_MODEL_PATH)
        self.neck_ext.init_weights()
        for param_t in self.backbone_t.parameters():
            param_t.requires_grad = False

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck."""

        s_feats = self.backbone(img)
        x = self.neck(s_feats)
        if self.training:
            s_feats = self.neck_ext(s_feats)
            with torch.no_grad():
                t_feats = self.backbone_t(img)
            return x, s_feats, t_feats[-1]
        else:
            return x

    def forward_train(
        self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None
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
        x, s_feats, t_feats = self.extract_feat(img)
        losses = dict()

        # RPN forward and loss
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

        roi_losses = self.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore
        )
        losses.update(roi_losses)

        if isinstance(s_feats, (tuple, list)):
            kd_loss = self.compute_loss(s_feats[-1], t_feats[-1])
        else:
            kd_loss = self.compute_loss(s_feats, t_feats)
        losses.update({"kd_loss": kd_loss})
        return losses
