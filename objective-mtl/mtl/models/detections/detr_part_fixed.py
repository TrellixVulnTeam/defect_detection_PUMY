import torch
import torch.nn as nn
import warnings

from mtl.cores.bbox import bbox2result
from mtl.utils.log_util import print_log
from mtl.blocks.block_builder import build_backbone, build_head, build_neck
from mtl.utils.config_util import convert_to_dict
from ..model_builder import DETECTORS
from .base_detectors import SingleStageDetector


@DETECTORS.register_module()
class DETRPartFixed(SingleStageDetector):
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

        if "PRETRAINED_BACKBONE_PATH" in cfg:
            backbone_pretrained = cfg.PRETRAINED_BACKBONE_PATH
        else:
            backbone_pretrained = ""
        if "PRETRAINED_NECK_PATH" in cfg:
            neck_pretrained = cfg.PRETRAINED_NECK_PATH
        else:
            neck_pretrained = ""
        self.init_weights(backbone_pretrained, neck_pretrained)

        self.fixed_backbone = False
        self.fixed_neck = False
        if "fixed_backbone" in cfg.EXTEND:
            if cfg.EXTEND.fixed_backbone:
                self.fixed_backbone = True
                for param_q in self.backbone.parameters():
                    param_q.requires_grad = False
        if "fixed_neck" in cfg.EXTEND:
            if cfg.EXTEND.fixed_neck:
                self.fixed_neck = True
                for param_q in self.neck.parameters():
                    param_q.requires_grad = False

    def init_weights(self, backbone_pretrained="", neck_pretrained=""):
        """Initialize the weights in detector."""
        if backbone_pretrained != "":
            print_log(f"load backbone from: {backbone_pretrained}", logger="root")
            self.backbone.init_weights(pretrained=backbone_pretrained)
        else:
            self.backbone.init_weights()
        if self.with_neck:
            if neck_pretrained != "":
                print_log(f"load neck from: {neck_pretrained}", logger="root")
                if isinstance(self.neck, nn.Sequential):
                    assert isinstance(neck_pretrained, list)
                    neck_idx = 0
                    for m in self.neck:
                        m.init_weights(neck_pretrained[neck_idx])
                        neck_idx += 1
                else:
                    self.neck.init_weights(pretrained=neck_pretrained)
            elif isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img, img_metas):
        """Extract features from images."""
        if self.fixed_backbone:
            with torch.no_grad():
                x = self.backbone(img)
        else:
            x = self.backbone(img)
        if self.with_neck:
            if self.fixed_neck:
                with torch.no_grad():
                    x = self.neck(x, img_metas)
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
        warnings.warn(
            "Warning! MultiheadAttention in DETR does not "
            "support flops computation! Do not use the "
            "results in your papers!"
        )

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
            mlres_list = self.bbox_head.get_bboxes(
                *outs, img_metas, rescale=rescale, with_nms=False
            )
            return mlres_list
        else:
            bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
            return bbox_results
