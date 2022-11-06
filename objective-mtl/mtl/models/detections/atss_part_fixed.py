import torch
import torch.nn as nn

from mtl.utils.log_util import print_log
from mtl.blocks.block_builder import build_backbone, build_head, build_neck
from mtl.utils.config_util import convert_to_dict
from ..model_builder import DETECTORS
from .base_detectors import SingleStageDetector


@DETECTORS.register_module()
class ATSSPartFixed(SingleStageDetector):
    """Fix part of model for atss"""

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

    def extract_feat(self, img):
        """Extract features from images."""
        if self.fixed_backbone:
            with torch.no_grad():
                x = self.backbone(img)
        else:
            x = self.backbone(img)
        if self.with_neck:
            if self.fixed_neck:
                with torch.no_grad():
                    x = self.neck(x)
            else:
                x = self.neck(x)
        return x
