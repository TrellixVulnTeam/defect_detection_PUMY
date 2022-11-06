import torch
import torch.nn as nn
import torch.nn.functional as F

from mtl.blocks.block_builder import build_backbone, build_head, build_neck, build_loss
from mtl.utils.config_util import convert_to_dict
from mtl.utils.metric_util import Accuracy
from ..augmentors import AugmentConstructor
from ..model_builder import CLASSIFIERS
from .base_cls import BaseClassifier


@CLASSIFIERS.register_module()
class ImageClassifier(BaseClassifier):
    """Basic image classifier"""

    def __init__(self, cfg):
        super(ImageClassifier, self).__init__()
        self.type = cfg.TYPE
        self.backbone = build_backbone(cfg.BACKBONE)

        if len(cfg.NECK) > 0:
            self.neck = build_neck(cfg.NECK)

        if len(cfg.CLS_HEAD) > 0:
            self.head = build_head(cfg.CLS_HEAD)

        self.metric_dict = convert_to_dict(cfg.EXTEND)
        if "topk" in self.metric_dict:
            topk = self.metric_dict["topk"]
            if isinstance(topk, list):
                topk = tuple(topk)
            elif isinstance(topk, int):
                topk = (topk,)
            for _topk in topk:
                assert _topk > 0, "Top-k should be larger than 0"
            self.topk = topk
            self.compute_accuracy = Accuracy(topk=self.topk)
            self.return_accuracy = True
        else:
            self.return_accuracy = False

        self.compute_loss = build_loss(cfg.LOSS)
        self.with_rdrop = self.metric_dict.get("with_rdrop", False)
        if self.with_rdrop:
            if len(cfg.EXTEND_LOSS) > 0:
                rdrop_loss = cfg.EXTEND_LOSS
            else:
                rdrop_loss = dict(type="DualKLLoss", loss_weight=1.0)
            self.compute_rdrop_loss = build_loss(rdrop_loss)

        self.augmentor = None

        if len(cfg.TRAIN_CFG):
            train_cfg = convert_to_dict(cfg.TRAIN_CFG)
            augments_cfg = train_cfg.get("augmentor", None)
            if augments_cfg is not None:
                self.augmentor = AugmentConstructor(augments_cfg)

        if "PRETRAINED_MODEL_PATH" in cfg:
            if cfg.PRETRAINED_MODEL_PATH != "":
                self.init_weights(pretrained=cfg.PRETRAINED_MODEL_PATH)
            else:
                self.init_weights()
        else:
            self.init_weights()

    def init_weights(self, pretrained=None):
        super(ImageClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_head:
            self.head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck"""
        x = self.backbone(img)
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
        if self.augmentor is not None:
            img, gt_label = self.augmentor(img, gt_label)

        x = self.extract_feat(img)
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

        return losses

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img)
        x = self.head(x)
        if isinstance(x, list):
            x = sum(x) / float(len(x))
        pred = F.softmax(x, dim=1) if x is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def get_losses(self, x, gt_label, return_acc=True):
        num_samples = len(x)
        losses = dict()

        if isinstance(gt_label, list):
            gt_label = torch.tensor(
                gt_label, dtype=torch.long, device=gt_label[0].device
            )
        if len(gt_label.shape) > 1:
            gt_label = gt_label.squeeze()
        # compute loss
        loss_cls = self.compute_loss(x, gt_label, avg_factor=num_samples)
        losses["loss_cls"] = loss_cls
        # compute accuracy
        if return_acc and self.return_accuracy:
            acc = self.compute_accuracy(x, gt_label)
            assert len(acc) == len(self.topk)
            losses["accuracy"] = {f"top-{k}": a for k, a in zip(self.topk, acc)}
        return losses

    def _merge_loss(self, losses_1, losses_2):
        losses = dict()
        for key in losses_1:
            if isinstance(losses_1[key], dict):
                losses[key] = self._merge_loss(losses_1[key], losses_2[key])
            else:
                losses[key] = 0.5 * (losses_1[key] + losses_2[key])
        return losses
