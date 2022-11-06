import torch

from mtl.blocks.block_builder import build_backbone, build_head
from mtl.blocks.block_builder import build_neck, build_loss
from ..model_builder import EMBEDDERS
from .base_emb import BaseEmbedder


@EMBEDDERS.register_module()
class DINOEmbedder(BaseEmbedder):
    def __init__(self, cfg):
        super(DINOEmbedder, self).__init__()

        self.type = cfg.TYPE
        self.backbone_t = build_backbone(cfg.BACKBONE)
        self.backbone_s = build_backbone(cfg.BACKBONE)

        if len(cfg.NECK) > 0:
            self.neck_t = build_neck(cfg.NECK)
            self.neck_s = build_neck(cfg.NECK)
        else:
            self.neck_t = None
            self.neck_s = None

        self.head_t = build_head(cfg.EXTEND_HEAD)
        self.head_s = build_head(cfg.EMB_HEAD)

        if "PRETRAINED_MODEL_PATH" in cfg:
            if cfg.PRETRAINED_MODEL_PATH != "":
                self.init_weights(pretrained=cfg.PRETRAINED_MODEL_PATH)
            else:
                self.init_weights()
        else:
            self.init_weights()

        for param_t, param_s in zip(
            self.backbone_t.parameters(), self.backbone_s.parameters()
        ):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False

        for param_t, param_s in zip(self.neck_t.parameters(), self.neck_s.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False

        for param_t, param_s in zip(self.head_t.parameters(), self.head_s.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False

        self.compute_loss = build_loss(cfg.LOSS)

        self.ncrops = cfg.LOSS.ncrops - 2

    @property
    def with_neck(self):
        return hasattr(self, "neck_s") and self.neck_s is not None

    @property
    def with_head(self):
        return True

    def init_weights(self, pretrained=None):
        super(DINOEmbedder, self).init_weights(pretrained)

        self.backbone_s.init_weights(pretrained=pretrained)
        if self.with_neck:
            self.neck_s.init_weights()
        self.head_s.init_weights()

    def forward_with_module(self, x, backbone, head, neck=None, return_feats=False):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]), return_counts=True
            )[1],
            0,
        )
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = backbone(torch.cat(x[start_idx:end_idx]))
            if neck is not None:
                _out = neck(_out)
            if isinstance(_out, tuple):
                _out = _out[0]
            if start_idx == 0 and return_feats:
                len_set = end_idx - start_idx
                out_feats = _out.chunk(len_set)[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx

        if return_feats:
            return head(output), out_feats
        else:
            # Run the head forward on the concatenated features.
            return head(output)

    def forward_train(self, img, img_k, img_s=None, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        imgs = [img, img_k]
        if img_s is not None:
            # img_s = img_s.chunk(self.ncrops)
            imgs.extend(img_s)

        student_output = self.forward_with_module(
            imgs, self.backbone_s, self.head_s, self.neck_s
        )

        # compute key features
        with torch.no_grad():  # no gradient to keys
            teacher_output = self.forward_with_module(
                imgs[:2], self.backbone_t, self.head_t, self.neck_t
            )  # only the 2 global views pass through the teacher

        losses = dict()
        # compute loss
        loss = self.compute_loss(student_output, teacher_output, kwargs["epoch"])

        losses["loss"] = loss

        return losses

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        return self.forward_with_module(
            [img], self.backbone_s, self.head_s, self.neck_s
        )
