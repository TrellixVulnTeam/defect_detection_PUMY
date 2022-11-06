import torch
import torch.nn as nn
import torch.distributed as dist
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from mtl.blocks.block_builder import build_backbone, build_head
from mtl.blocks.block_builder import build_loss, build_neck
from ..model_builder import EMBEDDERS
from .base_emb import BaseEmbedder


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


@torch.no_grad()
def distributed_sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    # Q = torch.exp(out / epsilon).t()
    clamp_out = torch.clamp(out / epsilon, max=10)
    Q = torch.exp(
        clamp_out
    ).t()  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * torch.distributed.get_world_size()  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= torch.clamp(sum_Q, min=0.0001)

    for _ in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= torch.clamp(sum_of_rows, min=0.0001)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.clamp(torch.sum(Q, dim=0, keepdim=True), min=0.0001)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


@EMBEDDERS.register_module()
class HVPMTLEmbedder(BaseEmbedder):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(self, cfg):
        super(HVPMTLEmbedder, self).__init__()

        self.backbone = build_backbone(cfg.BACKBONE)
        self.head = build_head(cfg.HEAD)
        self.compute_loss = build_loss(cfg.LOSS)

        # classification loss compute part
        self.position_embedding = build_backbone(cfg.EXTEND_BACKBONE)
        self.neck_mcls = build_neck(cfg.EXTEND_NECK)
        self.head_mcls = build_head(cfg.CLS_HEAD)
        self.head_mcls.init_weights()
        self.compute_cls_loss = build_loss(cfg.CLS_LOSS)

        # embedding
        self.emb_neck = build_neck(cfg.NECK)
        self.emb_head = build_head(cfg.EMB_HEAD)
        self.emb_head.init_weights()
        self.compute_emb_loss = build_loss(cfg.CROSS_LOSS)

        # teacher net
        self.backbone_t = build_backbone(cfg.BACKBONE)
        self.emb_neck_t = build_neck(cfg.NECK)
        self.emb_head_t = build_head(cfg.EMB_HEAD)
        self.neck_mcls_t = build_neck(cfg.EXTEND_NECK)
        self.head_mcls_t = build_head(cfg.CLS_HEAD)
        self.m_factor = cfg.EXTEND.momentum_factor
        self.compute_momentum_loss = build_loss(cfg.EXTEND_LOSS)

        if "PRETRAINED_MODEL_PATH" in cfg:
            if cfg.PRETRAINED_MODEL_PATH != "":
                self.init_weights(pretrained=cfg.PRETRAINED_MODEL_PATH)
            else:
                self.init_weights()
        else:
            self.init_weights()

        # initalize teacher model
        for param_q, param_k in zip(
            self.backbone.parameters(), self.backbone_t.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(
            self.neck_mcls.parameters(), self.neck_mcls_t.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(
            self.head_mcls.parameters(), self.head_mcls_t.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(
            self.emb_neck.parameters(), self.emb_neck_t.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(
            self.emb_head.parameters(), self.emb_head_t.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        super(HVPMTLEmbedder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.head.init_weights()
        self.apply(self._init_weights)

    @torch.no_grad()
    def _momentum_update_parameters(self):
        """Momentum update of the teacher model"""
        for param_q, param_k in zip(
            self.backbone.parameters(), self.backbone_t.parameters()
        ):
            param_k.data = param_k.data * self.m_factor + param_q.data * (
                1.0 - self.m_factor
            )

        for param_q, param_k in zip(
            self.neck_mcls.parameters(), self.neck_mcls_t.parameters()
        ):
            param_k.data = param_k.data * self.m_factor + param_q.data * (
                1.0 - self.m_factor
            )

        for param_q, param_k in zip(
            self.head_mcls.parameters(), self.head_mcls_t.parameters()
        ):
            param_k.data = param_k.data * self.m_factor + param_q.data * (
                1.0 - self.m_factor
            )

        for param_q, param_k in zip(
            self.emb_neck.parameters(), self.emb_neck_t.parameters()
        ):
            param_k.data = param_k.data * self.m_factor + param_q.data * (
                1.0 - self.m_factor
            )

        for param_q, param_k in zip(
            self.emb_head.parameters(), self.emb_head_t.parameters()
        ):
            param_k.data = param_k.data * self.m_factor + param_q.data * (
                1.0 - self.m_factor
            )

    def get_num_layers(self):
        return len(self.blocks)

    @property
    def with_neck(self):
        return False

    @property
    def with_head(self):
        return True

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "mask_token"}

    def forward_train(self, img, img_k, patch_mask, gt_label, **kwargs):
        """Forward computation during training."""
        x = self.backbone(img, patch_mask)
        losses = dict()

        # compute reconstruction loss
        x_rec = self.head(x)
        mask = patch_mask.unsqueeze(1).contiguous()
        loss = self.compute_loss(img, x_rec, mask, self.backbone.in_chans)
        losses["rec_loss"] = loss

        # compute classification loss
        x_mcls = self.neck_mcls(x)
        pos = self.position_embedding(x_mcls).to(x_mcls.dtype)
        cls_out = self.head_mcls(x_mcls, pos)
        cls_loss = self.compute_cls_loss(cls_out, gt_label)
        losses.update({"cls_loss": cls_loss})

        # compute contrastive loss
        x_emb = self.emb_neck(x)
        x_emb = self.emb_head(x_emb)[0]
        with torch.no_grad():
            self._momentum_update_parameters()
            q_x = distributed_sinkhorn(x_emb.detach())
            x_k = self.backbone_t(img_k)
            x_k_emb = self.emb_neck_t(x_k)
            x_k_emb = self.emb_head_t(x_k_emb)[0]
            q_x_k = distributed_sinkhorn(x_k_emb.detach())

        emb_loss = self.compute_emb_loss(x_emb, x_k_emb, q_x, q_x_k)
        losses.update({"emb_loss": emb_loss})

        # compute momentum loss
        with torch.no_grad():  # no gradient to keys
            x_t = self.backbone_t(img_k)
            x_t = self.neck_mcls_t(x_t)
            x_t_out = self.head_mcls_t(x_t, pos)
        mom_loss = self.compute_momentum_loss(cls_out, x_t_out)
        losses.update({"mom_loss": mom_loss})

        return losses

    def simple_test(self, img, patch_mask, **kwargs):
        x = self.backbone(img, patch_mask)
        x_rec = self.head(x)
        return x_rec
