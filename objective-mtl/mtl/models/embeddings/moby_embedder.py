import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from diffdist import functional

from mtl.blocks.block_builder import build_backbone, build_head
from mtl.blocks.block_builder import build_neck
from ..model_builder import EMBEDDERS
from .base_emb import BaseEmbedder


def dist_collect(x):
    """collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [
        torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous()
        for _ in range(dist.get_world_size())
    ]
    out_list = functional.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()


@EMBEDDERS.register_module()
class MobyEmbedder(BaseEmbedder):
    def __init__(self, cfg):
        super(MobyEmbedder, self).__init__()

        self.type = cfg.TYPE

        # transformer backbone encoder
        # encoder -> online
        # encoder_k -> target

        self.encoder = build_backbone(cfg.BACKBONE)
        self.encoder_k = build_backbone(cfg.BACKBONE)

        # Neck for AvgPoooling
        if len(cfg.NECK) > 0:
            self.neck = build_neck(cfg.NECK)
            self.neck_k = build_neck(cfg.NECK)
        else:
            self.neck = None
            self.neck_k = None

        # Head for projector and predictor
        self.contrast_momentum = 0.99
        self.contrast_temperature = 0.2
        self.contrast_num_negative = 16384

        self.head = build_head(cfg.EMB_HEAD)
        self.predictor = build_head(cfg.EXTEND_HEAD)

        self.head_k = build_head(cfg.EMB_HEAD)

        if "PRETRAINED_MODEL_PATH" in cfg:
            if cfg.PRETRAINED_MODEL_PATH != "":
                self.init_weights(pretrained=cfg.PRETRAINED_MODEL_PATH)
            else:
                self.init_weights()
        else:
            self.init_weights()

        for param_q, param_k in zip(
            self.encoder.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.neck.parameters(), self.neck_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.head.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # nn.SyncBatchNorm.convert_sync_batchnorm(self.head)
        # nn.SyncBatchNorm.convert_sync_batchnorm(self.head_k)
        # nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

        self.all_k = 65536

        self.mini_k = 0

        # create the queue
        self.register_buffer("queue1", torch.randn(256, self.contrast_num_negative))
        self.register_buffer("queue2", torch.randn(256, self.contrast_num_negative))
        self.queue1 = F.normalize(self.queue1, dim=0)
        self.queue2 = F.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @property
    def with_neck(self):
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_head(self):
        return True

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        _contrast_momentum = (
            1.0
            - (1.0 - self.contrast_momentum)
            * (np.cos(np.pi * self.mini_k / self.all_k) + 1)
            / 2.0
        )
        self.mini_k = self.mini_k + 1

        for param_q, param_k in zip(
            self.encoder.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (
                1.0 - _contrast_momentum
            )

        for param_q, param_k in zip(self.head.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (
                1.0 - _contrast_momentum
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys1, keys2):
        # gather keys before updating queue

        keys1 = dist_collect(keys1)
        keys2 = dist_collect(keys2)

        batch_size = keys1.shape[0]

        ptr = int(self.queue_ptr)
        assert self.contrast_num_negative % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue1[:, ptr : ptr + batch_size] = keys1.T
        self.queue2[:, ptr : ptr + batch_size] = keys2.T
        ptr = (ptr + batch_size) % self.contrast_num_negative  # move pointer

        self.queue_ptr[0] = ptr

    def contrastive_loss(self, q, k, queue):

        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.contrast_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return F.cross_entropy(logits, labels)

    def forward_train(self, img, img_k, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feat_1 = self.encoder(img)  # queries: NxC
        if self.with_neck:
            feat_1 = self.neck(feat_1)
        proj_1 = self.head(feat_1)
        pred_1 = self.predictor(proj_1)
        pred_1 = F.normalize(pred_1, dim=1)

        feat_2 = self.encoder(img_k)
        if self.with_neck:
            feat_2 = self.neck(feat_2)
        proj_2 = self.head(feat_2)
        pred_2 = self.predictor(proj_2)
        pred_2 = F.normalize(pred_2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            feat_1_ng = self.encoder_k(img)  # keys: NxC
            if self.with_neck:
                feat_1_ng = self.neck_k(feat_1_ng)
            proj_1_ng = self.head_k(feat_1_ng)
            proj_1_ng = F.normalize(proj_1_ng, dim=1)

            feat_2_ng = self.encoder_k(img_k)
            if self.with_neck:
                feat_2_ng = self.neck_k(feat_2_ng)
            proj_2_ng = self.head_k(feat_2_ng)
            proj_2_ng = F.normalize(proj_2_ng, dim=1)

        # compute loss
        loss = self.contrastive_loss(
            pred_1, proj_2_ng, self.queue2
        ) + self.contrastive_loss(pred_2, proj_1_ng, self.queue1)

        self._dequeue_and_enqueue(proj_1_ng, proj_2_ng)

        losses = dict()
        losses["loss"] = loss

        return losses

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""

        x = self.encoder(img)
        if self.with_neck:
            x = self.neck(x)
        return x
