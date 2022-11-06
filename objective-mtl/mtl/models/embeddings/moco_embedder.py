import torch
import torch.nn as nn

from mtl.blocks.block_builder import build_backbone, build_head
from mtl.blocks.block_builder import build_neck, build_loss
from mtl.utils.metric_util import Accuracy
from ..model_builder import EMBEDDERS
from .base_emb import BaseEmbedder


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@EMBEDDERS.register_module()
class MocoEmbedder(BaseEmbedder):
    def __init__(self, cfg):
        super(MocoEmbedder, self).__init__()

        self.type = cfg.TYPE
        self.backbone_q = build_backbone(cfg.BACKBONE)
        self.backbone_k = build_backbone(cfg.BACKBONE)

        if len(cfg.NECK) > 0:
            self.neck_q = build_neck(cfg.NECK)
            self.neck_k = build_neck(cfg.NECK)

        if len(cfg.EMB_HEAD) > 0:
            self.head_q = build_head(cfg.EMB_HEAD)
            self.head_k = build_head(cfg.EMB_HEAD)

        if "PRETRAINED_MODEL_PATH" in cfg:
            if cfg.PRETRAINED_MODEL_PATH != "":
                self.init_weights(pretrained=cfg.PRETRAINED_MODEL_PATH)
            else:
                self.init_weights()
        else:
            self.init_weights()

        for param_q, param_k in zip(
            self.backbone_q.parameters(), self.backbone_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        if len(cfg.NECK) > 0:
            for param_q, param_k in zip(
                self.neck_q.parameters(), self.neck_k.parameters()
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

        if len(cfg.EMB_HEAD) > 0:
            for param_q, param_k in zip(
                self.head_q.parameters(), self.head_k.parameters()
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

        self.k = cfg.EXTEND.K
        self.m = cfg.EXTEND.m
        self.tempreture = cfg.EXTEND.T
        if len(cfg.EMB_HEAD) > 0:
            self.dim = cfg.EMB_HEAD.num_classes
        else:
            self.dim = cfg.BACKBONE.out_dim
        self.topk = cfg.EXTEND.topk
        if isinstance(self.topk, list):
            self.topk = tuple(self.topk)

        if len(cfg.LOSS) > 0:
            self.compute_loss = build_loss(cfg.LOSS)
        self.compute_accuracy = Accuracy(topk=self.topk)

        self.register_buffer("queue", torch.randn(self.dim, self.k))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @property
    def with_neck(self):
        return hasattr(self, "neck_q") and self.neck_q is not None

    @property
    def with_head(self):
        return hasattr(self, "head_q") and self.head_q is not None

    def init_weights(self, pretrained=None):
        super(MocoEmbedder, self).init_weights(pretrained)

        self.backbone_q.init_weights(pretrained=pretrained)
        if self.with_neck:
            self.neck_q.init_weights()
        if self.with_head:
            self.head_q.init_weights()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(
            self.backbone_q.parameters(), self.backbone_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        if self.with_neck:
            for param_q, param_k in zip(
                self.neck_q.parameters(), self.neck_k.parameters()
            ):
                param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        if self.with_head:
            for param_q, param_k in zip(
                self.head_q.parameters(), self.head_k.parameters()
            ):
                param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.k % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.k  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def extract_embedding_q(self, img):
        """Directly extract features from the backbone + neck"""
        x = self.backbone_q(img)
        if self.with_neck:
            x = self.neck_q(x)
        if self.with_head:
            x = self.head_q(x)

        return x

    def extract_embedding_k(self, img):
        """Directly extract features from the backbone + neck"""
        x = self.backbone_k(img)
        if self.with_neck:
            x = self.neck_k(x)
        if self.with_head:
            x = self.head_k(x)

        return x

    def forward_train(self, img, img_k, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        q = self.extract_embedding_q(img)
        num_max = torch.norm(q, p=float("inf")).detach()
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.extract_embedding_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.tempreture

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        losses = dict()
        # compute loss
        loss = self.compute_loss(logits, labels)
        # compute accuracy
        acc = self.compute_accuracy(logits, labels)
        assert len(acc) == len(self.topk)
        losses["loss"] = loss
        losses["accuracy"] = {f"top-{k}": a for k, a in zip(self.topk, acc)}
        losses["num_max"] = num_max

        return losses

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        x = self.extract_embedding_q(img)
        return x
