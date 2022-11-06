import torch
import torch.nn as nn
from yacs.config import CfgNode

from mtl.utils.metric_util import accuracy
from mtl.cores.layer_ops.layer_resize import resize
from mtl.blocks.block_builder import build_head, build_loss
from mtl.utils.misc_util import add_prefix
from mtl.utils.config_util import convert_to_dict
from ..model_builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class CascadeEncoderDecoder(EncoderDecoder):
    """Cascade Encoder Decoder segmentors.

    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders of
    CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.
    """

    def __init__(self, cfg):
        self.num_stages = cfg.NUM_STAGES
        super(CascadeEncoderDecoder, self).__init__(cfg)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        if isinstance(decode_head, CfgNode):
            decode_head_dict = convert_to_dict(decode_head)
        else:
            decode_head_dict = decode_head

        assert len(decode_head_dict) == self.num_stages

        self.decode_head = nn.ModuleList()
        for key in decode_head_dict:
            self.decode_head.append(build_head(decode_head_dict[key]))

        self.align_corners = self.decode_head[-1].align_corners
        self.num_classes = self.decode_head[-1].num_classes

    def _init_loss(self, loss_dict):
        self.compute_loss_list = []
        for key in loss_dict:
            self.compute_loss_list.append(build_loss(loss_dict[key]))

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        self.backbone.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            self.decode_head[i].init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self.decode_head[0](x)
        for i in range(1, self.num_stages):
            out = self.decode_head[i](x, out)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return out

    def get_losses(self, seg_logit, seg_label, num_stage):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss["loss_seg"] = self.compute_loss_list[num_stage](
            seg_logit, seg_label, weight=seg_weight, ignore_index=self.ignore_index
        )
        loss["acc_seg"] = accuracy(seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        seg_logits = self.decode_head[0](x)
        seg_logits = torch.clamp(seg_logits, min=1e-5)
        loss_decode = self.get_losses(seg_logits, gt_semantic_seg, 0)

        losses.update(add_prefix(loss_decode, "decode_0"))

        for i in range(1, self.num_stages):
            # forward test again, maybe unnecessary for most methods.
            prev_outputs = self.decode_head[i - 1](x)
            seg_logits = self.decode_head[i](x, prev_outputs)
            loss_decode = self.get_losses(seg_logits, gt_semantic_seg, i)
            losses.update(add_prefix(loss_decode, f"decode_{i}"))

        return losses
