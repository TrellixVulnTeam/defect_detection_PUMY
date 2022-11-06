# -*- encoding: utf-8 -*-

# --------------------------------------------------------
# @File    :   transformer_ops.py
# @Time    :   2022/04/11 18:52:26
# @Content :
# @Author  :   Qian Zhiming
# @Contact :   zhiming.qian@micro-i.com.cn
# --------------------------------------------------------

import math
import copy
import torch
from torch import nn

from mtl.cores.ops.ops_builder import build_norm_layer
from mtl.utils.reg_util import Registry, build_module_from_dict
from mtl.utils.init_util import xavier_init
from mtl.utils.misc_util import inverse_sigmoid
from mtl.utils.misc_util import multi_apply
from .transformer_layer import MSDeformAttn, build_transformer_layer, MLP

TRANSFORMER_LAYER_SEQUENCE = Registry("transformer-layers sequence")
TRANSFORMER = Registry("Transformer")


def build_transformer_layer_sequence(cfg, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return build_module_from_dict(cfg, TRANSFORMER_LAYER_SEQUENCE, default_args)


def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_module_from_dict(cfg, TRANSFORMER, default_args)


def _get_clones(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TransformerLayerSequence(nn.Module):
    """Base class for TransformerEncoder and TransformerDecoder in vision
    transformer.

    As base-class of Encoder and Decoder in vision transformer.
    Support customization such as specifying different kind
    of `transformer_layer` in `transformer_coder`.

    Args:
        transformerlayer ( |
            obj:): Config of transformerlayer
            in TransformerCoder. If it is obj:,
             it would be repeated `num_layer` times to a
             list[]. Default: None.
        num_layers (int): The number of `TransformerLayer`. Default: None.
            Default: None.
    """

    def __init__(self, transformerlayers=None, num_layers=None):
        super(TransformerLayerSequence, self).__init__()
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert (
                isinstance(transformerlayers, list)
                and len(transformerlayers) == num_layers
            )
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(build_transformer_layer(transformerlayers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for `TransformerCoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_queries, bs, embed_dims)`.
            key (Tensor): The key tensor with shape
                `(num_keys, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_keys, bs, embed_dims)`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor], optional): Each element is 2D Tensor
                which is used in calculation of corresponding attention in
                operation_order. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in self-attention
                Default: None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor:  results with shape [num_queries, bs, embed_dims].
        """
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DETRTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of DETR.

    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(self, *args, post_norm_cfg=dict(type="LN"), **kwargs):
        super(DETRTransformerEncoder, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = (
                build_norm_layer(post_norm_cfg, self.embed_dims)[1]
                if self.pre_norm
                else None
            )
        else:
            assert not self.pre_norm, (
                f"Use prenorm in "
                f"{self.__class__.__name__},"
                f"Please specify post_norm_cfg"
            )
            self.post_norm = None

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(DETRTransformerEncoder, self).forward(*args, **kwargs)
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DETRTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(
        self, *args, post_norm_cfg=dict(type="LN"), return_intermediate=False, **kwargs
    ):
        super(DETRTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg, self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self, query, *args, **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        intermediate = []
        for layer in self.layers:
            query = layer(query, *args, **kwargs)
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
        return torch.stack(intermediate)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DeformableDETRTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, hidden_dim=256, return_intermediate=False, **kwargs):

        super(DeformableDETRTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        bbox_embed = MLP(hidden_dim, hidden_dim, output_dim=4, num_layers=3)
        nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        self.bbox_embed = _get_clones(bbox_embed, self.num_layers)

    def forward(self, query, *args, reference_points=None, valid_ratios=None, **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points[:, :, None] * valid_ratios[:, None]
                )
            output = layer(
                output, *args, reference_points=reference_points_input, **kwargs
            )
            output = output.permute(1, 0, 2)

            tmp = self.bbox_embed[lid](output)
            new_reference_points = tmp + inverse_sigmoid(reference_points)
            new_reference_points = new_reference_points.sigmoid()
            reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(new_reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        else:
            return output, reference_points


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class SparseDeformableDETRTransformerEncoder(TransformerLayerSequence):
    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        reference_points=None,
        topk_inds=None,
        sparse_token_nums=None,
        **kwargs,
    ):
        sparsified_keys = False if topk_inds is None else True
        output = value
        if sparsified_keys:
            assert topk_inds is not None
            B_, N_, S_, P_ = reference_points.shape
            reference_points = torch.gather(
                reference_points.view(B_, N_, -1),
                1,
                topk_inds.unsqueeze(-1).repeat(1, 1, S_ * P_),
            ).view(B_, -1, S_, P_)
            tgt = torch.gather(
                query.permute(1, 0, 2),
                1,
                topk_inds.unsqueeze(-1).repeat(1, 1, query.size(-1)),
            ).permute(1, 0, 2)
            tgt_pos = torch.gather(
                query_pos.permute(1, 0, 2),
                1,
                topk_inds.unsqueeze(-1).repeat(1, 1, query_pos.size(-1)),
            ).permute(1, 0, 2)
        else:
            tgt = query
            tgt_pos = query_pos

        for layer in self.layers:
            tgt = layer(
                tgt,
                key,
                output,
                tgt_pos,
                key_pos,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                reference_points=reference_points,
                **kwargs,
            )
            if sparsified_keys:
                output_tmp = output.permute(1, 0, 2)
                if sparse_token_nums is None:
                    output = output_tmp.scatter(
                        1,
                        topk_inds.unsqueeze(-1).repeat(1, 1, tgt.size(-1)),
                        tgt.permute(1, 0, 2),
                    )
                else:
                    outputs = []
                    tgt = tgt.permute(1, 0, 2)
                    for i in range(topk_inds.shape[0]):
                        outputs.append(
                            output_tmp[i].scatter(
                                0,
                                topk_inds[i][: sparse_token_nums[i]]
                                .unsqueeze(-1)
                                .repeat(1, tgt.size(-1)),
                                tgt[i][: sparse_token_nums[i]],
                            )
                        )
                    output = torch.stack(outputs)
                    tgt = tgt.permute(1, 0, 2)
                output = output.permute(1, 0, 2)
            else:
                output = tgt

        return output


@TRANSFORMER.register_module()
class DETRTransformer(nn.Module):
    """Implements the DETR transformer.

    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:

        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        encoder (Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder (Dict): Config of
            TransformerDecoder. Defaults to None
    """

    def __init__(self, encoder=None, decoder=None):
        super(DETRTransformer, self).__init__()
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.encoder.embed_dims

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, "weight") and m.weight.dim() > 1:
                xavier_init(m, distribution="uniform")
        self._is_init = True

    def forward(self, srcs, masks, query_embed, pos_embeds):
        """Forward function for `Transformer`.

        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        outs = []
        memories = []
        for x, mask, pos_embed in zip(srcs, masks, pos_embeds):
            bs, c, h, w = x.shape
            # use `view` instead of `flatten` for dynamically exporting to ONNX
            x = x.view(bs, c, -1).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
            pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
            query_embed = query_embed.unsqueeze(1).repeat(
                1, bs, 1
            )  # [num_query, dim] -> [num_query, bs, dim]
            mask = mask.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
            memory = self.encoder(
                query=x,
                key=None,
                value=None,
                query_pos=pos_embed,
                query_key_padding_mask=mask,
            )
            target = torch.zeros_like(query_embed)
            # out_dec: [num_layers, num_query, bs, dim]
            out_dec = self.decoder(
                query=target,
                key=memory,
                value=memory,
                key_pos=pos_embed,
                query_pos=query_embed,
                key_padding_mask=mask,
            )
            out_dec = out_dec.transpose(1, 2)
            memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
            outs.append(out_dec)
            memories.append(memory)
        return outs, memories


class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, h_dim), nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1),
        )

    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out


@TRANSFORMER.register_module()
class DeformableDETRTransformer(nn.Module):
    def __init__(
        self,
        encoder=None,
        decoder=None,
        num_feature_levels=4,
        num_classes=1,
        temperature=10000,
        num_pos_feats=128,
        topk=300,
    ):
        super().__init__()
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.temperature = temperature
        self.num_pos_feats = num_pos_feats
        self.embed_dims = self.encoder.embed_dims
        self.topk_num = topk

        self.level_embed = nn.Parameter(
            torch.Tensor(num_feature_levels, self.embed_dims)
        )

        self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
        self.enc_output_norm = nn.LayerNorm(self.embed_dims)
        self.pos_trans = nn.Linear(self.embed_dims * 2, self.embed_dims * 2)
        self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)

        self.class_object_proposal = nn.Linear(
            self.embed_dims, num_classes
        )  # for select obj
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_object_proposal.bias.data = torch.ones(num_classes) * bias_value
        self.bbox_proposal = MLP(
            self.embed_dims, self.embed_dims, output_dim=4, num_layers=3
        )
        nn.init.constant_(self.bbox_proposal.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_proposal.layers[-1].bias.data, 0)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m.init_weights()
        nn.init.normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals, is_sigmoid=False):
        # proposals: N, L(top_k), 4(bbox coords.)
        scale = 2 * math.pi
        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=proposals.device
        )  # 128
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        if is_sigmoid:
            sigmoid_proposals = proposals * scale  # N, L, 4
        else:
            sigmoid_proposals = proposals.sigmoid() * scale  # N, L, 4
        pos = sigmoid_proposals[:, :, :, None] / dim_t  # N, L, 4, 128
        # apply sin/cos alternatively
        pos = torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4
        )  # N, L, 4, 64, 2
        pos = pos.flatten(2)  # N, L, 512 (4 x 128)
        return pos

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def gen_encoder_proposals(self, memory, memory_padding_mask, spatial_shapes):
        """Make region proposals for each multi-scale features considering their shapes and padding masks,
        and project & normalize the feats corresponding to these proposals.
            - center points: relative grid coordinates in the range of [0.01, 0.99] (additional mask)
            - width/height:  2^(layer_id) * s (s=0.05) / see the appendix A.4

        Tensor shape example:
            Args:
                memory: torch.Size([2, 15060, 256])
                memory_padding_mask: torch.Size([2, 15060])
                spatial_shape: torch.Size([4, 2])
            Returns:
                output_memory: torch.Size([2, 15060, 256])
                    - same shape with memory ( + additional mask + linear layer + layer norm )
                output_proposals: torch.Size([2, 15060, 4])
                    - x, y, w, h
        """
        N_, S_, C_ = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # level of encoded feature scale
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(
                N_, H_, W_, 1
            )
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H_ - 1, H_, dtype=torch.float32, device=memory.device
                ),
                torch.linspace(
                    0, W_ - 1, W_, dtype=torch.float32, device=memory.device
                ),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(
                N_, 1, 1, 2
            )
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += H_ * W_

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)
        ).all(-1, keepdim=True)
        output_proposals = torch.log(
            output_proposals / (1 - output_proposals)
        )  # inverse of sigmoid
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float("inf")
        )  # sigmoid(inf) = 1

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0)
        )
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals, (~memory_padding_mask).sum(axis=-1)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, mlvl_feats, mlvl_masks, mlvl_pos_embeds, **kwargs):
        # prepare input for encoder
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        # valid ratios across multi-scale features of the same image can be varied,
        # while they are interpolated and binarized on different resolutions.
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )
        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2
        )  # (H*W, bs, embed_dims)

        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        # top scoring bounding boxes are picked as the final region proposals.
        # these proposals are fed into the decoder as initial boxes for the iterative bounding box refinement.

        # finalize the first stage output
        # project & normalize the memory and make proposal bounding boxes on them
        output_memory, output_proposals, _ = self.gen_encoder_proposals(
            memory, mask_flatten, spatial_shapes
        )

        # hack implementation for two-stage Deformable DETR (using the last layer registered in class/bbox_embed)
        # 1) a linear projection for bounding box binary classification (fore/background)
        enc_outputs_class = self.class_object_proposal(output_memory)
        enc_outputs_fg_class = enc_outputs_class[
            ..., 0
        ]  # enc_outputs_class.topk(1, dim=2).values[..., 0]
        # 2) 3-layer FFN for bounding box regression
        enc_outputs_coord_offset = self.bbox_proposal(output_memory)
        enc_outputs_coord_unact = output_proposals + enc_outputs_coord_offset

        topk_proposals = torch.topk(enc_outputs_fg_class, self.topk_num, dim=1)[1]
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact.sigmoid()

        # # pos_embed -> linear layer -> layer norm
        pos_trans_out = self.pos_trans_norm(
            self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact))
        )
        query_pos, query = torch.split(pos_trans_out, c, dim=2)
        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        query_embeddings, bbox_predicts = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        return [
            query_embeddings,
            bbox_predicts,
            enc_outputs_class,
            enc_outputs_coord_unact.sigmoid(),
        ]


@TRANSFORMER.register_module()
class SparseDeformableDETRTransformer(DeformableDETRTransformer):
    def __init__(
        self,
        encoder=None,
        decoder=None,
        num_feature_levels=4,
        num_classes=1,
        rho=0.3,
        temperature=10000,
        num_pos_feats=128,
        topk=300,
    ):
        super(SparseDeformableDETRTransformer, self).__init__(
            encoder,
            decoder,
            num_feature_levels,
            num_classes,
            temperature,
            num_pos_feats,
            topk,
        )
        self.rho = rho
        self.enc_mask_predictor = MaskPredictor(self.embed_dims, self.embed_dims)

    def forward(self, feats, masks, pos_embeds, **kwargs):
        # prepare input for encoder
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(feats, masks, pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        # valid ratios across multi-scale features of the same image can be varied,
        # while they are interpolated and binarized on different resolutions.
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        ###########
        # prepare for sparse encoder
        backbone_output_memory, _, valid_token_nums = self.gen_encoder_proposals(
            feat_flatten + lvl_pos_embed_flatten, mask_flatten, spatial_shapes
        )
        # self.valid_token_nums = valid_token_nums
        sparse_token_nums = (valid_token_nums * self.rho).int() + 1
        backbone_topk = int(max(sparse_token_nums))
        self.sparse_token_nums = sparse_token_nums

        backbone_topk = min(backbone_topk, backbone_output_memory.shape[1])
        backbone_mask_prediction = self.enc_mask_predictor(
            backbone_output_memory
        ).squeeze(-1)
        # excluding pad area
        backbone_mask_prediction = backbone_mask_prediction.masked_fill(
            mask_flatten, backbone_mask_prediction.min()
        )
        backbone_topk_proposals = torch.topk(
            backbone_mask_prediction, backbone_topk, dim=1
        )[1]

        # encoder
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )
        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2
        )  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            topk_inds=backbone_topk_proposals,
            sparse_token_nums=sparse_token_nums,
            **kwargs,
        )

        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        # top scoring bounding boxes are picked as the final region proposals.
        # these proposals are fed into the decoder as initial boxes for the iterative bounding box refinement.

        # finalize the first stage output
        # project & normalize the memory and make proposal bounding boxes on them
        output_memory, output_proposals, _ = self.gen_encoder_proposals(
            memory, mask_flatten, spatial_shapes
        )

        # hack implementation for two-stage Deformable DETR (using the last layer registered in class/bbox_embed)
        # 1) a linear projection for bounding box binary classification (fore/background)
        enc_outputs_class = self.class_object_proposal(output_memory)
        enc_outputs_fg_class = enc_outputs_class[
            ..., 0
        ]  # enc_outputs_class.topk(1, dim=2).values[..., 0]
        # 2) 3-layer FFN for bounding box regression
        enc_outputs_coord_offset = self.bbox_proposal(output_memory)
        enc_outputs_coord_unact = output_proposals + enc_outputs_coord_offset

        topk_proposals = torch.topk(enc_outputs_fg_class, self.topk_num, dim=1)[1]
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact.sigmoid()

        # # pos_embed -> linear layer -> layer norm
        pos_trans_out = self.pos_trans_norm(
            self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact))
        )
        query_pos, query = torch.split(pos_trans_out, c, dim=2)
        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        query_embeddings, bbox_predicts = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        return [
            query_embeddings,
            bbox_predicts,
            enc_outputs_class,
            enc_outputs_coord_unact.sigmoid(),
        ]


@TRANSFORMER.register_module()
class SalientDETRTransformer(DeformableDETRTransformer):
    def __init__(
        self,
        encoder=None,
        decoder=None,
        num_feature_levels=4,
        num_classes=1,
        rho=0.3,
        temperature=10000,
        num_pos_feats=128,
        topk=300,
    ):
        super(DeformableDETRTransformer, self).__init__()
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.temperature = temperature
        self.num_pos_feats = num_pos_feats
        self.embed_dims = self.encoder.embed_dims
        self.topk_num = topk

        self.level_embed = nn.Parameter(
            torch.Tensor(num_feature_levels, self.embed_dims)
        )

        self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
        self.enc_output_norm = nn.LayerNorm(self.embed_dims)
        self.pos_trans = nn.Linear(self.embed_dims * 2, self.embed_dims)
        self.pos_trans_norm = nn.LayerNorm(self.embed_dims)

        self.class_object_proposal = nn.Linear(
            self.embed_dims, num_classes
        )  # for select obj
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_object_proposal.bias.data = torch.ones(num_classes) * bias_value
        self.bbox_proposal = MLP(
            self.embed_dims, self.embed_dims, output_dim=4, num_layers=3
        )
        nn.init.constant_(self.bbox_proposal.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_proposal.layers[-1].bias.data, 0)

        self.rho = rho
        self.enc_mask_predictor = MaskPredictor(self.embed_dims, self.embed_dims)

    def _get_gt_pair_reference_points_single(
        self,
        gt_bboxes,
        img_meta,
        mask_flatten,
        spatial_shapes,
        topk_proposals,
        reference_points,
    ):
        img_h, img_w, _ = img_meta["img_shape"]

        center_x = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / img_w
        center_y = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / img_h
        bbox_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) / img_w
        bbox_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]) / img_h
        bbox_area = bbox_w * bbox_h

        delta_pos_center = [
            (torch.rand_like(bbox_w) - 0.5) * bbox_w * 0.2,
            (torch.rand_like(bbox_h) - 0.5) * bbox_h * 0.2,
        ]
        delta_pos_bbox = [
            0.4 * (torch.rand_like(bbox_w) - 0.5) * bbox_w,
            0.4 * (torch.rand_like(bbox_h) - 0.5) * bbox_h,
        ]
        pos_center = torch.stack(
            [center_x + delta_pos_center[0], center_y + delta_pos_center[1]], dim=-1
        )  # add noise
        pos_wh = torch.stack(
            [bbox_w + delta_pos_bbox[0], bbox_h + delta_pos_bbox[1]], dim=-1
        )  # add noise

        delta_pair_center = [
            (torch.rand_like(bbox_w) - 0.5) * bbox_w * 0.4,
            (torch.rand_like(bbox_h) - 0.5) * bbox_h * 0.4,
        ]
        delta_pair_bbox = [
            0.8 * (torch.rand_like(bbox_w) - 0.5) * bbox_w,
            0.8 * (torch.rand_like(bbox_h) - 0.5) * bbox_h,
        ]
        add_center = torch.stack(
            [center_x + delta_pair_center[0], center_y + delta_pair_center[1]], dim=-1
        )
        add_wh = torch.stack(
            [bbox_w + delta_pair_bbox[0], bbox_h + delta_pair_bbox[1]], dim=-1
        )

        pair_center = torch.cat([pos_center, add_center], dim=0)
        pair_wh = torch.cat([pos_wh, add_wh], dim=0)

        pair_left_top = pair_center - pair_wh / 2
        pair_right_bottom = pair_center + pair_wh / 2
        pair_left_top = torch.clamp(pair_left_top, min=1e-3, max=0.999)
        pair_right_bottom = torch.clamp(pair_right_bottom, min=1e-3, max=0.999)

        pair_center = (pair_left_top + pair_right_bottom) / 2
        pair_wh = pair_right_bottom - pair_left_top

        _cur = 0
        gt_pair_indices = []
        gt_pair_points = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # level of encoded feature scale
            mask_flatten_single = mask_flatten[_cur : (_cur + H_ * W_)].view(H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_single[:, 0, 0])
            valid_W = torch.sum(~mask_flatten_single[0, :, 0])
            _cur += H_ * W_
            if lvl == 0:
                valid_indices = torch.nonzero(
                    bbox_area < 0.05 * (2.0 ** lvl) * 1.5, as_tuple=False
                )
            elif lvl < len(spatial_shapes) - 1:
                valid_indices = torch.nonzero(
                    (bbox_area < 0.05 * (2.0 ** lvl) * 1.5)
                    & (bbox_area > 0.05 * (2.0 ** lvl) * 0.75),
                    as_tuple=False,
                )
            else:
                valid_indices = torch.nonzero(
                    bbox_area > 0.05 * (2.0 ** lvl) * 0.75, as_tuple=False
                )

            index_center_x = (pair_center[valid_indices, 0] * valid_W + 0.5).long()
            index_center_y = (pair_center[valid_indices, 1] * valid_H + 0.5).long()
            ref_indices = index_center_y * W_ + index_center_x
            ref_points = torch.cat(
                [pair_center[valid_indices], pair_wh[valid_indices]], dim=-1
            )
            gt_pair_indices.append(ref_indices)
            gt_pair_points.append(ref_points)
        gt_pair_indices = torch.cat(gt_pair_indices, dim=0).squeeze(1)
        gt_pair_points = torch.cat(gt_pair_points, dim=0).squeeze(1)
        len_proposals = topk_proposals.shape[0]
        len_origin_keeps = len_proposals - gt_pair_indices.shape[0]
        if len_origin_keeps > 0:
            gt_added_points = torch.cat(
                [reference_points[:len_origin_keeps], gt_pair_points], dim=0
            )
            gt_added_indices = torch.cat(
                [topk_proposals[:len_origin_keeps], gt_pair_indices], dim=0
            )
        else:
            gt_added_points = gt_pair_points[:len_proposals]
            gt_added_indices = gt_pair_indices[:len_proposals]

        return gt_added_points, gt_added_indices, gt_pair_indices

    def get_gt_pair_reference_points(
        self,
        gt_bboxes,
        img_metas,
        mask_flatten,
        spatial_shapes,
        topk_proposals,
        reference_points,
    ):
        spatial_shapes_list = [spatial_shapes for _ in range(len(gt_bboxes))]
        gt_added_ref_points, gt_added_ref_inds, gt_pair_indices = multi_apply(
            self._get_gt_pair_reference_points_single,
            gt_bboxes,
            img_metas,
            mask_flatten,
            spatial_shapes_list,
            topk_proposals,
            reference_points,
        )
        gt_added_ref_points = torch.stack(gt_added_ref_points, dim=0)
        gt_added_ref_inds = torch.stack(gt_added_ref_inds, dim=0)
        return gt_added_ref_points, gt_added_ref_inds, gt_pair_indices

    def _filter_bboxes_single(self, gt_bboxes, reference_points, img_meta, dist_th=0.5):
        filtered_gt_boxes = []
        img_h, img_w, _ = img_meta["img_shape"]
        reference_points_x = reference_points[:, 0::2] * img_w
        reference_points_y = reference_points[:, 1::2] * img_h
        for gt_bbox in gt_bboxes:
            dist_x = (
                2 * reference_points_x[:, 0] - (gt_bbox[0] + gt_bbox[2])
            ) / reference_points_x[:, 1]
            dist_y = (
                2 * reference_points_y[:, 0] - (gt_bbox[1] + gt_bbox[3])
            ) / reference_points_y[:, 1]
            min_dist = torch.min(dist_x.pow(2) + dist_y.pow(2))
            if min_dist > dist_th:
                filtered_gt_boxes.append(gt_bbox)
        if len(filtered_gt_boxes) == 0:
            filtered_gt_boxes = torch.zeros(
                [0, 4], dtype=gt_bboxes.dtype, device=gt_bboxes.device
            )
        else:
            filtered_gt_boxes = torch.stack(filtered_gt_boxes, 0)
        return [filtered_gt_boxes]

    def filter_reference_points(self, gt_bboxes_list, reference_points_list, img_metas):
        filtered_gt_boxes = multi_apply(
            self._filter_bboxes_single, gt_bboxes_list, reference_points_list, img_metas
        )
        return filtered_gt_boxes[0]

    def forward(
        self, feats, masks, pos_embeds, img_metas=None, gt_bboxes=None, **kwargs
    ):
        # prepare input for encoder
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(feats, masks, pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        # valid ratios across multi-scale features of the same image can be varied,
        # while they are interpolated and binarized on different resolutions.
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # prepare for sparse encoder
        backbone_output_memory, _, valid_token_nums = self.gen_encoder_proposals(
            feat_flatten + lvl_pos_embed_flatten, mask_flatten, spatial_shapes
        )
        # self.valid_token_nums = valid_token_nums
        sparse_token_nums = (valid_token_nums * self.rho).int() + 1
        backbone_topk = int(max(sparse_token_nums))
        self.sparse_token_nums = sparse_token_nums

        backbone_topk = min(backbone_topk, backbone_output_memory.shape[1])
        backbone_mask_prediction = self.enc_mask_predictor(
            backbone_output_memory
        ).squeeze(-1)
        # excluding pad area
        backbone_mask_prediction = backbone_mask_prediction.masked_fill(
            mask_flatten, backbone_mask_prediction.min()
        )
        backbone_topk_proposals = torch.topk(
            backbone_mask_prediction, backbone_topk, dim=1
        )[1]

        # encoder
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )
        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2
        )  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            topk_inds=backbone_topk_proposals,
            sparse_token_nums=sparse_token_nums,
            **kwargs,
        )

        memory = memory.permute(1, 0, 2)

        topk_proposals = None

        # finalize the first stage output
        # project & normalize the memory and make proposal bounding boxes on them
        output_memory, output_proposals, _ = self.gen_encoder_proposals(
            memory, mask_flatten, spatial_shapes
        )

        # hack implementation for two-stage Deformable DETR (using the last layer registered in class/bbox_embed)
        # 1) a linear projection for bounding box binary classification (fore/background)
        enc_outputs_class = self.class_object_proposal(output_memory)
        enc_outputs_fg_class = enc_outputs_class[..., 0].sigmoid()
        # 2) 3-layer FFN for bounding box regression
        enc_outputs_coord_offset = self.bbox_proposal(output_memory)
        enc_outputs_coord_unact = output_proposals + enc_outputs_coord_offset

        topk_proposals = torch.topk(enc_outputs_fg_class, self.topk_num, dim=1)[1]
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact.sigmoid()

        if gt_bboxes is not None:
            filtered_bboxes = self.filter_reference_points(
                gt_bboxes, reference_points, img_metas
            )
            reference_points, topk_proposals, _ = self.get_gt_pair_reference_points(
                filtered_bboxes,
                img_metas,
                mask_flatten,
                spatial_shapes,
                topk_proposals,
                reference_points,
            )

            # # pos_embed -> linear layer -> layer norm
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(
                    self.get_proposal_pos_embed(reference_points, is_sigmoid=True)
                )
            )
            tgt = torch.gather(
                memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, memory.size(-1))
            )
        else:
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(
                    self.get_proposal_pos_embed(reference_points, is_sigmoid=True)
                )
            )
            tgt = torch.gather(
                memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, memory.size(-1))
            )

        # decoder
        query = tgt.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = pos_trans_out.permute(1, 0, 2)
        query_embeddings, bbox_predicts = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )
        return [
            query_embeddings,
            bbox_predicts,
            enc_outputs_fg_class,
            enc_outputs_coord_unact.sigmoid(),
            spatial_shapes,
            valid_ratios,
        ]
