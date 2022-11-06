import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mtl.utils.checkpoint_util import load_checkpoint
from mtl.utils.log_util import get_root_logger
from mtl.cores.ops import build_activation_layer, build_norm_layer
from ..block_builder import BACKBONES


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_cfg=dict(type="GELU"),
        drop=0.0,
    ):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    b, h, w, c = x.shape
    nh, nw = int(h / window_size), int(w / window_size)
    x = x.view(b, nh, window_size, nw, window_size, c)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(b * nh * nw, window_size * window_size, c)
    )
    return windows


def window_reverse(windows, window_size, b, h, w, c):
    nh, nw = int(h / window_size), int(w / window_size)
    x = (
        windows.view(b, nh, nw, window_size, window_size, c)
        .permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(b, h, w, c)
    )
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative
    position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output.
            Default: 0.0
    """

    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0
    ):
        super(WindowAttention, self).__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))

        self.rpb = Mlp(2, 512, self.num_heads, act_cfg=dict(type="ReLU"))

        self.register_buffer(
            "relative_coords", self.get_relative_coords(self.window_size)
        )
        self.register_buffer(
            "relative_index", self.get_relative_index(self.window_size)
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def get_relative_coords(self, window_size):
        relative_coords_h = torch.arange(-window_size[0] + 1, window_size[0]).float()
        relative_coords_w = torch.arange(-window_size[1] + 1, window_size[1]).float()
        relative_coords = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij"),
            dim=-1,
        ).view(-1, 2)
        relative_coords = torch.sign(relative_coords) * torch.log(
            1 + relative_coords.abs()
        )

        return relative_coords.contiguous()

    def get_relative_index(self, window_size):
        coord_h = torch.arange(window_size[0])
        coord_w = torch.arange(window_size[1])
        coords = torch.stack(
            torch.meshgrid([coord_h, coord_w], indexing="ij"), dim=-1
        ).view(-1, 2)

        relative_coords = coords[:, None, :] - coords[None, :, :]
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1

        relative_index = relative_coords.sum(dim=-1).contiguous()

        return relative_index

    def forward(self, x, mask=None):
        b, l, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, l, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * torch.exp(-self.scale)

        relative_position_bias = self.rpb(self.relative_coords)
        relative_position_bias = (
            relative_position_bias[self.relative_index.view(-1)]
            .view(l, l, self.num_heads)
            .permute(2, 0, 1)
            .contiguous()
        )

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(
                int(b / num_windows), num_windows, self.num_heads, l, l
            ) + mask.unsqueeze(1)
            attn = attn.view(b, self.num_heads, l, l)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, l, c)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        str = f"dim={self.dim}"
        str += f", window_size={self.window_size}"
        str += f", num_heads={self.num_heads}"
        return str


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk
            scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate.
            Default: 0.0
        attn_drop (float, optional): Attention dropout rate.
            Default: 0.0
        drop_path (float, optional): Stochastic depth rate.
            Default: 0.0
        act_layer (nn.Module, optional): Activation layer.
            Default: nn.GELU
        norm_cfg (dict, optional): Normalization layer cfg.
            Default: dict(type='LN')
        downsample (nn.Module | None, optional): Downsample layer at the end
            of the block. Default: None
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN"),
        post_norm=False,
        extra_norm=False,
    ):
        super(SwinTransformerBlock, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.post_norm = post_norm
        self.extra_norm = extra_norm

        self.ape_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_cfg=act_cfg, drop=drop
        )

        if self.post_norm:
            self.norm3 = build_norm_layer(norm_cfg, dim)[1]
            self.norm4 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm3 = nn.Identity()
            self.norm4 = nn.Identity()

        if self.extra_norm:
            self.norm5 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm5 = nn.Identity()

    def gen_attn_mask(self, x):
        _, h, w, _ = x.shape

        if min(h, w) <= self.window_size:
            shift_size = 0
        else:
            shift_size = self.shift_size

        if shift_size > 0:
            img_mask = x.new_zeros((1, h, w, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -shift_size),
                slice(-shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -shift_size),
                slice(-shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        return shift_size, attn_mask

    def pad(self, x):
        _, h, w, _ = x.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        return x

    def forward(self, x):
        x = x + self.ape_conv(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)

        _, h, w, c = x.shape
        shortcut = x

        x = self.norm1(x)
        x = self.pad(x)
        shift_size, attn_mask = self.gen_attn_mask(x)
        # cyclic shift
        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, *x.shape)

        # reverse cyclic shift
        if shift_size > 0:
            x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x[:, :h, :w, :]

        x = shortcut + self.drop_path(self.norm3(x))
        x = x + self.drop_path(self.norm4(self.mlp(self.norm2(x))))

        x = self.norm5(x)

        return x

    def extra_repr(self) -> str:
        str = f"dim={self.dim}"
        str += f", num_heads={self.num_heads}"
        str += f", window_size={self.window_size}"
        str += f", shift_size={self.shift_size}"
        str += f", mlp_ratio={self.mlp_ratio}"
        return str


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels.
            Default: 3.
        embed_dim (int): Number of linear projection output channels.
            Default: 96.
        norm_cfg (dict, optional): Normalization layer cfg.
            Default: None
    """

    def __init__(
        self, patch_size=7, patch_stride=4, in_chans=3, embed_dim=96, norm_cfg=None
    ):
        super(PatchEmbed, self).__init__()

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_stride,
            padding=(patch_size - 1) // 2,
        )
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = None

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)

        if self.norm is not None:
            x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate.
            Default: 0.0
        attn_drop (float, optional): Attention dropout rate.
            Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate.
            Default: 0.0
        norm_cfg (dict, optional): Normalization layer cfg.
            Default: dict(type='LN')
        downsample (nn.Module | None, optional): Downsample layer at the end
            of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory.
            Default: False.
    """

    def __init__(
        self,
        patch_size,
        patch_stride,
        dim,
        dim_out,
        depth,
        num_heads,
        window_size,
        patch_norm=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_cfg=dict(type="LN"),
        post_norm=False,
        extra_norm=False,
        sum_depth=None,
        use_checkpoint=False,
    ):
        super(BasicLayer, self).__init__()

        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.patch_embed = PatchEmbed(
            in_chans=dim,
            embed_dim=dim_out,
            patch_size=patch_size,
            patch_stride=patch_stride,
            norm_cfg=norm_cfg if patch_norm else None,
        )

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim_out,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_cfg=norm_cfg,
                    post_norm=post_norm,
                    extra_norm=(i + 1 + sum_depth) % 6 == 0 if extra_norm else False,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, mask=None, mask_token=None):
        x = self.patch_embed(x)

        if mask is not None and mask_token is not None:
            _, h, w, _ = x.shape
            mask = mask.unsqueeze(1)
            mask = F.interpolate(mask, (h, w), mode="nearest").permute(0, 2, 3, 1)

            x = (1 - mask) * x + mask * mask_token

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

    def extra_repr(self) -> str:
        str = f"dim={self.dim}"
        str += f", depth={self.depth}"
        return str


@BACKBONES.register_module()
class SwinTransformerV2(nn.Module):
    r"""Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030 # noqa

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_sizes (list): Patch sizes.
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_cfg (dict): Normalization layer. Default: dict(type='LN').
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        patch_sizes=[7, 3, 3, 3],
        patch_strides=[4, 2, 2, 2],
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        post_norm=False,
        extra_norm=False,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_cfg=dict(type="LN"),
        patch_norm=True,
        use_checkpoint=False,
        frozen_stages=-1,
        norm_eval=False,
        out_indices=(3,),
    ):
        super(SwinTransformerV2, self).__init__()

        self.patch_sizes = patch_sizes
        self.patch_strides = patch_strides
        self.window_size = window_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.out_indices = out_indices
        self.use_checkpoint = use_checkpoint
        self.in_chans = in_chans

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.stages = nn.ModuleList()
        num_features = [in_chans] + [
            int(embed_dim * 2 ** i) for i in range(self.num_layers)
        ]
        for i in range(self.num_layers):
            layer = BasicLayer(
                patch_size=patch_sizes[i],
                patch_stride=patch_strides[i],
                dim=num_features[i],
                dim_out=num_features[i + 1],
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_cfg=norm_cfg,
                post_norm=post_norm,
                extra_norm=extra_norm,
                sum_depth=sum(depths[:i]),
                use_checkpoint=use_checkpoint,
            )
            self.stages.append(layer)

            if i in self.out_indices:
                if norm_cfg is not None:
                    norm_layer = build_norm_layer(norm_cfg, num_features[i + 1])[1]
                else:
                    norm_layer = nn.Identity()

                self.add_module(f"norm{i}", norm_layer)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone."""

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            logger = get_root_logger()
            if torch.cuda.is_available():
                load_checkpoint(self, pretrained, strict=False, logger=logger)
            else:
                load_checkpoint(
                    self, pretrained, map_location="cpu", strict=False, logger=logger
                )
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x, mask=None, mask_token=None):
        x = x.permute(0, 2, 3, 1)

        outs = []
        for i, stage in enumerate(self.stages):
            if i == 0:
                x = stage(x, mask, mask_token)
            else:
                x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                out = norm_layer(x)
                out = out.permute(0, 3, 1, 2).contiguous()

                outs.append(out)

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            m = self.stages[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        for i in self.out_indices:
            if i <= self.frozen_stages:
                for param in getattr(self, f"norm{i}").parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(SwinTransformerV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()


@BACKBONES.register_module()
class SwinTransformerV2ForSimMIM(SwinTransformerV2):
    def __init__(self, **kwargs):
        super(SwinTransformerV2ForSimMIM, self).__init__(**kwargs)

        self.mask_token = nn.Parameter(torch.zeros(self.embed_dim))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        return super(SwinTransformerV2ForSimMIM, self).forward(x, mask, self.mask_token)

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {"mask_token"}
