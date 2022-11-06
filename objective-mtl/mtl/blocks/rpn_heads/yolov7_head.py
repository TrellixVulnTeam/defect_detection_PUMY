import math
import torch
import torch.nn as nn

from mtl.cores.layer_ops.yolo_layer import (
    YoloConv,
    YoloConcat,
    SPPCSPC,
    RepConv,
    IDetect,
)
from mtl.cores.ops import multiclass_nms
from mtl.cores.bbox.bbox_transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from ..block_builder import HEADS, build_loss
from .base_det_head import BaseDetHead
from .dense_test_mixins import BBoxTestMixin


class YoloV7MPResBlock(nn.Module):
    """Basic convolutions for block feature extration"""

    def __init__(self, in_channels=256, basic_channels=128, stride=2):
        super(YoloV7MPResBlock, self).__init__()
        self.mp = nn.MaxPool2d(kernel_size=stride, stride=stride)
        self.conv_mp = YoloConv(in_channels, basic_channels, k=1, s=1)

        self.conv1 = YoloConv(in_channels, basic_channels, k=1, s=1)
        self.conv2 = YoloConv(basic_channels, basic_channels, k=3, s=stride)
        self.concat = YoloConcat(dimension=1)

    def forward(self, x, x_res):
        x1 = self.mp(x)
        x1 = self.conv_mp(x1)

        x2 = self.conv1(x)
        x2 = self.conv2(x2)
        x = self.concat([x2, x1, x_res])

        return x


class YoloV7UpSampleBlock(nn.Module):
    """UpSample block"""

    def __init__(self, in_channels=256, res_channels=512, basic_channels=128, stride=2):
        super(YoloV7UpSampleBlock, self).__init__()
        self.conv_up = YoloConv(in_channels, basic_channels, k=1, s=1)
        self.up = nn.Upsample(scale_factor=stride, mode="nearest")

        self.conv_res = YoloConv(res_channels, basic_channels, k=1, s=1)
        self.concat = YoloConcat(dimension=1)

    def forward(self, x, x_res):
        x = self.conv_up(x)
        x1 = self.up(x)

        x2 = self.conv_res(x_res)

        x = self.concat([x2, x1])
        return x


class YoloV7BottleBlock(nn.Module):
    """Basic convolutions for block feature extration"""

    def __init__(
        self,
        in_channels=128,
        basic_channels=64,
        out_channels=256,
        num_convs=6,
        concat_list=[-1, -2, -3, -4, -5, -6],
        use_basic=False,
    ):
        super(YoloV7BottleBlock, self).__init__()
        layers = []
        if use_basic:
            bottle_channels = basic_channels
        else:
            bottle_channels = basic_channels // 2
        for i in range(num_convs):
            if i < 2:
                layers.append(YoloConv(in_channels, basic_channels, k=1, s=1))
            elif i == 2:
                layers.append(YoloConv(basic_channels, bottle_channels, k=3, s=1))
            else:
                layers.append(YoloConv(bottle_channels, bottle_channels, k=3, s=1))
        self.convs = nn.ModuleList(layers)

        self.concat = YoloConcat(dimension=1)
        concat_channels = 0
        for i in concat_list:
            if i + num_convs < 2:
                concat_channels += basic_channels
            else:
                concat_channels += bottle_channels

        self.conv_out = YoloConv(concat_channels, out_channels, k=1, s=1)
        self.num_convs = num_convs
        self.concat_list = concat_list

    def forward(self, x):
        mid_list = []
        for i, conv in enumerate(self.convs):
            if i - self.num_convs in self.concat_list:
                mid_list.append(conv(x))
            if i > 0:
                x = conv(x)
        x = self.concat(mid_list[::-1])
        x = self.conv_out(x)

        return x


@HEADS.register_module()
class YOLOV7Head(BaseDetHead, BBoxTestMixin):
    """Head for yolov7"""

    def __init__(
        self,
        num_classes,
        in_channels=[512, 1024, 1024],
        strides=[8, 16, 32],
        spp_basic_channels=128,
        bottle_channels=128,
        num_convs=6,
        concat_list=[-1, -2, -3, -4, -5, -6],
        use_basic=False,
        loss_det=dict(type="YoloV7OTALoss"),
        train_cfg=None,
        test_cfg=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        anchors = train_cfg["anchors"]
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.no = num_classes + 5  # number of outputs per anchor
        self.strides = strides
        self.spp = SPPCSPC(c1=in_channels[-1], c2=spp_basic_channels * 4)
        self.upsample1 = YoloV7UpSampleBlock(
            in_channels=spp_basic_channels * 4,
            res_channels=in_channels[-2],
            basic_channels=spp_basic_channels * 2,
        )
        self.upsample_block1 = YoloV7BottleBlock(
            in_channels=spp_basic_channels * 4,
            basic_channels=bottle_channels * 2,
            out_channels=spp_basic_channels * 2,
            num_convs=num_convs,
            concat_list=concat_list,
            use_basic=use_basic,
        )  # 63

        self.upsample2 = YoloV7UpSampleBlock(
            in_channels=spp_basic_channels * 2,
            res_channels=in_channels[-3],
            basic_channels=spp_basic_channels,
        )
        self.upsample_block2 = YoloV7BottleBlock(
            in_channels=spp_basic_channels * 2,
            basic_channels=bottle_channels,
            out_channels=spp_basic_channels,
            num_convs=num_convs,
            concat_list=concat_list,
            use_basic=use_basic,
        )  # 75

        self.mp1 = YoloV7MPResBlock(
            in_channels=spp_basic_channels, basic_channels=spp_basic_channels
        )
        self.mp_block1 = YoloV7BottleBlock(
            in_channels=spp_basic_channels * 4,
            basic_channels=bottle_channels * 2,
            out_channels=spp_basic_channels * 2,
            num_convs=num_convs,
            concat_list=concat_list,
            use_basic=use_basic,
        )  # 88

        self.mp2 = YoloV7MPResBlock(
            in_channels=spp_basic_channels * 2, basic_channels=spp_basic_channels * 2
        )
        self.mp_block2 = YoloV7BottleBlock(
            in_channels=spp_basic_channels * 8,
            basic_channels=bottle_channels * 4,
            out_channels=spp_basic_channels * 4,
            num_convs=num_convs,
            concat_list=concat_list,
            use_basic=use_basic,
        )  # 101

        self.repconv1 = RepConv(spp_basic_channels, spp_basic_channels * 2)
        self.repconv2 = RepConv(spp_basic_channels * 2, spp_basic_channels * 4)
        self.repconv3 = RepConv(spp_basic_channels * 4, spp_basic_channels * 8)

        self.conv_out = IDetect(
            nc=num_classes,
            na=self.na,
            ch=[spp_basic_channels * 2, spp_basic_channels * 4, spp_basic_channels * 8],
        )
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer(
            "anchor_grid", a.view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)
        loss_det.update(
            {
                "num_classes": self.num_classes,
                "strides": self.strides,
                "anchors": anchors,
            }
        )

        self.loss_det = build_loss(loss_det)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
            elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                m.inplace = True

        for mi, s in zip(self.conv_out.m, self.strides):  # from
            b = mi.bias.view(self.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.99))
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x_in):
        # spp
        s1 = x = self.spp(x_in[-1])

        # up sample
        x = self.upsample1(x, x_in[-2])
        u1 = x = self.upsample_block1(x)
        x = self.upsample2(x, x_in[-3])
        u2 = x = self.upsample_block2(x)

        # down sample
        x = self.mp1(x, u1)
        d1 = x = self.mp_block1(x)
        x = self.mp2(x, s1)
        d2 = x = self.mp_block2(x)

        r1 = self.repconv1(u2)
        r2 = self.repconv2(d1)
        r3 = self.repconv3(d2)

        return (self.conv_out([r1, r2, r3]),)

    def get_losses(
        self, pred_maps, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None
    ):
        """Compute losses of the head."""
        num_imgs = len(img_metas)
        targets = []
        for i in range(num_imgs):
            img_shape = img_metas[i]["batch_input_shape"]
            len_bbox = gt_bboxes[i].shape[0]
            single_targets = torch.zeros((len_bbox, 6), device=gt_bboxes[i].device)
            single_targets[:, 0] = i
            single_targets[:, 1] = gt_labels[i].to(dtype=gt_bboxes[i].dtype)
            single_targets[:, 2:] = bbox_xyxy_to_cxcywh(gt_bboxes[i])
            single_targets[:, [3, 5]] /= img_shape[0]
            single_targets[:, [2, 4]] /= img_shape[1]
            targets.append(single_targets)
        targets = torch.cat(targets, 0)

        losses = self.loss_det(pred_maps, targets, img_metas)

        return losses

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def get_bboxes(self, x, img_metas, rescale=False, with_nms=True):
        """Transform network output for a batch into bbox predictions."""
        z = []
        for i in range(self.nl):
            bs, _, ny, nx, _ = x[i].shape  # bs,3,20,20,85
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.strides[
                i
            ]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, self.no))

        out = torch.cat(z, 1)

        bbox_list = []
        for _idx in range(len(img_metas)):
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            bboxes = out[_idx, :, :4]
            scores = out[_idx, :, 5:]
            confidences = out[_idx, :, 4]
            conf_sel = confidences > self.test_cfg["conf_thr"]
            bboxes = bboxes[conf_sel]
            scores = scores[conf_sel]
            confidences = confidences[conf_sel]
            if len(bboxes) > self.test_cfg["nms_pre"]:
                _, topk_inds = confidences.topk(self.test_cfg["nms_pre"])
                bboxes = bboxes[topk_inds]
                scores = scores[topk_inds]
                confidences = confidences[topk_inds]

            # Compute conf
            scores *= confidences[:, None]  # conf = obj_conf * cls_conf
            bboxes = bbox_cxcywh_to_xyxy(bboxes)

            # Run NMS
            # Rescale bboxes
            if rescale:
                scale_factor = img_metas[_idx]["scale_factor"]
                bboxes /= bboxes.new_tensor(scale_factor)
            if with_nms:
                det_bboxes, det_labels = multiclass_nms(
                    bboxes,
                    scores,
                    self.test_cfg["score_thr"],
                    self.test_cfg["nms"],
                    self.test_cfg["max_per_img"],
                    score_factors=None,
                )
            else:
                det_bboxes = bboxes
                det_labels = scores
            bbox_list.append((det_bboxes, det_labels))
        return bbox_list

    def get_target(self, **kwargs):
        pass
