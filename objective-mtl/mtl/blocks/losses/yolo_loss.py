import torch
import torch.nn as nn
import torch.nn.functional as F

from mtl.utils.bbox_util import bbox_iou, box_iou, xywh2xyxy
from ..block_builder import LOSSES


def smooth_BCE(eps=0.1):
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class SigmoidBin(nn.Module):
    """Activate method"""

    def __init__(
        self,
        bin_count=10,
        min=0.0,
        max=1.0,
        reg_scale=2.0,
        use_loss_regression=True,
        use_fw_regression=True,
        BCE_weight=1.0,
        smooth_eps=0.0,
    ):
        super(SigmoidBin, self).__init__()

        self.bin_count = bin_count
        self.length = bin_count + 1
        self.min = min
        self.max = max
        self.scale = float(max - min)
        self.shift = self.scale / 2.0

        self.use_loss_regression = use_loss_regression
        self.use_fw_regression = use_fw_regression
        self.reg_scale = reg_scale
        self.BCE_weight = BCE_weight

        start = min + (self.scale / 2.0) / self.bin_count
        end = max - (self.scale / 2.0) / self.bin_count
        step = self.scale / self.bin_count
        self.step = step

        bins = torch.range(start, end + 0.0001, step).float()
        self.register_buffer("bins", bins)

        self.cp = 1.0 - 0.5 * smooth_eps
        self.cn = 0.5 * smooth_eps

        self.BCEbins = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([BCE_weight]))
        self.MSELoss = nn.MSELoss()

    def get_length(self):
        return self.length

    def forward(self, pred):
        assert pred.shape[-1] == self.length, (
            "pred.shape[-1]=%d is not equal to self.length=%d"
            % (pred.shape[-1], self.length)
        )

        pred_reg = (pred[..., 0] * self.reg_scale - self.reg_scale / 2.0) * self.step
        pred_bin = pred[..., 1 : (1 + self.bin_count)]

        _, bin_idx = torch.max(pred_bin, dim=-1)
        bin_bias = self.bins[bin_idx]

        if self.use_fw_regression:
            result = pred_reg + bin_bias
        else:
            result = bin_bias
        result = result.clamp(min=self.min, max=self.max)

        return result

    def training_loss(self, pred, target):
        assert pred.shape[-1] == self.length, (
            "pred.shape[-1]=%d is not equal to self.length=%d"
            % (pred.shape[-1], self.length)
        )
        assert pred.shape[0] == target.shape[0], (
            "pred.shape=%d is not equal to the target.shape=%d"
            % (pred.shape[0], target.shape[0])
        )
        device = pred.device

        pred_reg = (
            pred[..., 0].sigmoid() * self.reg_scale - self.reg_scale / 2.0
        ) * self.step
        pred_bin = pred[..., 1 : (1 + self.bin_count)]

        diff_bin_target = torch.abs(target[..., None] - self.bins)
        _, bin_idx = torch.min(diff_bin_target, dim=-1)

        bin_bias = self.bins[bin_idx]
        bin_bias.requires_grad = False
        result = pred_reg + bin_bias

        target_bins = torch.full_like(pred_bin, self.cn, device=device)  # targets
        n = pred.shape[0]
        target_bins[range(n), bin_idx] = self.cp

        loss_bin = self.BCEbins(pred_bin, target_bins)  # BCE

        if self.use_loss_regression:
            loss_regression = self.MSELoss(result, target)  # MSE
            loss = loss_bin + loss_regression
        else:
            loss = loss_bin

        out_result = result.clamp(min=self.min, max=self.max)

        return loss, out_result


class YoloFocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(YoloFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


@LOSSES.register_module()
class YoloLoss(nn.Module):
    # Compute losses
    def __init__(
        self,
        num_classes,
        strides,
        anchors,
        label_smoothing=0.0,
        cls_pw=1.0,
        obj_pw=1.0,
        gamma=0.0,
        box_factor=0.05,
        cls_factor=0.3,
        obj_factor=0.7,
        autobalance=False,
        anchor_t=4.0,
    ):
        super(YoloLoss, self).__init__()

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(
            eps=label_smoothing
        )  # positive, negative BCE targets
        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cls_pw))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(obj_pw))

        # Focal loss
        if gamma > 0:
            BCEcls, BCEobj = YoloFocalLoss(BCEcls, gamma), YoloFocalLoss(BCEobj, gamma)

        self.BCEcls, self.BCEobj = BCEcls, BCEobj
        self.nc = num_classes
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        a = torch.tensor(anchors).float().view(self.nl, -1, 2) / torch.tensor(
            strides
        ).float().view(-1, 1, 1)
        self.register_buffer("anchors", a)  # shape(nl,na,2)

        self.box_factor = box_factor
        self.cls_factor = cls_factor
        self.obj_factor = obj_factor

        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            self.nl, [4.0, 1.0, 0.25, 0.06, 0.02]
        )  # P3-P7
        # self.balance = {3: [4.0, 1.0, 0.4]}.get(self.nl, [4.0, 1.0, 0.25, 0.1, .05])  # P3-P7
        # self.balance = {3: [4.0, 1.0, 0.4]}.get(self.nl, [4.0, 1.0, 0.5, 0.4, .1])  # P3-P7
        self.ssi = list(strides).index(16) if autobalance else 0  # stride 16 index
        self.gr = 1.0
        self.autobalance = autobalance
        self.anchor_t = anchor_t

    def forward(self, preds, targets, img_metas):  # predictions, targets, model
        device = preds[0].device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(preds, targets)  # targets

        # Losses
        for i, pi in enumerate(preds):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(
                    pbox.T, tbox[i], x1y1x2y2=False, CIoU=True
                )  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(
                    0
                ).type(
                    tobj.dtype
                )  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    # t[t==self.cp] = iou.detach().clamp(0).type(t.dtype)
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = (
                    self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
                )

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.box_factor * 3.0 / self.nl
        lobj *= self.obj_factor * 3.0 / self.nl  # (imgsz / 640) ** 2
        lcls *= self.cls_factor * self.nc / 80.0 * 3.0 / self.nl
        bs = tobj.shape[0]  # batch size

        return {"loss_box": lbox * bs, "loss_obj": lobj * bs, "loss_cls": lcls * bs}

    def build_targets(self, preds, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(
            7, device=targets.device
        ).long()  # normalized to gridspace gain
        ai = (
            torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        )  # same as .repeat_interleave(nt)
        targets = torch.cat(
            (targets.repeat(na, 1, 1), ai[:, :, None]), 2
        )  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=targets.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1.0 / r).max(2)[0] < self.anchor_t  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                l, m = ((gxi % 1.0 < g) & (gxi > 1.0)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append(
                (b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1))
            )  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


@LOSSES.register_module()
class YoloOTALoss(nn.Module):
    # Compute losses
    def __init__(
        self,
        num_classes,
        strides,
        anchors,
        label_smoothing=0.0,
        cls_pw=1.0,
        obj_pw=1.0,
        gamma=0.0,
        box_factor=0.05,
        cls_factor=0.3,
        obj_factor=0.7,
        autobalance=False,
        anchor_t=4.0,
    ):
        super(YoloOTALoss, self).__init__()

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(
            eps=label_smoothing
        )  # positive, negative BCE targets

        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cls_pw))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(obj_pw))
        # Focal loss
        if gamma > 0:
            BCEcls, BCEobj = YoloFocalLoss(BCEcls, gamma), YoloFocalLoss(BCEobj, gamma)

        self.BCEcls, self.BCEobj = BCEcls, BCEobj
        self.nc = num_classes
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.strides = strides
        self.anchor_t = anchor_t
        a = torch.tensor(anchors).float().view(self.nl, -1, 2) / torch.tensor(
            strides
        ).float().view(-1, 1, 1)
        self.register_buffer("anchors", a)  # shape(nl,na,2)

        self.box_factor = box_factor
        self.cls_factor = cls_factor
        self.obj_factor = obj_factor

        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            self.nl, [4.0, 1.0, 0.25, 0.06, 0.02]
        )  # P3-P7
        self.ssi = list(strides).index(16) if autobalance else 0  # stride 16 index
        self.gr = 1.0
        self.autobalance = autobalance

    def forward(self, preds, targets, img_metas):  # predictions, targets, model
        device = targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(
            preds, targets, img_metas
        )
        pre_gen_gains = [
            torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in preds
        ]

        # Losses
        for i, pi in enumerate(preds):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                # pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(
                    pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True
                )  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(
                    0
                ).type(
                    tobj.dtype
                )  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = (
                    self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
                )

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.box_factor * 3.0 / self.nl
        imgsz = (
            preds[0].shape[2] * preds[0].shape[3] * self.strides[0] * self.strides[0]
        )
        lobj *= self.obj_factor * 3.0 / self.nl * imgsz / (640 * 640)
        lcls *= self.cls_factor * self.nc / 80.0 * 3.0 / self.nl
        bs_factor = tobj.shape[0]  # batch size

        return {
            "loss_box": lbox * bs_factor,
            "loss_obj": lobj * bs_factor,
            "loss_cls": lcls * bs_factor,
        }

    def build_targets(self, preds, targets, img_metas):
        indices, anch = self.find_triple_positive(preds, targets)

        matching_bs = [[] for _ in preds]
        matching_as = [[] for _ in preds]
        matching_gjs = [[] for _ in preds]
        matching_gis = [[] for _ in preds]
        matching_targets = [[] for _ in preds]
        matching_anchs = [[] for _ in preds]

        nl = len(preds)

        for batch_idx in range(preds[0].shape[0]):
            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue

            tx = this_target[:, 2] * img_metas[batch_idx]["batch_input_shape"][1]
            ty = this_target[:, 3] * img_metas[batch_idx]["batch_input_shape"][0]
            tw = this_target[:, 4] * img_metas[batch_idx]["batch_input_shape"][1]
            th = this_target[:, 5] * img_metas[batch_idx]["batch_input_shape"][0]
            txywh = torch.stack([tx, ty, tw, th], dim=1)
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []

            for i, pi in enumerate(preds):
                b, a, gj, gi = indices[i]
                idx = b == batch_idx
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)

                fg_pred = pi[b, a, gj, gi]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2.0 - 0.5 + grid) * self.strides[
                    i
                ]  # / 8.
                # pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.strides[i]
                pwh = (
                    (fg_pred[:, 2:4].sigmoid() * 2) ** 2
                    * anch[i][idx]
                    * self.strides[i]
                )  # / 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)

            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                torch.log(y / (1 - y)), gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_

            cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss

            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device="cuda:0", dtype=torch.int64)
                matching_as[i] = torch.tensor([], device="cuda:0", dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device="cuda:0", dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device="cuda:0", dtype=torch.int64)
                matching_targets[i] = torch.tensor(
                    [], device="cuda:0", dtype=torch.int64
                )
                matching_anchs[i] = torch.tensor([], device="cuda:0", dtype=torch.int64)

        return (
            matching_bs,
            matching_as,
            matching_gjs,
            matching_gis,
            matching_targets,
            matching_anchs,
        )

    def find_triple_positive(self, preds, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(
            7, device=targets.device
        ).long()  # normalized to gridspace gain
        ai = (
            torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        )  # same as .repeat_interleave(nt)
        targets = torch.cat(
            (targets.repeat(na, 1, 1), ai[:, :, None]), 2
        )  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=targets.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1.0 / r).max(2)[0] < self.anchor_t  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                l, m = ((gxi % 1.0 < g) & (gxi > 1.0)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append(
                (b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1))
            )  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch


@LOSSES.register_module()
class YoloBinOTALoss(nn.Module):
    # Compute losses
    def __init__(
        self,
        num_classes,
        strides,
        anchors,
        label_smoothing=0.0,
        cls_pw=1.0,
        obj_pw=1.0,
        gamma=0.0,
        box_factor=0.05,
        cls_factor=0.3,
        obj_factor=0.7,
        autobalance=False,
        bin_count=10,
        anchor_t=4.0,
    ):
        super(YoloBinOTALoss, self).__init__()

        self.cp, self.cn = smooth_BCE(
            eps=label_smoothing
        )  # positive, negative BCE targets

        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cls_pw))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(obj_pw))
        # Focal loss
        if gamma > 0:
            BCEcls, BCEobj = YoloFocalLoss(BCEcls, gamma), YoloFocalLoss(BCEobj, gamma)

        self.BCEcls, self.BCEobj = BCEcls, BCEobj
        self.nc = num_classes
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        a = torch.tensor(anchors).float().view(self.nl, -1, 2) / torch.tensor(
            strides
        ).float().view(-1, 1, 1)
        self.register_buffer("anchors", a)  # shape(nl,na,2)

        self.box_factor = box_factor
        self.cls_factor = cls_factor
        self.obj_factor = obj_factor

        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            self.nl, [4.0, 1.0, 0.25, 0.06, 0.02]
        )  # P3-P7
        self.ssi = list(strides).index(16) if autobalance else 0  # stride 16 index
        self.gr = 1.0
        self.autobalance = autobalance
        self.strides = strides
        self.bin_count = bin_count

        self.wh_bin_sigmoid = SigmoidBin(
            bin_count=self.bin_count, min=0.0, max=4.0, use_loss_regression=False
        )
        self.anchor_t = anchor_t

    def forward(self, preds, targets, img_metas):  # predictions, targets, model
        device = targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(
            preds, targets, img_metas
        )
        pre_gen_gains = [
            torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in preds
        ]

        # Losses
        for i, pi in enumerate(preds):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            obj_idx = (
                self.wh_bin_sigmoid.get_length() * 2 + 2
            )  # x,y, w-bce, h-bce     # xy_bin_sigmoid.get_length()*2

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid

                w_loss, pw = self.wh_bin_sigmoid.training_loss(
                    ps[..., 2 : (3 + self.bin_count)],
                    selected_tbox[..., 2] / anchors[i][..., 0],
                )
                h_loss, ph = self.wh_bin_sigmoid.training_loss(
                    ps[..., (3 + self.bin_count) : obj_idx],
                    selected_tbox[..., 3] / anchors[i][..., 1],
                )

                pw *= anchors[i][..., 0]
                ph *= anchors[i][..., 1]

                px = ps[:, 0].sigmoid() * 2.0 - 0.5
                py = ps[:, 1].sigmoid() * 2.0 - 0.5

                lbox += w_loss + h_loss  # + x_loss + y_loss

                pbox = torch.cat(
                    (
                        px.unsqueeze(1),
                        py.unsqueeze(1),
                        pw.unsqueeze(1),
                        ph.unsqueeze(1),
                    ),
                    1,
                ).to(
                    device
                )  # predicted box

                iou = bbox_iou(
                    pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True
                )  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(
                    0
                ).type(
                    tobj.dtype
                )  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(
                        ps[:, (1 + obj_idx) :], self.cn, device=device
                    )  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, (1 + obj_idx) :], t)  # BCE

            obji = self.BCEobj(pi[..., obj_idx], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = (
                    self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
                )

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.box_factor * 3.0 / self.nl
        lobj *= self.obj_factor * 3.0 / self.nl  # (imgsz / 640) ** 2
        lcls *= self.cls_factor * self.nc / 80.0 * 3.0 / self.nl
        bs = tobj.shape[0]  # batch size

        return {"loss_box": lbox * bs, "loss_obj": lobj * bs, "loss_cls": lcls * bs}

    def build_targets(self, preds, targets, img_metas):
        indices, anch = self.find_triple_positive(preds, targets)

        matching_bs = [[] for _ in preds]
        matching_as = [[] for _ in preds]
        matching_gjs = [[] for _ in preds]
        matching_gis = [[] for _ in preds]
        matching_targets = [[] for _ in preds]
        matching_anchs = [[] for _ in preds]

        nl = len(preds)

        for batch_idx in range(preds[0].shape[0]):
            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue

            tx = this_target[:, 2] * img_metas[batch_idx]["batch_input_shape"][1]
            ty = this_target[:, 3] * img_metas[batch_idx]["batch_input_shape"][0]
            tw = this_target[:, 4] * img_metas[batch_idx]["batch_input_shape"][1]
            th = this_target[:, 5] * img_metas[batch_idx]["batch_input_shape"][0]
            txywh = torch.stack([tx, ty, tw, th], dim=1)
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []

            for i, pi in enumerate(preds):
                obj_idx = self.wh_bin_sigmoid.get_length() * 2 + 2

                b, a, gj, gi = indices[i]
                idx = b == batch_idx
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)

                fg_pred = pi[b, a, gj, gi]
                p_obj.append(fg_pred[:, obj_idx : (obj_idx + 1)])
                p_cls.append(fg_pred[:, (obj_idx + 1) :])

                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2.0 - 0.5 + grid) * self.strides[
                    i
                ]  # / 8.
                # pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.strides[i] #/ 8.
                pw = (
                    self.wh_bin_sigmoid.forward(
                        fg_pred[..., 2 : (3 + self.bin_count)].sigmoid()
                    )
                    * anch[i][idx][:, 0]
                    * self.strides[i]
                )
                ph = (
                    self.wh_bin_sigmoid.forward(
                        fg_pred[..., (3 + self.bin_count) : obj_idx].sigmoid()
                    )
                    * anch[i][idx][:, 1]
                    * self.strides[i]
                )

                pxywh = torch.cat([pxy, pw.unsqueeze(1), ph.unsqueeze(1)], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)

            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                torch.log(y / (1 - y)), gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_

            cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss

            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            matching_bs[i] = torch.cat(matching_bs[i], dim=0)
            matching_as[i] = torch.cat(matching_as[i], dim=0)
            matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
            matching_gis[i] = torch.cat(matching_gis[i], dim=0)
            matching_targets[i] = torch.cat(matching_targets[i], dim=0)
            matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)

        return (
            matching_bs,
            matching_as,
            matching_gjs,
            matching_gis,
            matching_targets,
            matching_anchs,
        )

    def find_triple_positive(self, preds, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = (
            torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        )  # same as .repeat_interleave(nt)
        targets = torch.cat(
            (targets.repeat(na, 1, 1), ai[:, :, None]), 2
        )  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=targets.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1.0 / r).max(2)[0] < self.anchor_t  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                l, m = ((gxi % 1.0 < g) & (gxi > 1.0)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append(
                (b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1))
            )  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch
