import torch

from mtl.cores.core_bbox import build_assigner
from mtl.cores.bbox.bbox_transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mtl.utils.misc_util import multi_apply
from mtl.utils.parallel_util import reduce_mean
from ..block_builder import HEADS, build_loss
from .deformabledetr_head import DeformableDETRHead


@HEADS.register_module()
class SalientDETRHead(DeformableDETRHead):
    def __init__(
        self,
        num_classes,
        embed_dims,
        num_pred,
        num_query=300,
        sync_cls_avg_factor=False,
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
        loss_enc_cls=dict(type="MSELoss", loss_weight=1.0),
        train_cfg=dict(
            enc_assigner=dict(
                type="HungarianAssigner",
                cls_cost=dict(type="EntrophyCost", weight=10.0),
                reg_cost=dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
            ),
            assigner=dict(
                type="HungarianAssigner",
                cls_cost=dict(type="ClassificationCost", weight=1.0),
                reg_cost=dict(type="BBoxL1Cost", weight=5.0),
                iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
            ),
        ),
        test_cfg=dict(max_per_img=300),
        **kwargs
    ):
        super(SalientDETRHead, self).__init__(
            num_classes,
            embed_dims,
            num_pred,
            num_query=num_query,
            sync_cls_avg_factor=sync_cls_avg_factor,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_iou=loss_iou,
            loss_enc_cls=loss_enc_cls,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs
        )
        self.loss_enc_cls = build_loss(loss_enc_cls)
        enc_assigner = train_cfg["enc_assigner"]
        self.enc_assigner = build_assigner(enc_assigner)

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        img_metas,
        gt_bboxes_ignore_list=None,
        is_enc=False,
    ):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert (
            gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_preds = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_preds)]
        is_enc_list = [is_enc for _ in range(num_preds)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            img_metas,
            gt_bboxes_ignore_list,
            is_enc_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def _get_target_single(
        self,
        cls_score,
        bbox_pred,
        gt_bboxes,
        gt_labels,
        img_meta,
        gt_bboxes_ignore=None,
        is_enc=False,
    ):
        """ "Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)

        # label targets
        if is_enc:
            # print(cls_score.shape, gt_labels.shape)
            assign_result = self.enc_assigner.assign(
                bbox_pred, cls_score, gt_bboxes, gt_labels, img_meta, gt_bboxes_ignore
            )
            sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds
            labels = gt_labels
            label_weights = None
        else:
            assign_result = self.assigner.assign(
                bbox_pred, cls_score, gt_bboxes, gt_labels, img_meta, gt_bboxes_ignore
            )
            sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds
            labels = gt_bboxes.new_full(
                (num_bboxes,), self.num_classes, dtype=torch.long
            )
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred).float()
        bbox_weights = torch.zeros_like(bbox_pred).float()
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta["img_shape"]

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def _get_loss_single(
        self,
        cls_scores,
        bbox_preds,
        gt_bboxes_list,
        gt_labels_list,
        img_metas,
        gt_bboxes_ignore_list=None,
        is_enc=False,
    ):
        """ "Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            img_metas,
            gt_bboxes_ignore_list,
            is_enc,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # Compute the average number of gt boxes across all gpus
        num_total_pos = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float32).cuda()
        ).item()
        num_total_pos = max(num_total_pos, 1.0)

        # classification loss
        if is_enc:
            cls_scores = cls_scores.reshape(-1)
            loss_cls = self.loss_enc_cls(cls_scores, labels)
        else:
            label_weights = torch.cat(label_weights_list, 0)
            cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=num_total_pos
            )

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta["img_shape"]
            factor = (
                bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
                .unsqueeze(0)
                .repeat(bbox_pred.size(0), 1)
            )
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos
        )

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos
        )
        return loss_cls, loss_bbox, loss_iou

    def _get_proposal_map(self, gt_bboxes, spatial_shapes, valid_ratio, img_meta):
        proposal_maps = []
        for (H_, W_) in spatial_shapes:
            proposal_maps.append(
                torch.zeros(
                    [H_, W_], dtype=valid_ratio.dtype, device=valid_ratio.device
                )
            )
        img_h, img_w, _ = img_meta["img_shape"]
        if len(gt_bboxes) > 0:
            bbox_ranges_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) / img_w
            bbox_ranges_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]) / img_h
            bbox_begins_x = gt_bboxes[:, 0] / img_w
            bbox_begins_y = gt_bboxes[:, 1] / img_h

            for i, (H_, W_) in enumerate(spatial_shapes):
                for j in range(len(gt_bboxes)):
                    bbox_w = bbox_ranges_w[j] * W_ * valid_ratio[i][0]
                    bbox_h = bbox_ranges_h[j] * H_ * valid_ratio[i][1]
                    bbox_H_ = int(bbox_h + 0.5)
                    bbox_W_ = int(bbox_w + 0.5)
                    if bbox_H_ < 1 or bbox_W_ < 1:
                        continue
                    bbox_y, bbox_x = torch.meshgrid(
                        torch.linspace(
                            0.0,
                            bbox_h,
                            bbox_H_,
                            dtype=gt_bboxes[0].dtype,
                            device=gt_bboxes[0].device,
                        ),
                        torch.linspace(
                            0.0,
                            bbox_w,
                            bbox_W_,
                            dtype=gt_bboxes[0].dtype,
                            device=gt_bboxes[0].device,
                        ),
                    )
                    point_distance = (bbox_x - bbox_w / 2).pow(2) / bbox_w.pow(2) + (
                        bbox_y - bbox_y / 2
                    ).pow(2) / bbox_h.pow(2)
                    gaussian_box = torch.exp(-8 * point_distance)
                    begin_x = int(bbox_begins_x[j] * W_ * valid_ratio[i][0] + 0.5)
                    if begin_x < 0:
                        box_begin_x = -begin_x
                        begin_x = 0
                    else:
                        box_begin_x = 0
                    begin_y = int(bbox_begins_y[j] * H_ * valid_ratio[i][1] + 0.5)
                    if begin_y < 0:
                        box_begin_y = -begin_y
                        begin_y = 0
                    else:
                        box_begin_y = 0
                    end_x = begin_x + bbox_W_
                    if end_x > W_:
                        box_end_x = bbox_W_ - (end_x - W_)
                        end_x = W_
                    else:
                        box_end_x = bbox_W_
                    end_y = begin_y + bbox_H_
                    if end_y > H_:
                        box_end_y = bbox_H_ - (end_y - H_)
                        end_y = H_
                    else:
                        box_end_y = bbox_H_
                    tmp_map = torch.stack(
                        [
                            proposal_maps[i][begin_y:end_y, begin_x:end_x],
                            gaussian_box[box_begin_y:box_end_y, box_begin_x:box_end_x],
                        ],
                        dim=2,
                    )
                    proposal_maps[i][begin_y:end_y, begin_x:end_x] = torch.max(
                        tmp_map, dim=2
                    )[0]

        flatten_maps = []
        for proposal_map in proposal_maps:
            flatten_maps.append(proposal_map.flatten(0))
        flatten_maps = torch.cat(flatten_maps, 0)
        return [flatten_maps]

    def get_proposal_maps(
        self, gt_bboxes_list, spatial_shapes, valid_ratios, img_metas
    ):
        num_img = len(gt_bboxes_list)
        spatial_shapes_list = [spatial_shapes for _ in range(num_img)]
        proposal_maps_list = multi_apply(
            self._get_proposal_map,
            gt_bboxes_list,
            spatial_shapes_list,
            valid_ratios,
            img_metas,
        )
        proposal_maps = torch.stack(proposal_maps_list[0], 0)

        return proposal_maps

    def forward_train(
        self,
        x,
        dec_bbox_predicts,
        enc_class_scores,
        enc_bbox_predicts,
        spatial_shapes,
        valid_ratios,
        img_metas,
        gt_bboxes,
        gt_labels=None,
        gt_bboxes_ignore=None,
        proposal_cfg=None,
        **kwargs
    ):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        class_scores = self(x)

        # encoder loss
        gt_enc_maps = self.get_proposal_maps(
            gt_bboxes, spatial_shapes, valid_ratios, img_metas
        )
        enc_outs = (enc_class_scores, enc_bbox_predicts)
        enc_loss_inputs = enc_outs + (gt_bboxes, gt_enc_maps, img_metas)
        enc_losses = self.get_losses(
            *enc_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, is_enc=True
        )

        # decoder loss
        dec_outs = (class_scores, dec_bbox_predicts)
        dec_loss_inputs = dec_outs + (gt_bboxes, gt_labels, img_metas)
        dec_losses = self.get_losses(*dec_loss_inputs)

        dec_losses.update(enc_losses)
        return dec_losses
