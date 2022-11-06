import torch
import torch.nn as nn
import torch.nn.functional as F

from mtl.utils.log_util import get_root_logger
from mtl.utils.loss_util import weight_reduce_loss
from ..block_builder import LOSSES


def cross_entropy(
    pred,
    label,
    weight=None,
    reduction="mean",
    avg_factor=None,
    class_weight=None,
    ignore_index=-100,
):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = F.cross_entropy(
        pred,
        label.long(),
        weight=class_weight,
        reduction="none",
        ignore_index=ignore_index,
    )

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor
    )

    return loss


def _expand_onehot_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(
        (labels >= 0) & (labels < label_channels), as_tuple=False
    ).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels
        )

    return bin_labels, bin_label_weights


def _expand_onehot_seg_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights


def binary_cross_entropy(
    pred,
    label,
    weight=None,
    reduction="mean",
    avg_factor=None,
    class_weight=None,
    ignore_index=None,
):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        if ignore_index is None:
            label, weight = _expand_onehot_labels(label, weight, pred.size(-1))
        else:
            assert (pred.dim() == 2 and label.dim() == 1) or (
                pred.dim() == 4 and label.dim() == 3
            ), (
                "Only pred shape [N, C], label shape [N] or pred shape [N, C, "
                "H, W], label shape [N, H, W] are supported"
            )
            label, weight = _expand_onehot_seg_labels(
                label, weight, pred.shape, ignore_index
            )

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction="none"
    )
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(
    pred,
    target,
    label,
    reduction="mean",
    avg_factor=None,
    class_weight=None,
    ignore_index=None,
):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    # TODO: handle these two reserved arguments
    assert ignore_index is None, "BCE loss does not support ignore_index"
    assert reduction == "mean" and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction="mean"
    )[None]


def soft_cross_entropy(
    pred, label, weight=None, reduction="mean", class_weight=None, avg_factor=None
):
    """Calculate the Soft CrossEntropy loss.

    The label can be float.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction with shape (N, C).
            When using "mixup", the label can be float.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = -label * F.log_softmax(pred, dim=-1)
    if class_weight is not None:
        loss *= class_weight
    loss = loss.sum(dim=-1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor
    )

    return loss


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        use_sigmoid=False,
        use_mask=False,
        use_soft=False,
        reduction="mean",
        class_weight=None,
        loss_weight=1.0,
    ):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.use_soft = use_soft
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        elif self.use_soft:
            self.cls_criterion = soft_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device
            )
        else:
            class_weight = None
        loss = self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        if torch.isnan(loss):
            logger = get_root_logger()
            logger.info(
                f"Relevant CrossEntropyLoss Variables: {cls_score.sum()}, {label.sum()}, "
            )
            # raise ValueError("loss is nan.")
            return torch.Tensor([0]).to(cls_score.device).sum()
        else:
            return self.loss_weight * loss


@LOSSES.register_module()
class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, **kwargs):
        super(BCEWithLogitsLoss, self).__init__(**kwargs)


@LOSSES.register_module()
class ContrastiveCELoss(nn.Module):
    """Contrastive cross entropy loss"""

    def __init__(self, temperature=0.1, loss_weight=1.0):
        super(ContrastiveCELoss, self).__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight

    def forward(self, x, x_k, q, q_k):
        l_1 = q * F.log_softmax(x_k / self.temperature, dim=1)
        l_1 = -torch.mean(torch.sum(l_1, dim=1))
        l_2 = q_k * F.log_softmax(x / self.temperature, dim=1)
        l_2 = -torch.mean(torch.sum(l_2, dim=1))

        loss = l_1 + l_2

        if torch.isnan(loss):
            logger = get_root_logger()
            logger.info(
                f"Relevant ContrastiveCELoss Variables: {x.sum()}, "
                + f"{x_k.sum()}, {q.sum()}, {q_k.sum()}, "
                + f"{l_1}, {l_2}"
            )
            # raise ValueError("loss is nan.")
            return torch.Tensor([0]).to(x.device).sum()
        else:
            return self.loss_weight * loss
