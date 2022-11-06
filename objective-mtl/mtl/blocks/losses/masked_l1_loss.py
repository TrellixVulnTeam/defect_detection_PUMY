import torch
import torch.nn as nn
import torch.nn.functional as F

from mtl.utils.log_util import get_root_logger
from ..block_builder import LOSSES


def l1_loss(pred, target, reduction="none"):
    """L1 loss.
    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
    Returns:
        torch.Tensor: Calculated loss
    """
    loss = F.l1_loss(pred, target, reduction=reduction)
    return loss


@LOSSES.register_module()
class MaskedL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction="none", loss_weight=1.0):
        super(MaskedL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, mask, in_chans, reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        loss_recon = l1_loss(pred, target, reduction=reduction)
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / in_chans
        if torch.isnan(loss):
            logger = get_root_logger()
            logger.info(
                f"Relevant MaskedL1Loss Variables: {pred.sum()}, {target.sum()}, {mask.sum()}"
            )
            # raise ValueError("loss is nan.")
            return torch.Tensor([0]).to(pred.device).sum()
        else:
            return self.loss_weight * loss
