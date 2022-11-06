import torch
import torch.nn as nn

from mtl.utils.log_util import get_root_logger
from ..block_builder import LOSSES


@LOSSES.register_module()
class BCELoss(nn.Module):
    def __init__(self, reduction: str = "mean", loss_weight: float = 1.0):
        super(BCELoss, self).__init__()
        self.bce = nn.BCELoss(reduction=reduction)
        self.loss_weight = loss_weight

    def forward(
        self,
        logits,
        targets,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):

        loss = self.bce(logits, targets.float())
        if torch.isnan(loss):
            logger = get_root_logger()
            logger.info(
                f"Relevant SigmoidBCELoss Variables: {logits.sum()}, {targets.sum()}"
            )
            # raise ValueError("loss is nan.")
            return torch.Tensor([0]).to(logits.device).sum()
        else:
            return self.loss_weight * loss


@LOSSES.register_module()
class SigmoidBCELoss(nn.Module):
    def __init__(self, reduction: str = "mean", loss_weight: float = 1.0):
        super(SigmoidBCELoss, self).__init__()
        # self.sigmoid = nn.Sigmoid()
        # self.bce = nn.BCELoss(reduction=reduction)
        self.sigmoid_bce = nn.BCEWithLogitsLoss(reduction=reduction)
        self.loss_weight = loss_weight

    def forward(
        self,
        logits,
        targets,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):

        loss = self.sigmoid_bce(logits, targets.float())
        if torch.isnan(loss):
            logger = get_root_logger()
            logger.info(
                f"Relevant SigmoidBCELoss Variables: {logits.sum()}, {targets.sum()}"
            )
            # raise ValueError("loss is nan.")
            return torch.Tensor([0]).to(logits.device).sum()
        else:
            return self.loss_weight * loss
