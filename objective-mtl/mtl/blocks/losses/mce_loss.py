import torch
import torch.nn as nn

from mtl.utils.log_util import get_root_logger
from ..block_builder import LOSSES
from .cross_entropy_loss import cross_entropy


@LOSSES.register_module()
class MultiLevelCELoss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=None):
        """MultiLevelCELoss.

        Args:
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(MultiLevelCELoss, self).__init__()
        self.reduction = reduction
        assert isinstance(loss_weight, list)
        self.loss_weight = loss_weight
        self.num_layers = len(loss_weight)

    def forward(
        self,
        cls_scores,
        labels,
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

        loss = 0
        for i in range(self.num_layers):
            loss += self.loss_weight[i] * cross_entropy(
                cls_scores[i],
                labels[i],
                weight,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs,
            )
        if torch.isnan(loss):
            logger = get_root_logger()
            logger.info(
                f"Relevant MultiLevelCELoss Variables: {cls_scores.sum()}, {labels.sum()}"
            )
            # raise ValueError("loss is nan.")
            return torch.Tensor([0]).to(cls_scores.device).sum()
        else:
            return self.loss_weight * loss
