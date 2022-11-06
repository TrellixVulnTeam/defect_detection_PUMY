from typing import List
import torch
from torch import nn
from torch.nn.functional import cross_entropy

from mtl.utils.log_util import get_root_logger
from ..block_builder import LOSSES


@LOSSES.register_module()
class ClassificationCircleLoss(nn.Module):
    """Circle loss for class-level labels as described in the paper

    Args:
        scale (float): the scale factor. Default: 256.0
        margin (float): the relax margin value. Default: 0.25
        circle_center (list[float]): the center of the circle (logit_ap, logit_an). Default: (1, 0)
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    """

    def __init__(
        self,
        scale: float = 256.0,
        margin: float = 0.25,
        circle_center: List = [1, 0],
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super(ClassificationCircleLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.circle_center = circle_center
        self.reduction = reduction
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
        """
        Args:
            logits (torch.Tensor): The predicted logits before softmax,
                namely :math:`\cos \theta` in the above equation, with shape of :math:`(N, C)`
            targets (torch.LongTensor): The ground-truth label long vector,
                namely :math:`y` in the above equation, with shape of :math:`(N,)`

        Returns:
            torch.Tensor: loss
                the computed loss
        """
        mask = torch.zeros(
            logits.shape, dtype=torch.bool, device=logits.device
        ).scatter_(dim=1, index=targets[1], value=1)
        positive_weighting = torch.clamp(
            self.circle_center[0] + self.margin - logits.detach(), min=0
        )
        negative_weighting = torch.clamp(
            logits.detach() - self.circle_center[1] + self.margin, min=0
        )
        logits = torch.where(
            mask,
            self.scale
            * positive_weighting
            * (logits - (self.circle_center[0] - self.margin)),
            self.scale
            * negative_weighting
            * (logits - self.circle_center[1] - self.margin),
        )
        loss = cross_entropy(input=logits, target=targets, reduction=self.reduction)

        if torch.isnan(loss):
            logger = get_root_logger()
            logger.info(
                f"Relevant CircleLoss Variables: {logits.sum()}, {targets.sum()}, "
                + f"{mask.sum()}"
            )
            # raise ValueError("loss is nan.")
            return torch.Tensor([0]).to(logits.device).sum()
        else:
            return self.loss_weight * loss
