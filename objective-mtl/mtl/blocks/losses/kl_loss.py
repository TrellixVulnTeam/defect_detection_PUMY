import torch
import torch.nn as nn
import torch.nn.functional as F

from mtl.utils.log_util import get_root_logger
from ..block_builder import LOSSES


@LOSSES.register_module()
class KLLoss(nn.Module):
    """KL loss Distilling the Knowledge in a Neural Network"""

    def __init__(self, temperature=4, reduction="mean", loss_weight=1.0):
        super(KLLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, student_output, teacher_output):
        p_s = F.log_softmax(student_output / self.temperature, dim=1)
        p_t = F.softmax(teacher_output / self.temperature, dim=1)

        loss = (
            F.kl_div(p_s, p_t, size_average=False, reduction=self.reduction)
            * (self.temperature ** 2)
            / student_output.shape[0]
        )
        if torch.isnan(loss):
            logger = get_root_logger()
            logger.info(
                f"Relevant KLLoss Variables: {student_output.sum()}, {teacher_output.sum()}"
            )
            # raise ValueError("loss is nan.")
            return torch.Tensor([0]).to(student_output.device).sum()
        else:
            return self.loss_weight * loss


@LOSSES.register_module()
class DualKLLoss(nn.Module):
    """KL loss Distilling the Knowledge in a Neural Network"""

    def __init__(self, reduction="none", loss_weight=1.0):
        super(DualKLLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, p, q, pad_mask=None):
        p_loss = F.kl_div(
            F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction=self.reduction
        )
        q_loss = F.kl_div(
            F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction=self.reduction
        )

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.0)
            q_loss.masked_fill_(pad_mask, 0.0)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss * self.loss_weight
