import torch
import torch.nn as nn
import torch.nn.functional as F

from mtl.utils.log_util import get_root_logger
from ..block_builder import LOSSES


@LOSSES.register_module()
class MultilevelKLLoss(nn.Module):
    """KL loss Distilling the Knowledge in a Neural Network"""

    def __init__(self, temperature=4, reduction="mean", loss_weights=None):
        super(MultilevelKLLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.loss_weights = loss_weights

    def forward(self, student_outputs, teacher_outputs):
        loss = torch.Tensor([0]).to(student_outputs[0].device).sum()
        i = 0
        for student_output, teacher_output in zip(student_outputs, teacher_outputs):
            p_s = F.log_softmax(student_output / self.temperature, dim=1)
            p_t = F.softmax(teacher_output / self.temperature, dim=1)

            single_loss = (
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
            else:
                loss += self.loss_weights[i] * single_loss
            i += 1

        return loss
