# -------------------------------------------------------
# code adopted from https://github.com/Alibaba-MIIL/ASL
# -------------------------------------------------------

import torch
import torch.nn as nn

from mtl.utils.log_util import get_root_logger
from ..block_builder import LOSSES


@LOSSES.register_module()
class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label classification as described in the paper
    `"Asymmetric Loss For Multi-Label Classification" `
    $ L = -y L_+ - (1-y) L_- $
    $ L+ = (1-p)^{\gamma_+} \log(p)$
    $ L_ = (p_m)^{\gamma_-} \log(1-p_m)$
    $ p_m = max(p-m, 0) $
    $ p = \sigma(x)$, where x is the output logit

    Args:
        gamma_neg: negative focusing parameter
        gamma_pos: positive focusing parameter
        clip: prob margin for asymmetric prob shifting
        eps: for numerical stable
        disable_torch_grad_focal_loss:
    """

    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        pos_weight=1.0,
        clip=0.05,
        eps=1e-5,
        disable_torch_grad_focal_loss=True,
        loss_weight=1.0,
    ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.pos_weight = pos_weight
        self.loss_weight = loss_weight

    def forward(self, x, y, avg_factor=None):
        """
        x: input logits
        y: targets (multi-label binarized vector, i.e. 0-1 vector)
        """
        # Calculating Probabilities
        dtype = x.dtype
        x = x.float()
        x_sigmoid = torch.sigmoid(x).to(dtype)
        # x_sigmoid = torch.sigmoid(x.clamp(max=10))
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1 - self.eps)

        # Basic CE calculation
        los_pos = self.pos_weight * y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            pt = pt.clamp(min=self.eps, max=1 - self.eps)
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        loss = -loss.sum()
        if torch.isnan(loss) or loss < 0:
            logger = get_root_logger()
            logger.info(
                f"Relevant ASL Variables: {x.sum()}, {xs_pos.sum()}, {xs_neg.sum()}, "
                + f"{pt0.sum()}, {pt1.sum()}, {one_sided_gamma.sum()}, "
                + f"{one_sided_w.sum()}"
            )
            # raise ValueError("loss is nan.")
            return torch.Tensor([0]).to(x.device).sum()
        else:
            return self.loss_weight * loss
