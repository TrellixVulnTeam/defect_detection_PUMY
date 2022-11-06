from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn as nn

from mtl.utils.log_util import print_log


class BaseModel(nn.Module, metaclass=ABCMeta):
    """Base model"""

    def __init__(self):
        super(BaseModel, self).__init__()

    @property
    def with_neck(self):
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_head(self):
        return hasattr(self, "head") and self.head is not None

    @abstractmethod
    def forward_train(self, img, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, img, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log(f"load model from: {pretrained}", logger="root")

    def forward_test(self, img, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
        """
        return self.simple_test(img, **kwargs)

    def forward(self, img, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        training. Note this setting will change the expected inputs.
        When training, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if self.training:
            losses = self.forward_train(img, **kwargs)
            loss, log_vars = self._parse_losses(losses)
            outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(img))
        else:
            outputs = self.forward_test(img, **kwargs)

        return outputs

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, float):
                log_vars[loss_name] = loss_value
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if not isinstance(loss_value, (float, int)):
                if dist.is_available() and dist.is_initialized():
                    loss_value = loss_value.data.clone()
                    dist.all_reduce(loss_value.div_(dist.get_world_size()))
                log_vars[loss_name] = loss_value.item()
            else:
                log_vars[loss_name] = loss_value

        return loss, log_vars
