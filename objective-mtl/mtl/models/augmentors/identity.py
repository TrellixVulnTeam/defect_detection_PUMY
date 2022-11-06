import torch.nn.functional as F

from ..model_builder import AUGMENTORS


def one_hot_encoding(gt, num_classes):
    """Change gt_label to one_hot encoding.
    If the shape has 2 or more
    dimensions, return it without encoding.
    Args:
        gt (Tensor): The gt label with shape (N,) or shape (N, */).
        num_classes (int): The number of classes.
    Return:
        Tensor: One hot gt label.
    """
    if gt.ndim == 1:
        # multi-class classification
        return F.one_hot(gt, num_classes=num_classes)
    else:
        return gt


@AUGMENTORS.register_module()
class Identity(object):
    """Change gt_label to one_hot encoding and keep img as the same.
    Args:
        num_classes (int): The number of classes.
        prob (float): MixUp probability. It should be in range [0, 1].
            Default to 1.0
    """

    def __init__(self, num_classes, prob=1.0):
        super(Identity, self).__init__()

        assert isinstance(num_classes, int)
        assert isinstance(prob, float) and 0.0 <= prob <= 1.0

        self.num_classes = num_classes
        self.prob = prob

    def one_hot(self, gt_label):
        return one_hot_encoding(gt_label, self.num_classes)

    def __call__(self, img, gt_label):
        return img, self.one_hot(gt_label)
