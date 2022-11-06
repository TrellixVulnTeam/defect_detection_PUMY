from .cls_cifar import CIFAR10, CIFAR100
from .cls_common import ClsCommonDataset
from .cls_oid import OIDDataset
from .cls_imagenet import ImageNetDataset
from .cls_lupinus import ClsLupinusDataset


__all__ = [
    "CIFAR10",
    "CIFAR100",
    "ClsCommonDataset",
    "OIDDataset",
    "ImageNetDataset",
    "ClsLupinusDataset",
]
