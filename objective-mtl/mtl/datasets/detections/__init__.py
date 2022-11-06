from .det_base import DetBaseDataset
from .det_voc import VOCDataset
from .det_coco import CocoDataset
from .det_animal import DetAnimalDataset
from .det_multiobj import MultiObjectDataset
from .det_oneobj import OneObjectDataset
from .det_lupinus import DetLupinusDataset


__all__ = [
    "DetBaseDataset",
    "VOCDataset",
    "CocoDataset",
    "DetAnimalDataset",
    "MultiObjectDataset",
    "OneObjectDataset",
    "DetLupinusDataset",
]
