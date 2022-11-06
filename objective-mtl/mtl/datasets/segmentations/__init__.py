from .seg_base import SegBaseDataset
from .seg_voc import SegVOCDataset
from .seg_cityscapes import CityscapesDataset
from .seg_ade20k import ADE20KDataset


__all__ = ["SegBaseDataset", "SegVOCDataset", "CityscapesDataset", "ADE20KDataset"]
