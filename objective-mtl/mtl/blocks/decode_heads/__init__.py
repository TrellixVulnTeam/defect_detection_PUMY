from .base_decode_head import BaseDecodeHead
from .base_cascade_decode_head import BaseCascadeDecodeHead
from .fcn_head import SegFCNHead
from .fpn_head import SegFPNHead
from .ocr_head import OCRHead
from .uper_head import UPerHead


__all__ = [
    "BaseDecodeHead",
    "BaseCascadeDecodeHead",
    "SegFCNHead",
    "SegFPNHead",
    "OCRHead",
    "UPerHead",
]
