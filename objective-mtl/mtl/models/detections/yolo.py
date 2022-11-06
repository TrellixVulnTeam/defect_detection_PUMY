from ..model_builder import DETECTORS
from .base_detectors import SingleStageDetector


@DETECTORS.register_module()
class YOLO(SingleStageDetector):
    """Yolo model for V3~V5"""

    def __init__(self, cfg):
        super(YOLO, self).__init__(cfg)
