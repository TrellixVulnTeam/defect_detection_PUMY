import torch

from mtl.cores.bbox import bbox2result
from ..model_builder import DETECTORS
from .base_detectors import SingleStageDetector


@DETECTORS.register_module()
class YOLOV7(SingleStageDetector):
    """Yolo V7 for detection."""

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        x = self.bbox_head(x)

        bbox_list = self.bbox_head.get_bboxes(x[0], img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results


if __name__ == "__main__":
    pass
