
from mtl.cores.bbox import bbox2result
from ..model_builder import DETECTORS
from .base_detectors import SingleStageDetector
import torch

@DETECTORS.register_module()
class YOLOX(SingleStageDetector):
    def __init__(self, cfg):
        super(YOLOX, self).__init__(cfg)

    def forward_train(
        self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs
    ):
        x = self.extract_feat(img)
        loss, iou_loss, conf_loss, cls_loss, l1_loss = self.bbox_head.forward_train(
            x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore
        )
        outputs = {
            "total_loss": loss,
            "iou_loss": iou_loss,
            "l1_loss": l1_loss,
            "conf_loss": conf_loss,
            "cls_loss": cls_loss,
        }

        return outputs

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.
        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            mlres_list = self.bbox_head.get_bboxes(
                *outs, img_metas, rescale=rescale, with_nms=False
            )
            return mlres_list
        else:
            print("scare:"+str(len(outs)))
            bbox_list = self.bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
            
            print("-----------test  yolox.py-----------------")
            #bbox_list = torch.squeeze(torch.tensor(bbox_list, device="cpu"))
            #print(bbox_list.shape)
            #t_det_bboxes=bbox_list[:,:5]
            #t_det_labels=bbox_list[:,5]
            #bbox_list =[t_det_bboxes,t_det_labels]
            bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
            ]
            #bbox_results = [
            #    bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            #    for det_bboxes, det_labels in bbox_list]
            #bbox_results = [bbox2result( t_det_bboxes,t_det_labels,self.bbox_head.num_classes)]
            print(bbox_results)
            return bbox_results
