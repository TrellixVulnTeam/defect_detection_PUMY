import torch
import numpy as np

from mtl.cores.layer_ops.layer_resize import resize
from ..model_builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class EncoderDecoderHeatMap(EncoderDecoder):
    """Decoder using heatmap"""

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        assert self.test_cfg["mode"] in ["slide", "whole"]
        ori_shape = img_meta[0]["ori_shape"]
        assert all(_["ori_shape"] == ori_shape for _ in img_meta)
        if self.test_cfg["mode"] == "slide":
            seg_logit = self._slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self._whole_inference(img, img_meta, rescale)
        flip = img_meta[0]["flip"]
        if flip:
            flip_direction = img_meta[0]["flip_direction"]
            assert flip_direction in ["horizontal", "vertical"]
            if flip_direction == "horizontal":
                seg_logit = seg_logit.flip(dims=(3,))
            elif flip_direction == "vertical":
                seg_logit = seg_logit.flip(dims=(2,))
        if torch.onnx.is_in_onnx_export():
            return seg_logit
        seg_logit = seg_logit.cpu().numpy()
        return seg_logit

    def get_losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )

        loss["loss_seg"] = self.compute_loss(seg_logit, seg_label)
        return loss

    def show_result(
        self,
        img,
        result,
        palette=None,
        win_name="",
        show=False,
        wait_time=0,
        out_file=None,
    ):
        seg = np.zeros((result.shape[-2], result.shape[-1]), dtype=np.uint8)
        for i in range(len(result.shape[1])):
            seg_slice = result[0, i, :, :] > 0.3
            seg += seg_slice
        return super(EncoderDecoderHeatMap, self).show_result(
            img, result, palette, win_name, show, wait_time, out_file
        )
