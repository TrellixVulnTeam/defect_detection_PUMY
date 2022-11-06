import warnings
from abc import ABCMeta, abstractmethod
import numpy as np

from mtl.utils.io_util import imread, imwrite
from mtl.utils.vis_util import imshow
from ..base_model import BaseModel


class BaseSegmentor(BaseModel, metaclass=ABCMeta):
    """Base class for segmentors."""

    def __init__(self):
        super(BaseSegmentor, self).__init__()

    @property
    def with_auxiliary_head(self):
        """bool: whether the segmentor has auxiliary head"""
        return hasattr(self, "auxiliary_head") and self.auxiliary_head is not None

    @property
    def with_decode_head(self):
        """bool: whether the segmentor has decode head"""
        return hasattr(self, "decode_head") and self.decode_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        """Placeholder for extract features from images."""
        pass

    @abstractmethod
    def encode_decode(self, img, img_metas):
        """Placeholder for encode images with backbone and decode into a
        semantic segmentation map of the same size as input."""
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Placeholder for augmentation test."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see `Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self(inputs)
        # seg_logits = torch.clamp(seg_logits, min=1e-5)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, "imgs"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError(f"{name} must be a list, but got " f"{type(var)}")

        num_augs = len(imgs)
        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas, **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

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
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = imread(img, channel_order="rgb")
        img = img.copy()
        seg = result[0]
        if palette is None:
            if self.palette is None:
                palette = np.random.randint(0, 255, size=(len(self.class_names), 3))
            else:
                palette = self.palette
        palette = np.array(palette)
        assert palette.shape[0] == len(self.class_names)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        # img = img * 0.5 + color_seg * 0.5
        img = img.astype(np.uint8)
        # img = color_seg.astype(np.uint8)

        if show:
            imshow(img, win_name, wait_time)
        if out_file is not None:
            imwrite(img, out_file)
            imwrite(color_seg, out_file[:-3] + "png")

        if not (show or out_file):
            warnings.warn(
                "show==False and out_file is not specified, only "
                "result image will be returned"
            )
            return img
        else:
            return None
