import warnings
import numpy as np
import cv2

from mtl.utils.vis_util import imshow
from mtl.utils.io_util import imread, imwrite
from mtl.utils.vis_util import color_val
from ..base_model import BaseModel


class BaseClassifier(BaseModel):
    """Base class for classifiers"""

    def __init__(self):
        super(BaseClassifier, self).__init__()

    def show_result(
        self,
        img,
        result,
        text_color="green",
        font_scale=0.5,
        row_width=20,
        show=False,
        win_name="",
        wait_time=0,
        out_file=None,
    ):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The classification results to draw over `img`.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            font_scale (float): Font scales of texts.
            row_width (int): width between each row of results on the image.
            show (bool): Whether to show the image.
                Default: False.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        if type(img) is np.ndarray:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = img[:, :, [2, 1, 0]]
        else:
            img = imread(img)

        img = img.copy()

        # write results on left-top of the image
        x, y = 0, row_width
        text_color = color_val(text_color)
        for k, v in result.items():
            if isinstance(v, float):
                v = f"{v:.2f}"
            label_text = f"{k}: {v}"
            cv2.putText(
                img,
                label_text,
                (x, y),
                cv2.FONT_HERSHEY_COMPLEX,
                font_scale,
                text_color,
            )
            y += row_width

        if show:
            imshow(img, win_name, wait_time)
        if out_file is not None:
            imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn(
                "show==False and out_file is not specified, only "
                "result image will be returned"
            )
            return img
