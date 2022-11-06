import numpy as np
import random
import math
import cv2
from numbers import Number
from typing import Sequence
from PIL import Image, ImageOps

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

from mtl.utils.geometric_util import imresize, imflip, impad, impad_to_multiple, imcrop
from mtl.utils.photometric_util import imnormalize, bgr2hsv, hsv2bgr, rgb2gray
from ..data_wrapper import PIPELINES


@PIPELINES.register_module()
class ImgResizeWithPad:
    """Resize images for classification."""

    def __init__(self, size, interpolation="bilinear", backend="cv2", pad_value=-1):
        """Initialization of resize operator for classification.
        Args:
            size (int | list): Images scales for resizing (h, w).
                When size is int, the default behavior is to resize an image
                to (size, size). When size is tuple and the second value is -1,
                the long edge of an image is resized to its first value.
                For example, when size is 384, the image is resized to 384x384.
                When size is (384, -1), the long side is resized to 384 and the
                other side is computed based on the long side, maintaining the
                aspect ratio.
                After that, the image is padded with pad_value.
            interpolation (str): Interpolation method, accepted values are
                "nearest", "bilinear", "bicubic", "area", "lanczos".
                More details can be found in `geometric`.
            backend (str): The image resize backend type, accpeted values are
                `cv2` and `pillow`. Default: `cv2`.
        """
        assert isinstance(size, int) or (isinstance(size, list) and len(size) == 2)
        self.resize_w_long_side = False
        if isinstance(size, int):
            assert size > 0
            size = (size, size)
        else:
            assert size[0] > 0 and (size[1] > 0 or size[1] == -1)
            if size[1] == -1:
                self.resize_w_long_side = True
        assert interpolation in ("nearest", "bilinear", "bicubic", "area", "lanczos")
        if backend not in ["cv2", "pillow"]:
            raise ValueError(
                f"backend: {backend} is not supported for resize."
                'Supported backends are "cv2", "pillow"'
            )

        self.size = size
        self.interpolation = interpolation
        self.backend = backend
        self.pad_value = pad_value

    def _resize_img(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            ignore_resize = False
            if self.resize_w_long_side:
                h, w = img.shape[:2]
                long_side = self.size[0]
                if (w <= h and h == long_side) or (h <= w and w == long_side):
                    ignore_resize = True
                else:
                    if w < h:
                        height = long_side
                        width = int(long_side * w / h)
                    else:
                        width = long_side
                        height = int(long_side * h / w)
            else:
                height, width = self.size
            if not ignore_resize:
                img = imresize(
                    img,
                    size=(width, height),
                    interpolation=self.interpolation,
                    return_scale=False,
                    backend=self.backend,
                )

            # image padding with pad_value
            padded_img = impad(
                img, shape=(self.size[0], self.size[0]), pad_val=self.pad_value
            )
            results[key] = padded_img
            results["img_shape"] = padded_img.shape

    def __call__(self, results):
        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"interpolation={self.interpolation})"
        return repr_str


@PIPELINES.register_module()
class ImgResize:
    """Resize images for classification."""

    def __init__(self, size, interpolation="bilinear", backend="cv2"):
        """Initialization of resize operator for classification.

        Args:
            size (int | tuple): Images scales for resizing (h, w).
                When size is int, the default behavior is to resize an image
                to (size, size). When size is tuple and the second value is -1,
                the short edge of an image is resized to its first value.
                For example, when size is 224, the image is resized to 224x224.
                When size is (224, -1), the short side is resized to 224 and the
                other side is computed based on the short side, maintaining the
                aspect ratio.
            interpolation (str): Interpolation method, accepted values are
                "nearest", "bilinear", "bicubic", "area", "lanczos".
                More details can be found in `geometric`.
            backend (str): The image resize backend type, accpeted values are
                `cv2` and `pillow`. Default: `cv2`.
        """
        assert isinstance(size, int) or (
            isinstance(size, (tuple, list)) and len(size) == 2
        )
        self.resize_w_short_side = False
        if isinstance(size, int):
            assert size > 0
            size = (size, size)
        else:
            assert size[0] > 0 and (size[1] > 0 or size[1] == -1)
            if size[1] == -1:
                self.resize_w_short_side = True
        assert interpolation in ("nearest", "bilinear", "bicubic", "area", "lanczos")
        if backend not in ["cv2", "pillow"]:
            raise ValueError(
                f"backend: {backend} is not supported for resize."
                'Supported backends are "cv2", "pillow"'
            )

        self.size = size
        self.interpolation = interpolation
        self.backend = backend

    def _resize_img(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            ignore_resize = False
            if self.resize_w_short_side:
                h, w = img.shape[:2]
                short_side = self.size[0]
                if (w <= h and w == short_side) or (h <= w and h == short_side):
                    ignore_resize = True
                else:
                    if w < h:
                        width = short_side
                        height = int(short_side * h / w)
                    else:
                        height = short_side
                        width = int(short_side * w / h)
            else:
                height, width = self.size
            if not ignore_resize:
                img = imresize(
                    img,
                    size=(width, height),
                    interpolation=self.interpolation,
                    return_scale=False,
                    backend=self.backend,
                )
                results[key] = img
                results["img_shape"] = img.shape

    def __call__(self, results):
        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"interpolation={self.interpolation})"
        return repr_str


@PIPELINES.register_module()
class RandomGrayscale:
    """Randomly convert image to grayscale with a probability of gray_prob."""

    def __init__(self, gray_prob=0.1):
        """Initialization for random grayscale

        Args:
            gray_prob (float): Probability that image should be converted to
                grayscale. Default: 0.1.

        Returns:
            ndarray: Grayscale version of the input image with probability
                gray_prob and unchanged with probability (1-gray_prob).
                - If input image is 1 channel: grayscale version is 1 channel.
                - If input image is 3 channel: grayscale version is 3 channel
                    with r == g == b.
        """
        self.gray_prob = gray_prob

    def __call__(self, results):
        """
        Args:
            img (ndarray): Image to be converted to grayscale.

        Returns:
            ndarray: Randomly grayscaled image.
        """
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            num_output_channels = img.shape[2]
            if random.random() < self.gray_prob:
                if num_output_channels > 1:
                    img = rgb2gray(img)[:, :, None]
                    results[key] = np.dstack([img for _ in range(num_output_channels)])
                    return results
            results[key] = img
        return results

    def __repr__(self):

        return self.__class__.__name__ + f"(gray_prob={self.gray_prob})"


@PIPELINES.register_module()
class ImgRandomFlip:
    """Flip the image randomly.
    Flip the image randomly based on flip probaility and flip direction.
    """

    def __init__(self, flip_prob=0.5, direction="horizontal"):
        """Initialization of random flip for classification

        Args:
            flip_prob (float): probability of the image being flipped. Default: 0.5
            direction (str, optional): The flipping direction. Options are
                'horizontal' and 'vertical'. Default: 'horizontal'.
        """
        assert 0 <= flip_prob <= 1
        assert direction in ["horizontal", "vertical"]
        self.flip_prob = flip_prob
        self.direction = direction

    def __call__(self, results):
        """Call function to flip image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        flip = True if np.random.rand() < self.flip_prob else False
        results["flip"] = flip
        results["flip_direction"] = self.direction
        if results["flip"]:
            # flip image
            for key in results.get("img_fields", ["img"]):
                results[key] = imflip(results[key], direction=results["flip_direction"])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(flip_prob={self.flip_prob})"


@PIPELINES.register_module()
class Pad:
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0, seg_pad_val=255):
        """Initialization for padding images

        Args:
            size (tuple, optional): Fixed padding size.
            size_divisor (int, optional): The divisor of padded size.
            pad_val (float, optional): Padding value, 0 by default.
        """
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        for key in results.get("img_fields", ["img"]):
            if self.size is not None:
                padded_img = impad(results[key], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                padded_img = impad_to_multiple(
                    results[key], self.size_divisor, pad_val=self.pad_val
                )
            results[key] = padded_img
        results["pad_shape"] = padded_img.shape
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results["pad_shape"][:2]
        for key in results.get("mask_fields", []):
            results[key] = results[key].pad(pad_shape, pad_val=self.pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        for key in results.get("seg_fields", []):
            results[key] = impad(
                results[key], shape=results["pad_shape"][:2], pad_val=self.seg_pad_val
            )

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_val={self.pad_val})"
        repr_str += f"seg_pad_val={self.seg_pad_val})"
        return repr_str


@PIPELINES.register_module()
class Normalize:
    """Normalize the image."""

    def __init__(self, mean, std, to_rgb=True):
        """Initialization for normalization.

        Args:
            mean (sequence): Mean values of 3 channels.
            std (sequence): Std values of 3 channels.
            to_rgb (bool): Whether to convert the image from BGR to RGB,
                default is true.
        """
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get("img_fields", ["img"]):
            results[key] = imnormalize(results[key], self.mean, self.std, self.to_rgb)
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@PIPELINES.register_module()
class ImgRandomCrop:
    """Crop the given Image at a random location."""

    def __init__(
        self,
        size,
        padding=None,
        pad_if_needed=False,
        pad_val=0,
        padding_mode="constant",
    ):
        """Initialization of random crop for classification

        Args:
            size (sequence or int): Desired output size of the crop. If size is an
                int instead of sequence like (h, w), a square crop (size, size) is
                made.
            padding (int or sequence, optional): Optional padding on each border
                of the image. If a sequence of length 4 is provided, it is used to
                pad left, top, right, bottom borders respectively.  If a sequence
                of length 2 is provided, it is used to pad left/right, top/bottom
                borders, respectively. Default: None, which means no padding.
            pad_if_needed (boolean): It will pad the image if smaller than the
                desired size to avoid raising an exception. Since cropping is done
                after padding, the padding seems to be done at a random offset.
                Default: False.
            pad_val (Number | Sequence[Number]): Pixel pad_val value for constant
                fill. If a tuple of length 3, it is used to pad_val R, G, B
                channels respectively. Default: 0.
            padding_mode (str): Type of padding. Should be: constant, edge,
                reflect or symmetric. Default: constant.
                -constant: Pads with a constant value, this value is specified
                    with pad_val.
                -edge: pads with the last value at the edge of the image.
                -reflect: Pads with reflection of image without repeating the
                    last value on the edge. For example, padding [1, 2, 3, 4]
                    with 2 elements on both sides in reflect mode will result
                    in [3, 2, 1, 2, 3, 4, 3, 2].
                -symmetric: Pads with reflection of image repeating the last
                    value on the edge. For example, padding [1, 2, 3, 4] with
                    2 elements on both sides in symmetric mode will result in
                    [2, 1, 1, 2, 3, 4, 4, 3].
        """
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        # check padding mode
        assert padding_mode in ["constant", "edge", "reflect", "symmetric"]
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.pad_val = pad_val
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``crop`` for random crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        target_height, target_width = output_size
        if width == target_width and height == target_height:
            return 0, 0, height, width

        xmin = np.random.randint(0, height - target_height)
        ymin = np.random.randint(0, width - target_width)
        return xmin, ymin, target_height, target_width

    def __call__(self, results):
        """Call for running

        Args:
            img (ndarray): Image to be cropped.
        """
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            if self.padding is not None:
                img = impad(img, padding=self.padding, pad_val=self.pad_val)

            # pad the height if needed
            if self.pad_if_needed and img.shape[0] < self.size[0]:
                img = impad(
                    img,
                    padding=(
                        0,
                        self.size[0] - img.shape[0],
                        0,
                        self.size[0] - img.shape[0],
                    ),
                    pad_val=self.pad_val,
                    padding_mode=self.padding_mode,
                )

            # pad the width if needed
            if self.pad_if_needed and img.shape[1] < self.size[1]:
                img = impad(
                    img,
                    padding=(
                        self.size[1] - img.shape[1],
                        0,
                        self.size[1] - img.shape[1],
                        0,
                    ),
                    pad_val=self.pad_val,
                    padding_mode=self.padding_mode,
                )

            xmin, ymin, height, width = self.get_params(img, self.size)
            results[key] = imcrop(
                img, np.array([ymin, xmin, ymin + width - 1, xmin + height - 1])
            )
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.size}, padding={self.padding})"


@PIPELINES.register_module()
class ImgCenterCrop:
    """Center crop the image."""

    def __init__(self, size):
        """Initialization of center crop for classification.

        Args:
            crop_size (int | tuple): Expected size after cropping, (h, w).

        Notes:
            If the image is smaller than the crop size, return the original image
        """
        assert isinstance(size, int) or (
            isinstance(size, (tuple, list)) and len(size) == 2
        )
        if isinstance(size, int):
            size = (size, size)
        assert size[0] > 0 and size[1] > 0
        self.crop_size = size

    def __call__(self, results):
        crop_height, crop_width = self.crop_size[0], self.crop_size[1]
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            # img.shape has length 2 for grayscale, length 3 for color
            img_height, img_width = img.shape[:2]

            y1 = max(0, int(round((img_height - crop_height) / 2.0)))
            x1 = max(0, int(round((img_width - crop_width) / 2.0)))
            y2 = min(img_height, y1 + crop_height) - 1
            x2 = min(img_width, x1 + crop_width) - 1

            # crop the image
            img = imcrop(img, bboxes=np.array([x1, y1, x2, y2]))
            img_shape = img.shape
            results[key] = img
        results["img_shape"] = img_shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(crop_size={self.crop_size})"


@PIPELINES.register_module()
class ImgRandomResizedCrop:
    """Crop the given image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation="bilinear",
        backend="cv2",
    ):
        """Initialization of random resized crop for classification

        Args:
            size (sequence or int): Desired output size of the crop. If size is an
                int instead of sequence like (h, w), a square crop (size, size) is
                made.
            scale (tuple): Range of the random size of the cropped image compared
                to the original image. Default: (0.08, 1.0).
            ratio (tuple): Range of the random aspect ratio of the cropped image
                compared to the original image. Default: (3. / 4., 4. / 3.).
            interpolation (str): Interpolation method, accepted values are
                'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Default:
                'bilinear'.
            backend (str): The image resize backend type, accpeted values are
                `cv2` and `pillow`. Default: `cv2`.
        """
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError(
                "range should be of kind (min, max). " f"But received {scale}"
            )
        if backend not in ["cv2", "pillow"]:
            raise ValueError(
                f"backend: {backend} is not supported for resize."
                'Supported backends are "cv2", "pillow"'
            )

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.backend = backend

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (ndarray): Image to be cropped.
            scale (tuple): Range of the random size of the cropped image
                compared to the original image size.
            ratio (tuple): Range of the random aspect ratio of the cropped
                image compared to the original image area.

        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``crop`` for a random sized crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        area = height * width

        for _ in range(10):
            target_area = np.random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))

            target_width = int(round(math.sqrt(target_area * aspect_ratio)))
            target_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_width <= width and 0 < target_height <= height:
                xmin = np.random.randint(0, height - target_height + 1)
                ymin = np.random.randint(0, width - target_width + 1)
                return xmin, ymin, target_height, target_width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            target_width = width
            target_height = int(round(target_width / min(ratio)))
        elif in_ratio > max(ratio):
            target_height = height
            target_width = int(round(target_height * max(ratio)))
        else:  # whole image
            target_width = width
            target_height = height
        xmin = (height - target_height) // 2
        ymin = (width - target_width) // 2
        return xmin, ymin, target_height, target_width

    def __call__(self, results):
        """Call for running
        Args:
            img (ndarray): Image to be cropped and resized.

        Returns:
            ndarray: Randomly cropped and resized image.
        """
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            xmin, ymin, target_height, target_width = self.get_params(
                img, self.scale, self.ratio
            )
            img = imcrop(
                img,
                np.array(
                    [ymin, xmin, ymin + target_width - 1, xmin + target_height - 1]
                ),
            )
            results[key] = imresize(
                img,
                tuple(self.size[::-1]),
                interpolation=self.interpolation,
                backend=self.backend,
            )
        return results

    def __repr__(self):
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={self.interpolation})"
        return format_string


@PIPELINES.register_module()
class PhotoMetricDistortion:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        """Initialization for photo metric distortion.

        Args:
            brightness_delta (int): delta of brightness.
            contrast_range (tuple): range of contrast.
            saturation_range (tuple): range of saturation.
            hue_delta (int): delta of hue.
        """
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        if "img_fields" in results:
            assert results["img_fields"] == ["img"], "Only single img_fields is allowed"
        img = results["img"].astype(np.float32)
        assert img.dtype == np.float32, (
            "PhotoMetricDistortion needs the input image of dtype np.float32,"
            " please set to_float32=True"
        )

        # random brightness
        if np.random.randint(2):
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            if np.random.randint(2):
                alpha = np.random.uniform(self.contrast_lower, self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = bgr2hsv(img)

        # random saturation
        if np.random.randint(2):
            img[..., 1] *= np.random.uniform(
                self.saturation_lower, self.saturation_upper
            )

        # random hue
        if np.random.randint(2):
            img[..., 0] += np.random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = hsv2bgr(img)

        # random contrast
        if mode == 0:
            if np.random.randint(2):
                alpha = np.random.uniform(self.contrast_lower, self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if np.random.randint(2):
            img = img[..., np.random.permutation(3)]

        results["img"] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


@PIPELINES.register_module()
class VideoPhotoMetricDistortion:
    """
    Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if np.random.randint(2):
            beta = random.uniform(-self.brightness_delta, self.brightness_delta)
            img_ = []
            for i_img in img:
                img_.append(self.convert(i_img, beta=beta))
            return img_
        else:
            return img

    def contrast(self, img):
        """Contrast distortion."""
        if np.random.randint(2):
            alpha = random.uniform(self.contrast_lower, self.contrast_upper)
            img_ = []
            for i_img in img:
                img_.append(self.convert(i_img, alpha=alpha))
            return img_
        else:
            return img

    def saturation(self, img):
        """Saturation distortion."""
        if np.random.randint(2):
            alpha = random.uniform(self.saturation_lower, self.saturation_upper)
            img_ = []
            for i_img in img:
                i_img = bgr2hsv(i_img)
                i_img[:, :, 1] = self.convert(i_img[:, :, 1], alpha=alpha)

                i_img = hsv2bgr(i_img)
                img_.append(i_img)
            return img_
        else:
            return img

    def hue(self, img):
        """Hue distortion."""
        if np.random.randint(2):
            hue_val = random.randint(-self.hue_delta, self.hue_delta)
            img_ = []
            for i_img in img:

                i_img = bgr2hsv(i_img)
                i_img[:, :, 0] = (i_img[:, :, 0].astype(int) + hue_val) % 180
                i_img = hsv2bgr(i_img)
                img_.append(i_img)
            return img_
        else:
            return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        img = []
        for k in results.get("img_fields", []):
            img.append(results[k])
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        for i, k in enumerate(results.get("img_fields", [])):
            results[k] = img[i]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(brightness_delta={self.brightness_delta}, "
            f"contrast_range=({self.contrast_lower}, "
            f"{self.contrast_upper}), "
            f"saturation_range=({self.saturation_lower}, "
            f"{self.saturation_upper}), "
            f"hue_delta={self.hue_delta})"
        )
        return repr_str


@PIPELINES.register_module()
class HueSaturationValueJitter(object):
    def __init__(self, hue_ratio=0.5, saturation_ratio=0.5, value_ratio=0.5):
        self.h_ratio = hue_ratio
        self.s_ratio = saturation_ratio
        self.v_ratio = value_ratio

    def __call__(self, results):
        # random gains
        r = (
            np.array([random.uniform(-1.0, 1.0) for _ in range(3)])
            * [self.h_ratio, self.s_ratio, self.v_ratio]
            + 1
        )

        for key in results.get("img_fields", []):
            results[key] = np.ascontiguousarray(results[key])
            img = results[key]

            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=np.int16)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            img_hsv = cv2.merge(
                (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
            ).astype(dtype)
            cv2.cvtColor(
                img_hsv, cv2.COLOR_HSV2BGR, dst=results[key]
            )  # no return needed
        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"hue_ratio={self.h_ratio}, "
            f"saturation_ratio={self.s_ratio}, "
            f"value_ratio={self.v_ratio})"
        )
        return repr_str


@PIPELINES.register_module()
class Corrupt:
    """Corruption augmentation.
    Corruption transforms implemented based on
    `imagecorruptions <https://github.com/bethgelab/imagecorruptions>`_.
    """

    def __init__(self, corruption, severity=1):
        """Initialization for corrupt.

        Args:
            corruption (str): Corruption name.
            severity (int, optional): The severity of corruption. Default: 1.
        """
        self.corruption = corruption
        self.severity = severity

    def __call__(self, results):
        """Call function to corrupt image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images corrupted.
        """
        if corrupt is None:
            raise RuntimeError("imagecorruptions is not installed")
        if "img_fields" in results:
            assert results["img_fields"] == ["img"], "Only single img_fields is allowed"
        results["img"] = corrupt(
            results["img"].astype(np.uint8),
            corruption_name=self.corruption,
            severity=self.severity,
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(corruption={self.corruption}, "
        repr_str += f"severity={self.severity})"
        return repr_str


@PIPELINES.register_module()
class CutOut:
    """CutOut operation.
    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.
    """

    def __init__(
        self, n_holes, cutout_shape=None, cutout_ratio=None, fill_in=(0, 0, 0)
    ):
        """Initialization for cutout.

        Args:
            n_holes (int | tuple[int, int]): Number of regions to be dropped.
                If it is given as a list, number of holes will be randomly
                selected from the closed interval [`n_holes[0]`, `n_holes[1]`].
            cutout_shape (tuple[int, int] | list[tuple[int, int]]): The candidate
                shape of dropped regions. It can be `tuple[int, int]` to use a
                fixed cutout shape, or `list[tuple[int, int]]` to randomly choose
                shape from the list.
            cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
                candidate ratio of dropped regions. It can be `tuple[float, float]`
                to use a fixed ratio or `list[tuple[float, float]]` to randomly
                choose ratio from the list. Please note that `cutout_shape`
                and `cutout_ratio` cannot be both given at the same time.
            fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
                of pixel to fill in the dropped regions. Default: (0, 0, 0).
        """

        assert (cutout_shape is None) ^ (
            cutout_ratio is None
        ), "Either cutout_shape or cutout_ratio should be specified."
        assert isinstance(cutout_shape, (list, tuple)) or isinstance(
            cutout_ratio, (list, tuple)
        )
        if isinstance(n_holes, (list, tuple)):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        self.n_holes = n_holes
        self.fill_in = fill_in
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        if not isinstance(self.candidates, list):
            self.candidates = [self.candidates]

    def __call__(self, results):
        """Call function to drop some regions of image."""
        h, w, c = results["img"].shape
        n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
        for _ in range(n_holes):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            index = np.random.randint(0, len(self.candidates))
            if not self.with_ratio:
                cutout_w, cutout_h = self.candidates[index]
            else:
                cutout_w = int(self.candidates[index][0] * w)
                cutout_h = int(self.candidates[index][1] * h)

            x2 = np.clip(x1 + cutout_w, 0, w)
            y2 = np.clip(y1 + cutout_h, 0, h)
            results["img"][y1:y2, x1:x2, :] = self.fill_in

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(n_holes={self.n_holes}, "
        repr_str += (
            f"cutout_ratio={self.candidates}, "
            if self.with_ratio
            else f"cutout_shape={self.candidates}, "
        )
        repr_str += f"fill_in={self.fill_in})"
        return repr_str


@PIPELINES.register_module()
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [0.1, 5.0]
        self.sigma = sigma

    def __call__(self, results):
        sigma = int(random.uniform(self.sigma[0], self.sigma[1]))
        results["img"] = cv2.GaussianBlur(
            results["img"], (2 * sigma + 1, 2 * sigma + 1), 0
        )
        return results

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(" f"sigma={self.sigma})"
        return repr_str


@PIPELINES.register_module()
class Solarization(object):
    """Apply Solarization to the PIL image."""

    def __init__(self, p):
        self.p = p

    def __call__(self, results):
        if random.random() < self.p:
            pil_img = Image.fromarray(results["img"].astype("uint8")).convert("RGB")
            results["img"] = np.array(ImageOps.solarize(pil_img))

        return results


@PIPELINES.register_module()
class RandomErasing(object):
    """Randomly selects a rectangle region in an image and erase pixels.
    Args:
        erase_prob (float): Probability that image will be randomly erased.
            Default: 0.5
        min_area_ratio (float): Minimum erased area / input image area
            Default: 0.02
        max_area_ratio (float): Maximum erased area / input image area
            Default: 0.4
        aspect_range (sequence | float): Aspect ratio range of erased area.
            if float, it will be converted to (aspect_ratio, 1/aspect_ratio)
            Default: (3/10, 10/3)
        mode (str): Fill method in erased area, can be:
            - const (default): All pixels are assign with the same value.
            - rand: each pixel is assigned with a random value in [0, 255]
        fill_color (sequence | Number): Base color filled in erased area.
            Defaults to (128, 128, 128).
        fill_std (sequence | Number, optional): If set and ``mode`` is 'rand',
            fill erased area with random color from normal distribution
            (mean=fill_color, std=fill_std); If not set, fill erased area with
            random color from uniform distribution (0~255). Defaults to None.
    Note:
        See `Random Erasing Data Augmentation
        <https://arxiv.org/pdf/1708.04896.pdf>`_
        This paper provided 4 modes: RE-R, RE-M, RE-0, RE-255, and use RE-M as
        default. The config of these 4 modes are:
        - RE-R: RandomErasing(mode='rand')
        - RE-M: RandomErasing(mode='const', fill_color=(123.67, 116.3, 103.5))
        - RE-0: RandomErasing(mode='const', fill_color=0)
        - RE-255: RandomErasing(mode='const', fill_color=255)
    """

    def __init__(
        self,
        erase_prob=0.5,
        min_area_ratio=0.02,
        max_area_ratio=0.4,
        aspect_range=(3 / 10, 10 / 3),
        mode="const",
        fill_color=(128, 128, 128),
        fill_std=None,
    ):
        assert isinstance(erase_prob, float) and 0.0 <= erase_prob <= 1.0
        assert isinstance(min_area_ratio, float) and 0.0 <= min_area_ratio <= 1.0
        assert isinstance(max_area_ratio, float) and 0.0 <= max_area_ratio <= 1.0
        assert (
            min_area_ratio <= max_area_ratio
        ), "min_area_ratio should be smaller than max_area_ratio"
        if isinstance(aspect_range, float):
            aspect_range = min(aspect_range, 1 / aspect_range)
            aspect_range = (aspect_range, 1 / aspect_range)
        assert (
            isinstance(aspect_range, Sequence)
            and len(aspect_range) == 2
            and all(isinstance(x, float) for x in aspect_range)
        ), "aspect_range should be a float or Sequence with two float."
        assert all(x > 0 for x in aspect_range), "aspect_range should be positive."
        assert (
            aspect_range[0] <= aspect_range[1]
        ), "In aspect_range (min, max), min should be smaller than max."
        assert mode in ["const", "rand"]
        if isinstance(fill_color, Number):
            fill_color = [fill_color] * 3
        assert (
            isinstance(fill_color, Sequence)
            and len(fill_color) == 3
            and all(isinstance(x, Number) for x in fill_color)
        ), "fill_color should be a float or Sequence with three int."
        if fill_std is not None:
            if isinstance(fill_std, Number):
                fill_std = [fill_std] * 3
            assert (
                isinstance(fill_std, Sequence)
                and len(fill_std) == 3
                and all(isinstance(x, Number) for x in fill_std)
            ), "fill_std should be a float or Sequence with three int."

        self.erase_prob = erase_prob
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.aspect_range = aspect_range
        self.mode = mode
        self.fill_color = fill_color
        self.fill_std = fill_std

    def _fill_pixels(self, img, top, left, h, w):
        if self.mode == "const":
            patch = np.empty((h, w, 3), dtype=np.uint8)
            patch[:, :] = np.array(self.fill_color, dtype=np.uint8)
        elif self.fill_std is None:
            # Uniform distribution
            patch = np.random.uniform(0, 256, (h, w, 3)).astype(np.uint8)
        else:
            # Normal distribution
            patch = np.random.normal(self.fill_color, self.fill_std, (h, w, 3))
            patch = np.clip(patch.astype(np.int32), 0, 255).astype(np.uint8)

        img[top : top + h, left : left + w] = patch
        return img

    def __call__(self, results):
        """
        Args:
            results (dict): Results dict from pipeline

        Returns:
            dict: Results after the transformation.
        """
        for key in results.get("img_fields", ["img"]):
            if np.random.rand() > self.erase_prob:
                continue
            img = results[key]
            img_h, img_w = img.shape[:2]

            # convert to log aspect to ensure equal probability of aspect ratio
            log_aspect_range = np.log(np.array(self.aspect_range, dtype=np.float32))
            aspect_ratio = np.exp(np.random.uniform(*log_aspect_range))
            area = img_h * img_w
            area *= np.random.uniform(self.min_area_ratio, self.max_area_ratio)

            h = min(int(round(np.sqrt(area * aspect_ratio))), img_h)
            w = min(int(round(np.sqrt(area / aspect_ratio))), img_w)
            top = np.random.randint(0, img_h - h) if img_h > h else 0
            left = np.random.randint(0, img_w - w) if img_w > w else 0
            img = self._fill_pixels(img, top, left, h, w)

            results[key] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(erase_prob={self.erase_prob}, "
        repr_str += f"min_area_ratio={self.min_area_ratio}, "
        repr_str += f"max_area_ratio={self.max_area_ratio}, "
        repr_str += f"aspect_range={self.aspect_range}, "
        repr_str += f"mode={self.mode}, "
        repr_str += f"fill_color={self.fill_color}, "
        repr_str += f"fill_std={self.fill_std})"
        return repr_str
