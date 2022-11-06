import numpy as np

from mtl.utils.misc_util import is_list_of
from mtl.utils.geometric_util import imrescale, imresize
from ..data_wrapper import PIPELINES


@PIPELINES.register_module()
class SegResize(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
            Default:None.
        multiscale_mode (str): Either "range" or "value".
            Default: 'range'
        ratio_range (tuple[float]): (min_ratio, max_ratio).
            Default: None
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: True
    """

    def __init__(
        self, img_scale=None, multiscale_mode="range", ratio_range=None, keep_ratio=True
    ):
        if img_scale is None:
            self.img_scale = None
        else:
            if is_list_of(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert is_list_of(self.img_scale, list)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ["value", "range"]

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert is_list_of(img_scales, list)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long), max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short), max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, list) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """
        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results["img"].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h), self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range
                )
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == "range":
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == "value":
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results["scale"] = scale
        results["scale_idx"] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            img, scale_factor = imrescale(
                results["img"], results["scale"], return_scale=True
            )
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results["img"].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = imresize(
                results["img"], results["scale"], return_scale=True
            )
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        results["img"] = img
        results["img_shape"] = img.shape
        results["pad_shape"] = img.shape  # in case that there is no padding
        results["scale_factor"] = scale_factor
        results["keep_ratio"] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get("seg_fields", []):
            if self.keep_ratio:
                gt_seg = imrescale(
                    results[key], results["scale"], interpolation="nearest"
                )
            else:
                gt_seg = imresize(
                    results[key], results["scale"], interpolation="nearest"
                )
            results[key] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if "scale" not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(img_scale={self.img_scale}, "
            f"multiscale_mode={self.multiscale_mode}, "
            f"ratio_range={self.ratio_range}, "
            f"keep_ratio={self.keep_ratio})"
        )
        return repr_str


@PIPELINES.register_module()
class SegRandomCrop:
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1.0, ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        img = results["img"]
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.0:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results["gt_semantic_seg"], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results["img"] = img
        results["img_shape"] = img_shape

        # crop semantic seg
        for key in results.get("seg_fields", []):
            results[key] = self.crop(results[key], crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(crop_size={self.crop_size})"


@PIPELINES.register_module()
class SegRescale:
    """Rescale semantic segmentation maps."""

    def __init__(self, scale_factor=1, backend="cv2"):
        """Initialization of rescale for segmentation

        Args:
            scale_factor (float): The scale factor of the final output.
            backend (str): Image rescale backend, choices are 'cv2' and 'pillow'.
                These two backends generates slightly different results. Defaults
                to 'cv2'.
        """
        self.scale_factor = scale_factor
        self.backend = backend

    def __call__(self, results):
        """Call function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """
        for key in results.get("seg_fields", []):
            if self.scale_factor != 1:
                results[key] = imrescale(
                    results[key],
                    self.scale_factor,
                    interpolation="nearest",
                    backend=self.backend,
                )
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(scale_factor={self.scale_factor})"


@PIPELINES.register_module()
class GenerateHeatMap:
    """Generate heat maps for segmentation."""

    def __init__(self, sigma=3):
        """Initialization of rescale for segmentation

        Args:
            sigma (float): The scale factor of the final output.
        """
        self.sigma = sigma

    def __call__(self, results):
        """Call function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """
        # 3-sigma rule
        tmp_size = self.sigma * 3
        num_class = 2

        H = results["img_shape"][0]
        W = results["img_shape"][1]

        results["gt_semantic_seg"] = np.zeros((num_class, H, W), dtype=np.float32)
        results["seg_fields"].append("gt_semantic_seg")

        if results["ann"]["concat_type"] == 0:  # split with horizontal line
            feat_stride = results["ori_shape"][0] / H
            for concat_line in results["ann"]["concat_lines"]:
                split_y = int(concat_line / feat_stride + 0.5)
                if split_y < 0 or split_y >= H:
                    continue

                min_y = int(split_y - tmp_size)
                max_y = int(split_y + tmp_size + 1)

                size = 2 * tmp_size + 1
                y = np.arange(0, size, 1, np.float32)
                y0 = size // 2
                # The gaussian is not normalized,
                # we want the center value to equal 1
                g = np.exp(-((y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_y = max(0, -min_y), min(max_y, H) - min_y
                # Image range
                img_y = max(0, min_y), min(max_y, H)

                for i in range(W):
                    results["gt_semantic_seg"][0, img_y[0] : img_y[1], i] += g[
                        g_y[0] : g_y[1]
                    ]
        else:  # split with vertical line
            feat_stride = results["ori_shape"][1] / W
            for concat_line in results["ann"]["concat_lines"]:
                split_x = int(concat_line / feat_stride + 0.5)
                if split_x < 0 or split_x >= W:
                    continue

                min_x = int(split_x - tmp_size)
                max_x = int(split_x + tmp_size + 1)

                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                x0 = size // 2
                # The gaussian is not normalized,
                # we want the center value to equal 1
                g = np.exp(-((x - x0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -min_x), min(max_x, W) - min_x
                # Image range
                img_x = max(0, min_x), min(max_x, W)

                for i in range(H):
                    results["gt_semantic_seg"][1, i, img_x[0] : img_x[1]] += g[
                        g_x[0] : g_x[1]
                    ]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(sigma={self.sigma})"
        return repr_str


@PIPELINES.register_module()
class GenerateNodeHeatMap:
    """Generate heat maps for segmentation."""

    def __init__(self, sigma=5):
        """Initialization of rescale for segmentation

        Args:
            sigma (float): The scale factor of the final output.
        """
        self.sigma = sigma

    def __call__(self, results):
        """Call function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """
        # 3-sigma rule
        tmp_size = self.sigma * 3

        num_class = 1

        H = results["img_shape"][0]
        W = results["img_shape"][1]

        results["gt_semantic_seg"] = np.zeros((num_class, H, W), dtype=np.float32)
        results["seg_fields"].append("gt_semantic_seg")

        feat_stride = [results["ori_shape"][0] / H, results["ori_shape"][1] / W]

        for key_node in results["ann"]["key_nodes"]:
            kd_x = int(key_node[0] / feat_stride[1] + 0.5)
            kd_y = int(key_node[1] / feat_stride[0] + 0.5)

            if kd_x < 0 or kd_x >= W:
                continue
            if kd_y < 0 or kd_y >= H:
                continue

            min_x = int(kd_x - tmp_size)
            max_x = int(kd_x + tmp_size + 1)

            min_y = int(kd_y - tmp_size)
            max_y = int(kd_y + tmp_size + 1)

            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, None]
            x0 = y0 = size // 2

            # The gaussian is not normalized,
            # we want the center value to equal 1
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -min_x), min(max_x, W) - min_x
            g_y = max(0, -min_y), min(max_y, H) - min_y
            # Image range
            img_x = max(0, min_x), min(max_x, W)
            img_y = max(0, min_y), min(max_y, H)

            results["gt_semantic_seg"][
                0, img_y[0] : img_y[1], img_x[0] : img_x[1]
            ] += g[g_y[0] : g_y[1], g_x[0] : g_x[1]]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(sigma={self.sigma})"
        return repr_str
