from collections.abc import Sequence
import copy
import numpy as np
import torch
import math
import random
from einops import rearrange

from mtl.utils.misc_util import is_str
from mtl.utils.geometric_util import imcrop
from ..data_wrapper import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.
    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@PIPELINES.register_module()
class ToTensor:
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert data in results to :obj:`torch.Tensor`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted
                to :obj:`torch.Tensor`.
        """
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):

        return self.__class__.__name__ + f"(keys={self.keys})"


@PIPELINES.register_module()
class ImageToTensor:
    """Convert image to :obj:`torch.Tensor` by given keys.
    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """
        if "img1" in self.keys:
            results["imgs"] = []
            for key in self.keys:
                img = results[key]
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                results["imgs"].append(to_tensor(img.transpose(2, 0, 1)))
        else:
            for key in self.keys:
                img = results[key]
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                results[key] = to_tensor(img.transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(keys={self.keys})"


@PIPELINES.register_module()
class Transpose:
    """Transpose some results by given keys.
    Args:
        keys (Sequence[str]): Keys of results to be transposed.
        order (Sequence[int]): Order of transpose.
    """

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        """Call function to transpose the channel order of data in results.
        Args:
            results (dict): Result dict contains the data to transpose.

        Returns:
            dict: The result dict contains the data transposed to \
                ``self.order``.
        """
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(keys={self.keys}, order={self.order})"


@PIPELINES.register_module()
class DefaultFormatBundle:
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor
    - proposals: (1)to tensor
    - gt_bboxes: (1)to tensor
    - gt_bboxes_ignore: (1)to tensor
    - gt_labels: (1)to tensor
    - gt_masks: (1)to tensor
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """
        if "img" in results:
            img = results["img"]
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results["img"] = to_tensor(img)

        if "img1" in results:
            results["imgs"] = []
            for key in ["img1", "img2", "img3", "img4", "img5", "img6"]:
                if key in results:
                    img_slice = results[key]
                    if len(img_slice.shape) < 3:
                        img_slice = np.expand_dims(img_slice, -1)
                    img_slice = np.ascontiguousarray(img_slice.transpose(2, 0, 1))
                results["imgs"].append(to_tensor(img_slice))

        # gt_label is for classfication
        for key in [
            "proposals",
            "gt_bboxes",
            "gt_bboxes_ignore",
            "gt_labels",
            "gt_label",
            "gt_prob",
            "patches",
            "patch_labels",
            "patch_mask",
            "txt_id",
            "txt_mask",
            "short_v",
        ]:
            if key not in results:
                continue
            results[key] = to_tensor(results[key])

        for key in ["img_k", "img_s", "patch", "patch_k"]:
            if key not in results:
                continue
            key_img = results[key]
            if isinstance(key_img, list):
                key_img_ref = []
                for sub_img in key_img:
                    if len(sub_img.shape) < 3:
                        sub_img = np.expand_dims(sub_img, -1)
                    sub_img = np.ascontiguousarray(sub_img.transpose(2, 0, 1))
                    key_img_ref.append(to_tensor(sub_img))
                results[key] = key_img_ref
            else:
                if len(key_img.shape) < 3:
                    key_img = np.expand_dims(key_img, -1)
                key_img = np.ascontiguousarray(key_img.transpose(2, 0, 1))
                results[key] = to_tensor(key_img)

        if "gt_masks" in results:
            results["gt_masks"] = results["gt_masks"]
        if "gt_semantic_seg" in results:
            if len(results["gt_semantic_seg"].shape) == 2:
                results["gt_semantic_seg"] = to_tensor(
                    np.ascontiguousarray(results["gt_semantic_seg"][None, ...])
                )
            else:
                results["gt_semantic_seg"] = to_tensor(
                    np.ascontiguousarray(results["gt_semantic_seg"])
                )
        if "target" in results:  # for pose
            results["target"] = to_tensor(results["target"])
        if "target_weight" in results:  # for pose
            results["target_weight"] = to_tensor(results["target_weight"])
        return results

    def _add_default_meta_keys(self, results):
        """Add default meta keys.
        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results["img"]
        results.setdefault("pad_shape", img.shape)
        results.setdefault("scale_factor", 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            "img_norm_cfg",
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False,
            ),
        )
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class Collect:
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:
        - "img_shape": shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - "scale_factor": a float indicating the preprocessing scale
        - "flip": a boolean indicating if image flip transform was used
        - "file_name": path to the image file
        - "ori_shape": original shape of the image as a tuple (h, w, c)
        - "pad_shape": image shape after padding
        - "img_norm_cfg": a dict of normalization information:
            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be  collected in ``data[img_metas]``.
            Default: ``('file_name', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')``
    """

    def __init__(
        self,
        keys,
        meta_keys=(
            "file_name",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "img_norm_cfg",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in``self.keys``
                - ``img_metas``
        """
        data = {}
        img_meta = {}
        for key, value in results.items():
            if key in self.meta_keys:
                img_meta[key] = value
        data["img_metas"] = img_meta
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(keys={self.keys}, meta_keys={self.meta_keys})"
        )


@PIPELINES.register_module()
class ClsCollect:
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img" and "gt_label".
    """

    def __init__(
        self,
        keys,
        meta_keys=(
            "file_name",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "img_norm_cfg",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_meta = {}
        for key in self.keys:
            data[key] = results[key]

        for key, value in results.items():
            if key in self.meta_keys:
                img_meta[key] = value
        data["img_metas"] = img_meta
        return data

    def __repr__(self):
        return self.__class__.__name__ + f"(keys={self.keys})"


@PIPELINES.register_module()
class WrapFieldsToLists:
    """Wrap fields of the data dictionary into lists for evaluation.
    This class can be used as a last step of a test or validation
    pipeline for single image evaluation or inference.
    Example:
        >>> test_pipeline = [
        >>>    dict(type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
        >>>    dict(type='Pad', size_divisor=32),
        >>>    dict(type='ImageToTensor', keys=['img']),
        >>>    dict(type='Collect', keys=['img']),
        >>>    dict(type='WrapIntoLists')
        >>> ]
    """

    def __call__(self, results):
        """Call function to wrap fields into lists.
        Args:
            results (dict): Result dict contains the data to wrap.

        Returns:
            dict: The result dict where value of ``self.keys`` are wrapped \
                into list.
        """
        # Wrap dict fields into lists
        for key, val in results.items():
            results[key] = [val]
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}()"


@PIPELINES.register_module()
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, results):
        ori_results = copy.deepcopy(results)
        results = self.base_transform(results)
        results_k = self.base_transform(ori_results)
        results["img_k"] = results_k["img"]
        return results


@PIPELINES.register_module()
class SplitCrop:
    """Split image into patches and then using random crop to given size."""

    def __init__(self, num_row_splits=3, num_col_splits=3, crop_size=64):
        self.num_row_splits = num_row_splits
        self.num_col_splits = num_col_splits
        self.crop_size = crop_size

    def __call__(self, results):
        h, w = results["img"].shape[:2]
        row_h = h // self.num_row_splits
        col_w = w // self.num_col_splits
        margin_h = row_h - self.crop_size
        margin_w = col_w - self.crop_size
        boxes = []
        for i in range(self.num_row_splits):
            for j in range(self.num_col_splits):
                rand_w = np.random.randint(0, margin_w + 1)
                rand_h = np.random.randint(0, margin_h + 1)
                boxes.append(
                    [
                        j * col_w + rand_w,
                        i * row_h + rand_h,
                        j * col_w + rand_w + self.crop_size - 1,
                        i * row_h + rand_h + self.crop_size - 1,
                    ]
                )
        results["img_s"] = imcrop(results["img"], np.array(boxes))

        return results


@PIPELINES.register_module()
class MaskedPatching:
    """Split image into patches and then using random crop to given size."""

    def __init__(self, patch_size=16, normlize_target=True):
        self.patch_size = patch_size
        self.normlize_target = normlize_target

    def __call__(self, results):
        if self.normlize_target:
            images_squeeze = rearrange(
                results["img"],
                "(h p1) (w p2) c -> (h w) (p1 p2) c",
                p1=self.patch_size,
                p2=self.patch_size,
            )
            images_norm = (
                images_squeeze - images_squeeze.mean(axis=-2, keepdims=True)
            ) / (np.sqrt(images_squeeze.var(axis=-2, keepdims=True)) + 1e-6)

            images_patch = rearrange(images_norm, "n p c -> n (p c)")
        else:
            images_patch = rearrange(
                results["img"],
                "(h p1) (w p2) c -> (h w) (p1 p2 c)",
                p1=self.patch_size,
                p2=self.patch_size,
            )
        C = images_patch.shape[-1]

        results["patch_labels"] = images_patch[results["patch_mask"]].reshape(-1, C)

        return results


@PIPELINES.register_module()
class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = (
            num_masking_patches if max_num_patches is None else max_num_patches
        )

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, results):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta
        results["patch_mask"] = mask

        return results


@PIPELINES.register_module()
class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self, results):
        mask = np.hstack(
            [
                np.zeros(self.num_patches - self.num_mask, dtype=np.bool),
                np.ones(self.num_mask, dtype=np.bool),
            ]
        )
        np.random.shuffle(mask)
        results["patch_mask"] = mask

        return results


@PIPELINES.register_module()
class MaskImgGenerator:
    def __init__(
        self, input_size=224, mask_patch_size=32, model_patch_size=4, mask_ratio=0.5
    ):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self, results):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=np.float32)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        results["patch_mask"] = mask

        return results
