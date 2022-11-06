import copy
import cv2
import inspect
import random
import math
from numbers import Number
from typing import Sequence
import numpy as np

from mtl.utils.geometric_util import imshear, imrotate, imtranslate
from mtl.utils.photometric_util import (
    adjust_brightness,
    adjust_sharpness,
    adjust_color,
    adjust_contrast,
    imequalize,
    auto_contrast,
    iminvert,
    posterize,
    solarize,
)
from mtl.utils.reg_util import build_module_from_dict
from ..data_wrapper import PIPELINES
from .compose import Compose


# Default hyperparameters for all Ops
_MAX_LEVEL = 10
_HPARAMS_DEFAULT = dict(pad_val=128)


def level_to_value(level, max_value):
    """Map from level to values based on max_value."""
    return (level / _MAX_LEVEL) * max_value


def enhance_level_to_value(level, a=1.8, b=0.1):
    """Map from level to values."""
    return (level / _MAX_LEVEL) * a + b


def random_negative(value, random_negative_prob):
    """Randomly negate value based on random_negative_prob."""
    return -value if np.random.rand() < random_negative_prob else value


def merge_hparams(policy: dict, hparams: dict):
    """Merge hyperparameters into policy config.

    Only merge partial hyperparameters required of the policy.
    Args:
        policy (dict): Original policy config dict.
        hparams (dict): Hyperparameters need to be merged.
    Returns:
        dict: Policy config dict after adding ``hparams``.
    """
    op = PIPELINES.get(policy["type"])
    assert op is not None, f'Invalid policy type "{policy["type"]}".'
    for key, value in hparams.items():
        if policy.get(key, None) is not None:
            continue
        if key in inspect.getfullargspec(op.__init__).args:
            policy[key] = value
    return policy


def bbox2fields():
    """The key correspondence from bboxes to labels, masks and
    segmentations."""
    bbox2label = {"gt_bboxes": "gt_labels", "gt_bboxes_ignore": "gt_labels_ignore"}
    bbox2mask = {"gt_bboxes": "gt_masks", "gt_bboxes_ignore": "gt_masks_ignore"}
    bbox2seg = {"gt_bboxes": "gt_semantic_seg"}
    return bbox2label, bbox2mask, bbox2seg


@PIPELINES.register_module()
class RandomApply(object):
    """Apply randomly a list of transformations with a given probability.

    Args:
        policies (torch.nn.Module): list of transformation
        p (float): probability
    """

    def __init__(self, policies, p=0.5):
        super().__init__()
        assert isinstance(policies, list), "Each specific augmentation must be a list."

        self.policies = copy.deepcopy(policies)
        self.transforms = []
        for policy in self.policies:
            if isinstance(policy, dict):
                transform = build_module_from_dict(policy, PIPELINES)
                self.transforms.append(transform)
        self.p = p

    def __call__(self, results):
        if self.p < np.random.rand(1):
            return results
        for t in self.transforms:
            results = t(results)
        return results

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "\n    p={}".format(self.p)
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


@PIPELINES.register_module()
class RandomApplyTwice(object):
    """Apply randomly a list of transformations with a given probability.

    Args:
        policies (torch.nn.Module): list of transformation
        p (float): probability
    """

    def __init__(self, policies, p=0.5):
        super().__init__()
        assert isinstance(policies, list), "Each specific augmentation must be a list."

        self.policies = copy.deepcopy(policies)
        self.transforms = []
        for policy in self.policies:
            if isinstance(policy, dict):
                transform = build_module_from_dict(policy, PIPELINES)
                self.transforms.append(transform)
        self.p = p

    def __call__(self, results):
        if self.p < np.random.rand(1):
            return results
        for t in self.transforms:
            results = t(results)
        return results

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "\n    p={}".format(self.p)
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


@PIPELINES.register_module()
class MultiApply(object):
    """MultiApplyDINO.

    Args:
        policies (list[list[dict]]): The policies of auto augmentation. Each
            policy in ``policies`` is a specific augmentation policy, and is
            composed by several augmentations (dict). When AutoAugment is
            called, a random policy in ``policies`` will be selected to
            augment images.

    Examples:
        >>> policies = [
        >>>     [
        >>>         dict(
        >>>             type='Shear',
        >>>             prob=0.4,
        >>>             level=0)
        >>>     ],
        >>>     [
        >>>         dict(
        >>>             type='Rotate',
        >>>             prob=0.6,
        >>>             level=10)
        >>>     ]
        >>> ]
        >>> augmentation = MultiApply(policies)
    """

    def __init__(self, policies):
        assert (
            isinstance(policies, list) and len(policies) == 2
        ), "Policies must be a non-empty list."
        for policy in policies:
            assert (
                isinstance(policy, list) and len(policy) > 0
            ), "Each policy in policies must be a non-empty list."
            for transform in policy:
                assert isinstance(
                    transform, dict
                ), "Each specific augmentation must be a dict."

        self.policies = copy.deepcopy(policies)
        self.transforms = [Compose(policy) for policy in self.policies]

    def __call__(self, results):
        bk_results = copy.deepcopy(results)
        results = self.transforms[0](results)
        ori_results = copy.deepcopy(bk_results)
        results_k = self.transforms[1](ori_results)
        results["img_k"] = results_k["img"]

        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(policies={self.policies})"


@PIPELINES.register_module()
class MultiApplyDINO(object):
    """MultiApplyDINO."""

    def __init__(self, policies, local_crops_number=0):
        """Initialize for dino-style multi-apply

        Args:
            policies (list[list[dict]]): The policies of auto augmentation. Each
                policy in ``policies`` is a specific augmentation policy, and is
                composed by several augmentations (dict). When AutoAugment is
                called, a random policy in ``policies`` will be selected to
                augment images.
        """
        assert (
            isinstance(policies, list) and len(policies) >= 2
        ), "Policies must be a non-empty list."
        for policy in policies:
            assert (
                isinstance(policy, list) and len(policy) > 0
            ), "Each policy in policies must be a non-empty list."
            for transform in policy:
                assert isinstance(
                    transform, dict
                ), "Each specific augmentation must be a dict."

        self.policies = copy.deepcopy(policies)
        self.transforms = [Compose(policy) for policy in self.policies]
        self.local_crops_number = local_crops_number

    def __call__(self, results):
        bk_results = copy.deepcopy(results)
        results = self.transforms[0](results)
        ori_results = copy.deepcopy(bk_results)
        results_k = self.transforms[1](ori_results)
        results["img_k"] = results_k["img"]

        img_crops = []
        for _ in range(self.local_crops_number):
            ori_results = copy.deepcopy(bk_results)
            img_crops.append(self.transforms[2](ori_results)["img"])

        results["img_s"] = img_crops
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(policies={self.policies})"


@PIPELINES.register_module()
class MultiApplyDetCo(object):
    """MultiApplyDetCo."""

    def __init__(self, policies):
        """Initialize for detco-style multi-apply

        Args:
            policies (list[list[dict]]): The policies of auto augmentation. Each
                policy in ``policies`` is a specific augmentation policy, and is
                composed by several augmentations (dict). When AutoAugment is
                called, a random policy in ``policies`` will be selected to
                augment images.
        """
        assert (
            isinstance(policies, list) and len(policies) == 4
        ), "Policies must be a non-empty list."
        for policy in policies:
            assert (
                isinstance(policy, list) and len(policy) > 0
            ), "Each policy in policies must be a non-empty list."
            for transform in policy:
                assert isinstance(
                    transform, dict
                ), "Each specific augmentation must be a dict."

        self.policies = copy.deepcopy(policies)
        self.transforms = [Compose(policy) for policy in self.policies]

    def __call__(self, results):
        bk_results = copy.deepcopy(results)
        results = self.transforms[0](results)
        ori_results = copy.deepcopy(bk_results)
        results_k = self.transforms[1](ori_results)
        results["img_k"] = results_k["img"]

        ori_results = copy.deepcopy(bk_results)
        img_crops = self.transforms[2](ori_results)["img_s"]
        results["patch"] = img_crops

        ori_results = copy.deepcopy(bk_results)
        img_crops = self.transforms[2](ori_results)["img_s"]
        results["patch_k"] = img_crops
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(policies={self.policies})"


@PIPELINES.register_module()
class AutoAugment(object):
    """Auto augmentation.

    This data augmentation is proposed in `Learning Data Augmentation
    Strategies for Object Detection <https://arxiv.org/pdf/1906.11172>`_.

    Args:
        policies (list[list[dict]]): The policies of auto augmentation. Each
            policy in ``policies`` is a specific augmentation policy, and is
            composed by several augmentations (dict). When AutoAugment is
            called, a random policy in ``policies`` will be selected to
            augment images.

    Examples:
        >>> policies = [
        >>>     [
        >>>         dict(
        >>>             type='Shear',
        >>>             prob=0.4,
        >>>             level=0)
        >>>     ],
        >>>     [
        >>>         dict(
        >>>             type='Rotate',
        >>>             prob=0.6,
        >>>             level=10)
        >>>     ]
        >>> ]
        >>> augmentation = AutoAugment(policies)
        >>> img = np.ones((100, 100, 3))
        >>> gt_bboxes = np.ones((10, 4))
        >>> results = dict(img=img, gt_bboxes=gt_bboxes)
        >>> results = augmentation(results)
    """

    def __init__(self, policies):
        assert (
            isinstance(policies, list) and len(policies) > 0
        ), "Policies must be a non-empty list."
        for policy in policies:
            assert (
                isinstance(policy, list) and len(policy) > 0
            ), "Each policy in policies must be a non-empty list."
            for augment in policy:
                assert isinstance(
                    augment, dict
                ), "Each specific augmentation must be a dict."

        self.policies = copy.deepcopy(policies)
        self.transforms = [Compose(policy) for policy in self.policies]

    def __call__(self, results):
        transform = np.random.choice(self.transforms)
        if ("img" in results) and ("img_shape" not in results):
            results["img_shape"] = results["img"].shape
        return transform(results)

    def __repr__(self):
        return f"{self.__class__.__name__}(policies={self.policies})"


@PIPELINES.register_module()
class RandAugment(object):
    r"""Random augmentation.
    This data augmentation is proposed in `RandAugment: Practical automated
    data augmentation with a reduced search space
    <https://arxiv.org/abs/1909.13719>`_.
    Args:
        policies (list[dict]): The policies of random augmentation. Each
            policy in ``policies`` is one specific augmentation policy (dict).
            The policy shall at least have key `type`, indicating the type of
            augmentation. For those which have magnitude, (given to the fact
            they are named differently in different augmentation, )
            `magnitude_key` and `magnitude_range` shall be the magnitude
            argument (str) and the range of magnitude (tuple in the format of
            (val1, val2)), respectively. Note that val1 is not necessarily
            less than val2.
        num_policies (int): Number of policies to select from policies each
            time.
        magnitude_level (int | float): Magnitude level for all the augmentation
            selected.
        total_level (int | float): Total level for the magnitude. Defaults to
            30.
        magnitude_std (Number | str): Deviation of magnitude noise applied.
            - If positive number, magnitude is sampled from normal distribution
              (mean=magnitude, std=magnitude_std).
            - If 0 or negative number, magnitude remains unchanged.
            - If str "inf", magnitude is sampled from uniform distribution
              (range=[min, magnitude]).
        hparams (dict): Configs of hyperparameters. Hyperparameters will be
            used in policies that require these arguments if these arguments
            are not set in policy dicts. Defaults to use _HPARAMS_DEFAULT.
    Note:
        `magnitude_std` will introduce some randomness to policy, modified by
        https://github.com/rwightman/pytorch-image-models.
        When magnitude_std=0, we calculate the magnitude as follows:
        .. math::
            \text{magnitude} = \frac{\text{magnitude\_level}}
            {\text{total\_level}} \times (\text{val2} - \text{val1})
            + \text{val1}
    """

    def __init__(
        self,
        policies,
        num_policies,
        magnitude_level,
        magnitude_std=0.0,
        total_level=30,
        hparams=_HPARAMS_DEFAULT,
    ):
        assert isinstance(num_policies, int), (
            "Number of policies must be "
            f"of int type, got {type(num_policies)} instead."
        )
        assert isinstance(magnitude_level, (int, float)), (
            "Magnitude level must be of int or float type, "
            f"got {type(magnitude_level)} instead."
        )
        assert isinstance(total_level, (int, float)), (
            "Total level must be "
            f"of int or float type, got {type(total_level)} instead."
        )
        assert (
            isinstance(policies, list) and len(policies) > 0
        ), "Policies must be a non-empty list."

        assert isinstance(magnitude_std, (Number, str)), (
            "Magnitude std must be of number or str type, "
            f"got {type(magnitude_std)} instead."
        )
        if isinstance(magnitude_std, str):
            assert magnitude_std == "inf", (
                'Magnitude std must be of number or "inf", '
                f'got "{magnitude_std}" instead.'
            )

        assert num_policies > 0, "num_policies must be greater than 0."
        assert magnitude_level >= 0, "magnitude_level must be no less than 0."
        assert total_level > 0, "total_level must be greater than 0."

        self.num_policies = num_policies
        self.magnitude_level = magnitude_level
        self.magnitude_std = magnitude_std
        self.total_level = total_level
        if isinstance(hparams, list):
            hparams = hparams[0]
        self.hparams = hparams
        policies = copy.deepcopy(policies)
        self._check_policies(policies)
        self.policies = [merge_hparams(policy, hparams) for policy in policies]

    def _check_policies(self, policies):
        for policy in policies:
            assert (
                isinstance(policy, dict) and "type" in policy
            ), 'Each policy must be a dict with key "type".'
            type_name = policy["type"]

            magnitude_key = policy.get("magnitude_key", None)
            if magnitude_key is not None:
                assert (
                    "magnitude_range" in policy
                ), f"RandAugment policy {type_name} needs `magnitude_range`."
                magnitude_range = policy["magnitude_range"]
                assert (
                    isinstance(magnitude_range, Sequence) and len(magnitude_range) == 2
                ), (
                    f"`magnitude_range` of RandAugment policy {type_name} "
                    f"should be a Sequence with two numbers."
                )

    def _process_policies(self, policies):
        processed_policies = []
        for policy in policies:
            processed_policy = copy.deepcopy(policy)
            magnitude_key = processed_policy.pop("magnitude_key", None)
            if magnitude_key is not None:
                magnitude = self.magnitude_level
                # if magnitude_std is positive number or 'inf', move
                # magnitude_value randomly.
                if self.magnitude_std == "inf":
                    magnitude = random.uniform(0, magnitude)
                elif self.magnitude_std > 0:
                    magnitude = random.gauss(magnitude, self.magnitude_std)
                    magnitude = min(self.total_level, max(0, magnitude))

                val1, val2 = processed_policy.pop("magnitude_range")
                magnitude = (magnitude / self.total_level) * (val2 - val1) + val1

                processed_policy.update({magnitude_key: magnitude})
            processed_policies.append(processed_policy)
        return processed_policies

    def __call__(self, results):
        if self.num_policies == 0:
            return results
        sub_policy = random.choices(self.policies, k=self.num_policies)
        sub_policy = self._process_policies(sub_policy)
        sub_policy = Compose(sub_policy)
        return sub_policy(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(policies={self.policies}, "
        repr_str += f"num_policies={self.num_policies}, "
        repr_str += f"magnitude_level={self.magnitude_level}, "
        repr_str += f"total_level={self.total_level})"
        return repr_str


@PIPELINES.register_module()
class Shear(object):
    """Apply Shear Transformation to image (and its corresponding bbox, mask,
    segmentation).

    Args:
        level (int | float): The level should be in range [0,_MAX_LEVEL].
        img_fill_val (int | float | tuple): The filled values for image border.
            If float, the same fill value will be used for all the three
            channels of image. If tuple, the should be 3 elements.
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Default 255.
        prob (float): The probability for performing Shear and should be in
            range [0, 1].
        direction (str): The direction for shear, either "horizontal"
            or "vertical".
        max_shear_magnitude (float): The maximum magnitude for Shear
            transformation.
        random_negative_prob (float): The probability that turns the
                offset negative. Should be in range [0,1]
        interpolation (str): Same as in :func:`imshear`.
    """

    def __init__(
        self,
        level,
        img_fill_val=128,
        seg_ignore_label=255,
        prob=0.5,
        direction="horizontal",
        max_shear_magnitude=0.3,
        random_negative_prob=0.5,
        interpolation="bilinear",
    ):
        assert isinstance(level, (int, float)), (
            "The level must be type " f"int or float, got {type(level)}."
        )
        assert 0 <= level <= _MAX_LEVEL, (
            "The level should be in range " f"[0,{_MAX_LEVEL}], got {level}."
        )

        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, (
                "img_fill_val as tuple must "
                f"have 3 elements. got {len(img_fill_val)}."
            )
            img_fill_val = tuple([float(val) for val in img_fill_val])
        else:
            raise ValueError("img_fill_val must be float or tuple with 3 elements.")

        assert np.all([0 <= val <= 255 for val in img_fill_val]), (
            "all "
            "elements of img_fill_val should between range [0,255]."
            f"got {img_fill_val}."
        )
        assert 0 <= prob <= 1.0, (
            "The probability of shear should be in " f"range [0,1]. got {prob}."
        )
        assert direction in ("horizontal", "vertical"), (
            "direction must "
            f'in be either "horizontal" or "vertical". got {direction}.'
        )
        assert isinstance(max_shear_magnitude, float), (
            "max_shear_magnitude "
            f"should be type float. got {type(max_shear_magnitude)}."
        )
        assert 0.0 <= max_shear_magnitude <= 1.0, (
            "Defaultly "
            "max_shear_magnitude should be in range [0,1]. "
            f"got {max_shear_magnitude}."
        )

        self.level = level
        self.magnitude = level_to_value(level, max_shear_magnitude)
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob
        self.direction = direction
        self.max_shear_magnitude = max_shear_magnitude
        self.random_negative_prob = random_negative_prob
        self.interpolation = interpolation

    def _shear_img(
        self, results, magnitude, direction="horizontal", interpolation="bilinear"
    ):
        """Shear the image.

        Args:
            results (dict): Result dict from loading pipeline.
            magnitude (int | float): The magnitude used for shear.
            direction (str): The direction for shear, either "horizontal"
                or "vertical".
            interpolation (str): Same as in :func:`imshear`.
        """
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            img_sheared = imshear(
                img,
                magnitude,
                direction,
                border_value=self.img_fill_val,
                interpolation=interpolation,
            )
            results[key] = img_sheared.astype(img.dtype)

    def _shear_bboxes(self, results, magnitude):
        """Shear the bboxes."""
        h, w, _ = results["img_shape"]
        if self.direction == "horizontal":
            shear_matrix = np.stack([[1, magnitude], [0, 1]]).astype(
                np.float32
            )  # [2, 2]
        else:
            shear_matrix = np.stack([[1, 0], [magnitude, 1]]).astype(np.float32)
        for key in results.get("bbox_fields", []):
            min_x, min_y, max_x, max_y = np.split(
                results[key], results[key].shape[-1], axis=-1
            )
            coordinates = np.stack(
                [[min_x, min_y], [max_x, min_y], [min_x, max_y], [max_x, max_y]]
            )  # [4, 2, nb_box, 1]
            coordinates = (
                coordinates[..., 0].transpose((2, 1, 0)).astype(np.float32)
            )  # [nb_box, 2, 4]
            new_coords = np.matmul(
                shear_matrix[None, :, :], coordinates
            )  # [nb_box, 2, 4]
            min_x = np.min(new_coords[:, 0, :], axis=-1)
            min_y = np.min(new_coords[:, 1, :], axis=-1)
            max_x = np.max(new_coords[:, 0, :], axis=-1)
            max_y = np.max(new_coords[:, 1, :], axis=-1)
            min_x = np.clip(min_x, a_min=0, a_max=w)
            min_y = np.clip(min_y, a_min=0, a_max=h)
            max_x = np.clip(max_x, a_min=min_x, a_max=w)
            max_y = np.clip(max_y, a_min=min_y, a_max=h)
            results[key] = np.stack([min_x, min_y, max_x, max_y], axis=-1).astype(
                results[key].dtype
            )

    def _shear_masks(
        self,
        results,
        magnitude,
        direction="horizontal",
        fill_val=0,
        interpolation="bilinear",
    ):
        """Shear the masks."""
        h, w, _ = results["img_shape"]
        for key in results.get("mask_fields", []):
            masks = results[key]
            results[key] = masks.shear(
                (h, w),
                magnitude,
                direction,
                border_value=fill_val,
                interpolation=interpolation,
            )

    def _shear_seg(
        self,
        results,
        magnitude,
        direction="horizontal",
        fill_val=255,
        interpolation="bilinear",
    ):
        """Shear the segmentation maps."""
        for key in results.get("seg_fields", []):
            seg = results[key]
            results[key] = imshear(
                seg,
                magnitude,
                direction,
                border_value=fill_val,
                interpolation=interpolation,
            ).astype(seg.dtype)

    def _filter_invalid(self, results, min_bbox_size=0):
        """Filter bboxes and corresponding masks too small after shear
        augmentation."""
        bbox2label, bbox2mask, _ = bbox2fields()
        for key in results.get("bbox_fields", []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_inds = (bbox_w > min_bbox_size) & (bbox_h > min_bbox_size)
            valid_inds = np.nonzero(valid_inds)[0]
            results[key] = results[key][valid_inds]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_inds]

    def __call__(self, results):
        """Call function to shear images, bounding boxes, masks and semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Sheared results.
        """
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        self._shear_img(results, magnitude, self.direction, self.interpolation)
        self._shear_bboxes(results, magnitude)
        # fill_val set to 0 for background of mask.
        self._shear_masks(
            results,
            magnitude,
            self.direction,
            fill_val=0,
            interpolation=self.interpolation,
        )
        self._shear_seg(
            results,
            magnitude,
            self.direction,
            fill_val=self.seg_ignore_label,
            interpolation=self.interpolation,
        )
        self._filter_invalid(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(level={self.level}, "
        repr_str += f"img_fill_val={self.img_fill_val}, "
        repr_str += f"seg_ignore_label={self.seg_ignore_label}, "
        repr_str += f"prob={self.prob}, "
        repr_str += f"direction={self.direction}, "
        repr_str += f"max_shear_magnitude={self.max_shear_magnitude}, "
        repr_str += f"random_negative_prob={self.random_negative_prob}, "
        repr_str += f"interpolation={self.interpolation})"
        return repr_str


@PIPELINES.register_module()
class Rotate(object):
    """Apply Rotate Transformation to image (and its corresponding bbox, mask,
    segmentation).

    Args:
        level (int | float): The level should be in range (0,_MAX_LEVEL].
        scale (int | float): Isotropic scale factor. Same in ``imrotate``.
        center (int | float | tuple[float]): Center point (w, h) of the
            rotation in the source image. If None, the center of the
            image will be used. Same in ``imrotate``.
        img_fill_val (int | float | tuple): The fill value for image border.
            If float, the same value will be used for all the three
            channels of image. If tuple, the should be 3 elements (e.g.
            equals the number of channels for image).
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Default 255.
        prob (float): The probability for perform transformation and
            should be in range 0 to 1.
        max_rotate_angle (int | float): The maximum angles for rotate
            transformation.
        random_negative_prob (float): The probability that turns the
             offset negative.
    """

    def __init__(
        self,
        level,
        scale=1,
        center=None,
        img_fill_val=128,
        seg_ignore_label=255,
        prob=0.5,
        max_rotate_angle=30,
        random_negative_prob=0.5,
    ):
        assert isinstance(
            level, (int, float)
        ), f"The level must be type int or float. got {type(level)}."
        assert (
            0 <= level <= _MAX_LEVEL
        ), f"The level should be in range (0,{_MAX_LEVEL}]. got {level}."
        assert isinstance(
            scale, (int, float)
        ), f"The scale must be type int or float. got type {type(scale)}."
        if isinstance(center, (int, float)):
            center = (center, center)
        elif isinstance(center, tuple):
            assert len(center) == 2, (
                "center with type tuple must have "
                f"2 elements. got {len(center)} elements."
            )
        else:
            assert center is None, (
                "center must be None or type int, "
                f"float or tuple, got type {type(center)}."
            )
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, (
                "img_fill_val as tuple must "
                f"have 3 elements. got {len(img_fill_val)}."
            )
            img_fill_val = tuple([float(val) for val in img_fill_val])
        else:
            raise ValueError("img_fill_val must be float or tuple with 3 elements.")
        assert np.all([0 <= val <= 255 for val in img_fill_val]), (
            "all elements of img_fill_val should between range [0,255]. "
            f"got {img_fill_val}."
        )
        assert 0 <= prob <= 1.0, (
            "The probability should be in range [0,1]. " "got {prob}."
        )
        assert isinstance(max_rotate_angle, (int, float)), (
            "max_rotate_angle "
            f"should be type int or float. got type {type(max_rotate_angle)}."
        )
        self.level = level
        self.scale = scale
        # Rotation angle in degrees. Positive values mean
        # clockwise rotation.
        self.angle = level_to_value(level, max_rotate_angle)
        self.center = center
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob
        self.max_rotate_angle = max_rotate_angle
        self.random_negative_prob = random_negative_prob

    def _rotate_img(self, results, angle, center=None, scale=1.0):
        """Rotate the image.

        Args:
            results (dict): Result dict from loading pipeline.
            angle (float): Rotation angle in degrees, positive values
                mean clockwise rotation. Same in ``imrotate``.
            center (tuple[float], optional): Center point (w, h) of the
                rotation. Same in ``imrotate``.
            scale (int | float): Isotropic scale factor. Same in ``imrotate``.
        """
        for key in results.get("img_fields", ["img"]):
            img = results[key].copy()
            img_rotated = imrotate(
                img, angle, center, scale, border_value=self.img_fill_val
            )
            results[key] = img_rotated.astype(img.dtype)

    def _rotate_bboxes(self, results, rotate_matrix):
        """Rotate the bboxes."""
        h, w, c = results["img_shape"]
        for key in results.get("bbox_fields", []):
            min_x, min_y, max_x, max_y = np.split(
                results[key], results[key].shape[-1], axis=-1
            )
            coordinates = np.stack(
                [[min_x, min_y], [max_x, min_y], [min_x, max_y], [max_x, max_y]]
            )  # [4, 2, nb_bbox, 1]
            # pad 1 to convert from format [x, y] to homogeneous
            # coordinates format [x, y, 1]
            coordinates = np.concatenate(
                (
                    coordinates,
                    np.ones((4, 1, coordinates.shape[2], 1), coordinates.dtype),
                ),
                axis=1,
            )  # [4, 3, nb_bbox, 1]
            coordinates = coordinates.transpose((2, 0, 1, 3))  # [nb_bbox, 4, 3, 1]
            rotated_coords = np.matmul(rotate_matrix, coordinates)  # [nb_bbox, 4, 2, 1]
            rotated_coords = rotated_coords[..., 0]  # [nb_bbox, 4, 2]
            min_x, min_y = (
                np.min(rotated_coords[:, :, 0], axis=1),
                np.min(rotated_coords[:, :, 1], axis=1),
            )
            max_x, max_y = (
                np.max(rotated_coords[:, :, 0], axis=1),
                np.max(rotated_coords[:, :, 1], axis=1),
            )
            min_x, min_y = (
                np.clip(min_x, a_min=0, a_max=w),
                np.clip(min_y, a_min=0, a_max=h),
            )
            max_x, max_y = (
                np.clip(max_x, a_min=min_x, a_max=w),
                np.clip(max_y, a_min=min_y, a_max=h),
            )
            results[key] = np.stack([min_x, min_y, max_x, max_y], axis=-1).astype(
                results[key].dtype
            )

    def _rotate_masks(self, results, angle, center=None, scale=1.0, fill_val=0):
        """Rotate the masks."""
        h, w, c = results["img_shape"]
        for key in results.get("mask_fields", []):
            masks = results[key]
            results[key] = masks.rotate((h, w), angle, center, scale, fill_val)

    def _rotate_seg(self, results, angle, center=None, scale=1.0, fill_val=255):
        """Rotate the segmentation map."""
        for key in results.get("seg_fields", []):
            seg = results[key].copy()
            results[key] = imrotate(
                seg, angle, center, scale, border_value=fill_val
            ).astype(seg.dtype)

    def _filter_invalid(self, results, min_bbox_size=0):
        """Filter bboxes and corresponding masks too small after rotate
        augmentation."""
        bbox2label, bbox2mask, _ = bbox2fields()
        for key in results.get("bbox_fields", []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_inds = (bbox_w > min_bbox_size) & (bbox_h > min_bbox_size)
            valid_inds = np.nonzero(valid_inds)[0]
            results[key] = results[key][valid_inds]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_inds]

    def __call__(self, results):
        """Call function to rotate images, bounding boxes, masks and semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """
        if np.random.rand() > self.prob:
            return results
        h, w = results["img"].shape[:2]
        center = self.center
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        angle = random_negative(self.angle, self.random_negative_prob)
        self._rotate_img(results, angle, center, self.scale)
        rotate_matrix = cv2.getRotationMatrix2D(center, -angle, self.scale)
        self._rotate_bboxes(results, rotate_matrix)
        self._rotate_masks(results, angle, center, self.scale, fill_val=0)
        self._rotate_seg(
            results, angle, center, self.scale, fill_val=self.seg_ignore_label
        )
        self._filter_invalid(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(level={self.level}, "
        repr_str += f"scale={self.scale}, "
        repr_str += f"center={self.center}, "
        repr_str += f"img_fill_val={self.img_fill_val}, "
        repr_str += f"seg_ignore_label={self.seg_ignore_label}, "
        repr_str += f"prob={self.prob}, "
        repr_str += f"max_rotate_angle={self.max_rotate_angle}, "
        repr_str += f"random_negative_prob={self.random_negative_prob})"
        return repr_str


@PIPELINES.register_module()
class Translate(object):
    """Translate the images, bboxes, masks and segmentation maps horizontally
    or vertically.

    Args:
        level (int | float): The level for Translate and should be in
            range [0,_MAX_LEVEL].
        prob (float): The probability for performing translation and
            should be in range [0, 1].
        img_fill_val (int | float | tuple): The filled value for image
            border. If float, the same fill value will be used for all
            the three channels of image. If tuple, the should be 3
            elements (e.g. equals the number of channels for image).
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Default 255.
        direction (str): The translate direction, either "horizontal"
            or "vertical".
        max_translate_offset (int | float): The maximum pixel's offset for
            Translate.
        random_negative_prob (float): The probability that turns the
            offset negative.
        min_size (int | float): The minimum pixel for filtering
            invalid bboxes after the translation.
    """

    def __init__(
        self,
        level,
        prob=0.5,
        img_fill_val=128,
        seg_ignore_label=255,
        direction="horizontal",
        max_translate_offset=250.0,
        random_negative_prob=0.5,
        min_size=0,
    ):
        assert isinstance(level, (int, float)), "The level must be type int or float."
        assert 0 <= level <= _MAX_LEVEL, (
            "The level used for calculating Translate's offset should be "
            "in range [0,_MAX_LEVEL]"
        )
        assert (
            0 <= prob <= 1.0
        ), "The probability of translation should be in range [0, 1]."
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, "img_fill_val as tuple must have 3 elements."
            img_fill_val = tuple([float(val) for val in img_fill_val])
        else:
            raise ValueError("img_fill_val must be type float or tuple.")
        assert np.all(
            [0 <= val <= 255 for val in img_fill_val]
        ), "all elements of img_fill_val should between range [0,255]."
        assert direction in (
            "horizontal",
            "vertical",
        ), 'direction should be "horizontal" or "vertical".'
        assert isinstance(
            max_translate_offset, (int, float)
        ), "The max_translate_offset must be type int or float."
        # the offset used for translation
        self.offset = int(level_to_value(level, max_translate_offset))
        self.level = level
        self.prob = prob
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label
        self.direction = direction
        self.max_translate_offset = max_translate_offset
        self.random_negative_prob = random_negative_prob
        self.min_size = min_size

    def _translate_img(self, results, offset, direction="horizontal"):
        """Translate the image.

        Args:
            results (dict): Result dict from loading pipeline.
            offset (int | float): The offset for translate.
            direction (str): The translate direction, either "horizontal"
                or "vertical".
        """
        for key in results.get("img_fields", ["img"]):
            img = results[key].copy()
            results[key] = imtranslate(
                img, offset, direction, self.img_fill_val
            ).astype(img.dtype)

    def _translate_bboxes(self, results, offset):
        """Shift bboxes horizontally or vertically, according to offset."""
        h, w, c = results["img_shape"]
        for key in results.get("bbox_fields", []):
            min_x, min_y, max_x, max_y = np.split(
                results[key], results[key].shape[-1], axis=-1
            )
            if self.direction == "horizontal":
                min_x = np.maximum(0, min_x + offset)
                max_x = np.minimum(w, max_x + offset)
            elif self.direction == "vertical":
                min_y = np.maximum(0, min_y + offset)
                max_y = np.minimum(h, max_y + offset)

            # the boxs translated outside of image will be filtered along with
            # the corresponding masks, by invoking ``_filter_invalid``.
            results[key] = np.concatenate([min_x, min_y, max_x, max_y], axis=-1)

    def _translate_masks(self, results, offset, direction="horizontal", fill_val=0):
        """Translate masks horizontally or vertically."""
        h, w, c = results["img_shape"]
        for key in results.get("mask_fields", []):
            masks = results[key]
            results[key] = masks.translate((h, w), offset, direction, fill_val)

    def _translate_seg(self, results, offset, direction="horizontal", fill_val=255):
        """Translate segmentation maps horizontally or vertically."""
        for key in results.get("seg_fields", []):
            seg = results[key].copy()
            results[key] = imtranslate(seg, offset, direction, fill_val).astype(
                seg.dtype
            )

    def _filter_invalid(self, results, min_size=0):
        """Filter bboxes and masks too small or translated out of image."""
        bbox2label, bbox2mask, _ = bbox2fields()
        for key in results.get("bbox_fields", []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_inds = (bbox_w > min_size) & (bbox_h > min_size)
            valid_inds = np.nonzero(valid_inds)[0]
            results[key] = results[key][valid_inds]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_inds]
        return results

    def __call__(self, results):
        """Call function to translate images, bounding boxes, masks and
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Translated results.
        """
        if np.random.rand() > self.prob:
            return results
        offset = random_negative(self.offset, self.random_negative_prob)
        self._translate_img(results, offset, self.direction)
        self._translate_bboxes(results, offset)
        # fill_val defaultly 0 for BitmapMasks and None for PolygonMasks.
        self._translate_masks(results, offset, self.direction)
        # fill_val set to ``seg_ignore_label`` for the ignored value
        # of segmentation map.
        self._translate_seg(
            results, offset, self.direction, fill_val=self.seg_ignore_label
        )
        self._filter_invalid(results, min_size=self.min_size)
        return results


@PIPELINES.register_module()
class ColorTransform(object):
    """Apply Color transformation to image. The bboxes, masks, and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Color transformation.
    """

    def __init__(self, magnitude, prob=0.5, random_negative_prob=0.5):
        assert isinstance(magnitude, (int, float)), (
            "The magnitude type must "
            f"be int or float, but got {type(magnitude)} instead."
        )
        assert 0 <= prob <= 1.0, (
            "The prob should be in range [0,1], " f"got {prob} instead."
        )
        assert 0 <= random_negative_prob <= 1.0, (
            "The random_negative_prob "
            f"should be in range [0,1], got {random_negative_prob} instead."
        )

        self.magnitude = magnitude
        self.prob = prob
        self.random_negative_prob = random_negative_prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            img_color_adjusted = adjust_color(img, alpha=1 + magnitude)
            results[key] = img_color_adjusted.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(magnitude={self.magnitude}, "
        repr_str += f"prob={self.prob}, "
        repr_str += f"random_negative_prob={self.random_negative_prob})"
        return repr_str


@PIPELINES.register_module()
class EqualizeTransform(object):
    """Apply Equalize transformation to image. The bboxes, masks and
    segmentations are not modified.
    Args:
        prob (float): The probability for performing Equalize transformation.
    """

    def __init__(self, prob=0.5):
        assert 0 <= prob <= 1.0, "The probability should be in range [0,1]."
        self.prob = prob

    def _imequalize(self, results):
        """Equalizes the histogram of one image."""
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            results[key] = imequalize(img).astype(img.dtype)

    def __call__(self, results):
        """Call function for Equalize transformation.
        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
        self._imequalize(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(prob={self.prob})"


@PIPELINES.register_module()
class AutoContrast(object):
    """Auto adjust image contrast.

    Args:
        prob (float): The probability for performing invert therefore should
             be in range [0, 1]. Defaults to 0.5.
    """

    def __init__(self, prob=0.5):
        assert 0 <= prob <= 1.0, (
            "The prob should be in range [0,1], " f"got {prob} instead."
        )

        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            img_contrasted = auto_contrast(img)
            results[key] = img_contrasted.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(prob={self.prob})"
        return repr_str


@PIPELINES.register_module()
class AutoInvert(object):
    """Invert images.

    Args:
        prob (float): The probability for performing invert therefore should
             be in range [0, 1]. Defaults to 0.5.
    """

    def __init__(self, prob=0.5):
        assert 0 <= prob <= 1.0, (
            "The prob should be in range [0,1], " f"got {prob} instead."
        )

        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            img_inverted = iminvert(img)
            results[key] = img_inverted.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(prob={self.prob})"
        return repr_str


@PIPELINES.register_module()
class AutoRotate(object):
    """Rotate images.

    Args:
        angle (float): The angle used for rotate. Positive values stand for
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If None, the center of the image will be used.
            Defaults to None.
        scale (float): Isotropic scale factor. Defaults to 1.0.
        pad_val (int, Sequence[int]): Pixel pad_val value for constant fill.
            If a sequence of length 3, it is used to pad_val R, G, B channels
            respectively. Defaults to 128.
        prob (float): The probability for performing Rotate therefore should be
            in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the angle
            negative, which should be in range [0,1]. Defaults to 0.5.
        interpolation (str): Interpolation method. Options are 'nearest',
            'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to 'nearest'.
    """

    def __init__(
        self,
        angle,
        center=None,
        scale=1.0,
        pad_val=128,
        prob=0.5,
        random_negative_prob=0.5,
        interpolation="nearest",
    ):
        assert isinstance(angle, float), (
            "The angle type must be float, but " f"got {type(angle)} instead."
        )
        if isinstance(center, tuple):
            assert len(center) == 2, (
                "center as a tuple must have 2 "
                f"elements, got {len(center)} elements instead."
            )
        else:
            assert center is None, (
                "The center type" f"must be tuple or None, got {type(center)} instead."
            )
        assert isinstance(scale, float), (
            "the scale type must be float, but " f"got {type(scale)} instead."
        )
        if isinstance(pad_val, int):
            pad_val = tuple([pad_val] * 3)
        elif isinstance(pad_val, Sequence):
            assert len(pad_val) == 3, (
                "pad_val as a tuple must have 3 "
                f"elements, got {len(pad_val)} instead."
            )
            assert all(isinstance(i, int) for i in pad_val), (
                "pad_val as a " "tuple must got elements of int type."
            )
        else:
            raise TypeError("pad_val must be int or tuple with 3 elements.")
        assert 0 <= prob <= 1.0, (
            "The prob should be in range [0,1], " f"got {prob} instead."
        )
        assert 0 <= random_negative_prob <= 1.0, (
            "The random_negative_prob "
            f"should be in range [0,1], got {random_negative_prob} instead."
        )

        self.angle = angle
        self.center = center
        self.scale = scale
        self.pad_val = tuple(pad_val)
        self.prob = prob
        self.random_negative_prob = random_negative_prob
        self.interpolation = interpolation

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        angle = random_negative(self.angle, self.random_negative_prob)
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            img_rotated = imrotate(
                img,
                angle,
                center=self.center,
                scale=self.scale,
                border_value=self.pad_val,
                interpolation=self.interpolation,
            )
            results[key] = img_rotated.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(angle={self.angle}, "
        repr_str += f"center={self.center}, "
        repr_str += f"scale={self.scale}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"prob={self.prob}, "
        repr_str += f"random_negative_prob={self.random_negative_prob}, "
        repr_str += f"interpolation={self.interpolation})"
        return repr_str


@PIPELINES.register_module()
class Posterize(object):
    """Posterize images (reduce the number of bits for each color channel).

    Args:
        bits (int | float): Number of bits for each pixel in the output img,
            which should be less or equal to 8.
        prob (float): The probability for posterizing therefore should be in
            range [0, 1]. Defaults to 0.5.
    """

    def __init__(self, bits, prob=0.5):
        assert bits <= 8, f"The bits must be less than 8, got {bits} instead."
        assert 0 <= prob <= 1.0, (
            "The prob should be in range [0,1], " f"got {prob} instead."
        )

        # To align timm version, we need to round up to integer here.
        self.bits = math.ceil(bits)
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            img_posterized = posterize(img, bits=self.bits)
            results[key] = img_posterized.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(bits={self.bits}, "
        repr_str += f"prob={self.prob})"
        return repr_str


@PIPELINES.register_module()
class Solarize(object):
    """Solarize images (invert all pixel values above a threshold).

    Args:
        thr (int | float): The threshold above which the pixels value will be
            inverted.
        prob (float): The probability for solarizing therefore should be in
            range [0, 1]. Defaults to 0.5.
    """

    def __init__(self, thr, prob=0.5):
        assert isinstance(thr, (int, float)), (
            "The thr type must " f"be int or float, but got {type(thr)} instead."
        )
        assert 0 <= prob <= 1.0, (
            "The prob should be in range [0,1], " f"got {prob} instead."
        )

        self.thr = thr
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            img_solarized = solarize(img, thr=self.thr)
            results[key] = img_solarized.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(thr={self.thr}, "
        repr_str += f"prob={self.prob})"
        return repr_str


@PIPELINES.register_module()
class SolarizeAdd(object):
    """SolarizeAdd images (add a certain value to pixels below a threshold).

    Args:
        magnitude (int | float): The value to be added to pixels below the thr.
        thr (int | float): The threshold below which the pixels value will be
            adjusted.
        prob (float): The probability for solarizing therefore should be in
            range [0, 1]. Defaults to 0.5.
    """

    def __init__(self, magnitude, thr=128, prob=0.5):
        assert isinstance(magnitude, (int, float)), (
            "The thr magnitude must "
            f"be int or float, but got {type(magnitude)} instead."
        )
        assert isinstance(thr, (int, float)), (
            "The thr type must " f"be int or float, but got {type(thr)} instead."
        )
        assert 0 <= prob <= 1.0, (
            "The prob should be in range [0,1], " f"got {prob} instead."
        )

        self.magnitude = magnitude
        self.thr = thr
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            img_solarized = np.where(
                img < self.thr, np.minimum(img + self.magnitude, 255), img
            )
            results[key] = img_solarized.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(magnitude={self.magnitude}, "
        repr_str += f"thr={self.thr}, "
        repr_str += f"prob={self.prob})"
        return repr_str


@PIPELINES.register_module()
class ContrastTransform(object):
    """Adjust images contrast.

    Args:
        magnitude (int | float): The magnitude used for adjusting contrast. A
            positive magnitude would enhance the contrast and a negative
            magnitude would make the image grayer. A magnitude=0 gives the
            origin img.
        prob (float): The probability for performing contrast adjusting
            therefore should be in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
    """

    def __init__(self, magnitude, prob=0.5, random_negative_prob=0.5):
        assert isinstance(magnitude, (int, float)), (
            "The magnitude type must "
            f"be int or float, but got {type(magnitude)} instead."
        )
        assert 0 <= prob <= 1.0, (
            "The prob should be in range [0,1], " f"got {prob} instead."
        )
        assert 0 <= random_negative_prob <= 1.0, (
            "The random_negative_prob "
            f"should be in range [0,1], got {random_negative_prob} instead."
        )

        self.magnitude = magnitude
        self.prob = prob
        self.random_negative_prob = random_negative_prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            img_contrasted = adjust_contrast(img, factor=1 + magnitude)
            results[key] = img_contrasted.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(magnitude={self.magnitude}, "
        repr_str += f"prob={self.prob}, "
        repr_str += f"random_negative_prob={self.random_negative_prob})"
        return repr_str


@PIPELINES.register_module()
class BrightnessTransform(object):
    """Adjust images brightness.

    Args:
        magnitude (int | float): The magnitude used for adjusting brightness. A
            positive magnitude would enhance the brightness and a negative
            magnitude would make the image darker. A magnitude=0 gives the
            origin img.
        prob (float): The probability for performing contrast adjusting
            therefore should be in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
    """

    def __init__(self, magnitude, prob=0.5, random_negative_prob=0.5):
        assert isinstance(magnitude, (int, float)), (
            "The magnitude type must "
            f"be int or float, but got {type(magnitude)} instead."
        )
        assert 0 <= prob <= 1.0, (
            "The prob should be in range [0,1], " f"got {prob} instead."
        )
        assert 0 <= random_negative_prob <= 1.0, (
            "The random_negative_prob "
            f"should be in range [0,1], got {random_negative_prob} instead."
        )

        self.magnitude = magnitude
        self.prob = prob
        self.random_negative_prob = random_negative_prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            img_brightened = adjust_brightness(img, factor=1 + magnitude)
            results[key] = img_brightened.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(magnitude={self.magnitude}, "
        repr_str += f"prob={self.prob}, "
        repr_str += f"random_negative_prob={self.random_negative_prob})"
        return repr_str


@PIPELINES.register_module()
class SharpnessTransform(object):
    """Adjust images sharpness.

    Args:
        magnitude (int | float): The magnitude used for adjusting sharpness. A
            positive magnitude would enhance the sharpness and a negative
            magnitude would make the image bulr. A magnitude=0 gives the
            origin img.
        prob (float): The probability for performing contrast adjusting
            therefore should be in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
    """

    def __init__(self, magnitude, prob=0.5, random_negative_prob=0.5):
        assert isinstance(magnitude, (int, float)), (
            "The magnitude type must "
            f"be int or float, but got {type(magnitude)} instead."
        )
        assert 0 <= prob <= 1.0, (
            "The prob should be in range [0,1], " f"got {prob} instead."
        )
        assert 0 <= random_negative_prob <= 1.0, (
            "The random_negative_prob "
            f"should be in range [0,1], got {random_negative_prob} instead."
        )

        self.magnitude = magnitude
        self.prob = prob
        self.random_negative_prob = random_negative_prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            img_sharpened = adjust_sharpness(img, factor=1 + magnitude)
            results[key] = img_sharpened.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(magnitude={self.magnitude}, "
        repr_str += f"prob={self.prob}, "
        repr_str += f"random_negative_prob={self.random_negative_prob})"
        return repr_str


@PIPELINES.register_module()
class AutoShear(object):
    """Shear images.

    Args:
        magnitude (int | float): The magnitude used for shear.
        pad_val (int, Sequence[int]): Pixel pad_val value for constant fill.
            If a sequence of length 3, it is used to pad_val R, G, B channels
            respectively. Defaults to 128.
        prob (float): The probability for performing Shear therefore should be
            in range [0, 1]. Defaults to 0.5.
        direction (str): The shearing direction. Options are 'horizontal' and
            'vertical'. Defaults to 'horizontal'.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
        interpolation (str): Interpolation method. Options are 'nearest',
            'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to 'bicubic'.
    """

    def __init__(
        self,
        magnitude,
        pad_val=128,
        prob=0.5,
        direction="horizontal",
        random_negative_prob=0.5,
        interpolation="bicubic",
    ):
        assert isinstance(magnitude, (int, float)), (
            "The magnitude type must "
            f"be int or float, but got {type(magnitude)} instead."
        )
        if isinstance(pad_val, int):
            pad_val = tuple([pad_val] * 3)
        elif isinstance(pad_val, Sequence):
            assert len(pad_val) == 3, (
                "pad_val as a tuple must have 3 "
                f"elements, got {len(pad_val)} instead."
            )
            assert all(isinstance(i, int) for i in pad_val), (
                "pad_val as a " "tuple must got elements of int type."
            )
        else:
            raise TypeError("pad_val must be int or tuple with 3 elements.")
        assert 0 <= prob <= 1.0, (
            "The prob should be in range [0,1], " f"got {prob} instead."
        )
        assert direction in ("horizontal", "vertical"), (
            "direction must be "
            f'either "horizontal" or "vertical", got {direction} instead.'
        )
        assert 0 <= random_negative_prob <= 1.0, (
            "The random_negative_prob "
            f"should be in range [0,1], got {random_negative_prob} instead."
        )

        self.magnitude = magnitude
        self.pad_val = tuple(pad_val)
        self.prob = prob
        self.direction = direction
        self.random_negative_prob = random_negative_prob
        self.interpolation = interpolation

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            img_sheared = imshear(
                img,
                magnitude,
                direction=self.direction,
                border_value=self.pad_val,
                interpolation=self.interpolation,
            )
            results[key] = img_sheared.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(magnitude={self.magnitude}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"prob={self.prob}, "
        repr_str += f"direction={self.direction}, "
        repr_str += f"random_negative_prob={self.random_negative_prob}, "
        repr_str += f"interpolation={self.interpolation})"
        return repr_str


@PIPELINES.register_module()
class AutoTranslate(object):
    """Translate images.
    Args:
        magnitude (int | float): The magnitude used for translate. Note that
            the offset is calculated by magnitude * size in the corresponding
            direction. With a magnitude of 1, the whole image will be moved out
            of the range.
        pad_val (int, Sequence[int]): Pixel pad_val value for constant fill.
            If a sequence of length 3, it is used to pad_val R, G, B channels
            respectively. Defaults to 128.
        prob (float): The probability for performing translate therefore should
             be in range [0, 1]. Defaults to 0.5.
        direction (str): The translating direction. Options are 'horizontal'
            and 'vertical'. Defaults to 'horizontal'.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
        interpolation (str): Interpolation method. Options are 'nearest',
            'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to 'nearest'.
    """

    def __init__(
        self,
        magnitude,
        pad_val=128,
        prob=0.5,
        direction="horizontal",
        random_negative_prob=0.5,
        interpolation="nearest",
    ):
        assert isinstance(magnitude, (int, float)), (
            "The magnitude type must "
            f"be int or float, but got {type(magnitude)} instead."
        )
        if isinstance(pad_val, int):
            pad_val = tuple([pad_val] * 3)
        elif isinstance(pad_val, Sequence):
            assert len(pad_val) == 3, (
                "pad_val as a tuple must have 3 "
                f"elements, got {len(pad_val)} instead."
            )
            assert all(isinstance(i, int) for i in pad_val), (
                "pad_val as a " "tuple must got elements of int type."
            )
        else:
            raise TypeError("pad_val must be int or tuple with 3 elements.")
        assert 0 <= prob <= 1.0, (
            "The prob should be in range [0,1], " f"got {prob} instead."
        )
        assert direction in ("horizontal", "vertical"), (
            "direction must be "
            f'either "horizontal" or "vertical", got {direction} instead.'
        )
        assert 0 <= random_negative_prob <= 1.0, (
            "The random_negative_prob "
            f"should be in range [0,1], got {random_negative_prob} instead."
        )

        self.magnitude = magnitude
        self.pad_val = tuple(pad_val)
        self.prob = prob
        self.direction = direction
        self.random_negative_prob = random_negative_prob
        self.interpolation = interpolation

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            height, width = img.shape[:2]
            if self.direction == "horizontal":
                offset = magnitude * width
            else:
                offset = magnitude * height
            img_translated = imtranslate(
                img,
                offset,
                direction=self.direction,
                border_value=self.pad_val,
                interpolation=self.interpolation,
            )
            results[key] = img_translated.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(magnitude={self.magnitude}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"prob={self.prob}, "
        repr_str += f"direction={self.direction}, "
        repr_str += f"random_negative_prob={self.random_negative_prob}, "
        repr_str += f"interpolation={self.interpolation})"
        return repr_str
