import numpy as np
import cv2
import inspect

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

from mtl.utils.mask_util import PolygonMasks
from mtl.utils.misc_util import is_str
from ..data_wrapper import PIPELINES
from .compose import Compose as PipelineCompose


@PIPELINES.register_module()
class LetterResize:
    """from https://github.com/ultralytics/yolov5"""

    def __init__(
        self,
        img_scale=None,
        color=(114, 114, 114),
        auto=True,
        scaleFill=False,
        scaleup=True,
        backend="cv2",
    ):
        self.image_size_hw = img_scale
        self.color = color
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.backend = backend

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]

            shape = img.shape[:2]  # current shape [height, width]
            if isinstance(self.image_size_hw, int):
                self.image_size_hw = (self.image_size_hw, self.image_size_hw)

            # Scale ratio (new / old)
            r = min(self.image_size_hw[0] / shape[0], self.image_size_hw[1] / shape[1])
            if (
                not self.scaleup
            ):  # only scale down, do not scale up (for better test map)
                r = min(r, 1.0)
            ratio = r, r
            # find most proper size
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            # pad for fixed size
            dw, dh = (
                self.image_size_hw[1] - new_unpad[0],
                self.image_size_hw[0] - new_unpad[1],
            )  # wh padding
            if self.auto:  # minimum rectangle
                dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
            elif self.scaleFill:  # stretch
                dw, dh = 0.0, 0.0
                # scale to fixed size
                new_unpad = (self.image_size_hw[1], self.image_size_hw[0])
                ratio = (
                    self.image_size_hw[1] / shape[1],
                    self.image_size_hw[0] / shape[0],
                )  # width, height ratios

            # padding for left and right
            dw /= 2  # divide padding into 2 sides
            dh /= 2

            # no padding
            if shape[::-1] != new_unpad:  # resize
                img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            results["img_shape"] = img.shape
            scale_factor = np.array(
                [ratio[0], ratio[1], ratio[0], ratio[1]], dtype=np.float32
            )
            results["scale_factor"] = scale_factor

            # padding
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color
            )  # add border

            results[key] = img

            results["pad_shape"] = img.shape
            results["pad_param"] = np.array(
                [top, bottom, left, right], dtype=np.float32
            )
        return results


@PIPELINES.register_module()
class GtBBoxesFilter(object):
    def __init__(self, min_size=2, max_aspect_ratio=20):
        assert max_aspect_ratio > 1
        self.min_size = min_size
        self.max_aspect_ratio = max_aspect_ratio

    def __call__(self, results):
        bboxes = results["gt_bboxes"]
        labels = results["gt_labels"]
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        valid = (w > self.min_size) & (h > self.min_size) & (ar < self.max_aspect_ratio)
        results["gt_bboxes"] = bboxes[valid]
        results["gt_labels"] = labels[valid]
        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"min_size={self.min_size}, "
            f"max_aspect_ratio={self.max_aspect_ratio})"
        )
        return repr_str


@PIPELINES.register_module()
class Albu:
    """Albumentation augmentation.
    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.

    An example of ``transforms`` is as followed:
    .. code-block::
        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]
    """

    def __init__(
        self,
        transforms,
        bbox_params=None,
        keymap=None,
        update_pad_shape=False,
        skip_img_without_anno=False,
    ):
        """Initialization for albu augmentation.

        Args:
            transforms (list[dict]): A list of albu transformations
            bbox_params (dict): Bbox_params for albumentation `Compose`
            keymap (dict): Contains {'input key':'albumentation-style key'}
            skip_img_without_anno (bool): Whether to skip the image if no ann left
                after aug
        """
        if Compose is None:
            raise RuntimeError("albumentations is not installed")

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (
            isinstance(bbox_params, dict)
            and "label_fields" in bbox_params
            and "filter_lost_elements" in bbox_params
        ):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params["label_fields"]
            bbox_params["label_fields"] = ["idx_mapper"]
            del bbox_params["filter_lost_elements"]

        self.bbox_params = self.albu_builder(bbox_params) if bbox_params else None
        self.aug = Compose(
            [self.albu_builder(t) for t in self.transforms],
            bbox_params=self.bbox_params,
        )

        if not keymap:
            self.keymap_to_albu = {
                "img": "image",
                "gt_masks": "masks",
                "gt_bboxes": "bboxes",
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        It inherits some of: logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed object.
        """
        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()

        obj_type = args.pop("type")
        if is_str(obj_type):
            if albumentations is None:
                raise RuntimeError("albumentations is not installed")
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f"type must be a str or valid type, but got {type(obj_type)}"
            )

        if "transforms" in args:
            args["transforms"] = [
                self.albu_builder(transform) for transform in args["transforms"]
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper. Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}

        Returns:
            dict: new dict.
        """
        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)
        # TODO: add bbox_fields
        if "bboxes" in results:
            # to list of boxes
            if isinstance(results["bboxes"], np.ndarray):
                results["bboxes"] = [x for x in results["bboxes"]]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results["idx_mapper"] = np.arange(len(results["bboxes"]))

        # TODO: Support mask structure in albu
        if "masks" in results:
            if isinstance(results["masks"], PolygonMasks):
                raise NotImplementedError("Albu only supports BitMap masks now")
            ori_masks = results["masks"]
            results["masks"] = results["masks"].masks

        results = self.aug(**results)

        if "bboxes" in results:
            if isinstance(results["bboxes"], list):
                results["bboxes"] = np.array(results["bboxes"], dtype=np.float32)
            results["bboxes"] = results["bboxes"].reshape(-1, 4)

            # filter label_fields
            if self.filter_lost_elements:

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results["idx_mapper"]]
                    )
                if "masks" in results:
                    results["masks"] = np.array(
                        [results["masks"][i] for i in results["idx_mapper"]]
                    )
                    results["masks"] = ori_masks.__class__(
                        results["masks"],
                        results["image"].shape[0],
                        results["image"].shape[1],
                    )

                if not len(results["idx_mapper"]) and self.skip_img_without_anno:
                    return None

        if "gt_labels" in results:
            if isinstance(results["gt_labels"], list):
                results["gt_labels"] = np.array(results["gt_labels"])
            results["gt_labels"] = results["gt_labels"].astype(np.int64)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results["pad_shape"] = results["img"].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f"(transforms={self.transforms})"
        return repr_str


@PIPELINES.register_module()
class MosaicPipeline(object):
    def __init__(self, individual_pipeline, pad_val=0):
        self.individual_pipeline = PipelineCompose(individual_pipeline)
        self.pad_val = pad_val

    def __call__(self, results):
        input_results = results.copy()
        mosaic_results = [results]
        dataset = results["dataset"]
        # load another 3 images
        indices = dataset.batch_rand_others(results["_idx"], 3)
        for idx in indices:
            img_info = dataset.getitem_info(idx)
            ann_info = dataset.get_ann_info(idx)
            if "img" in img_info:
                _results = dict(
                    img_info=img_info, ann_info=ann_info, _idx=idx, img=img_info["img"]
                )
            else:
                _results = dict(img_info=img_info, ann_info=ann_info, _idx=idx)
            if dataset.proposals is not None:
                _results["proposals"] = dataset.proposals[idx]
            dataset.pre_pipeline(_results)
            mosaic_results.append(_results)

        for idx in range(4):
            mosaic_results[idx] = self.individual_pipeline(mosaic_results[idx])

        shapes = [results["pad_shape"] for results in mosaic_results]
        cxy = max(shapes[0][0], shapes[1][0], shapes[0][1], shapes[2][1])
        canvas_shape = (cxy * 2, cxy * 2, shapes[0][2])

        # base image with 4 tiles
        canvas = dict()
        for key in mosaic_results[0].get("img_fields", []):
            canvas[key] = np.full(canvas_shape, self.pad_val, dtype=np.uint8)
        for i, results in enumerate(mosaic_results):
            h, w = results["pad_shape"][:2]
            # place img in img4
            if i == 0:  # top left
                x1, y1, x2, y2 = cxy - w, cxy - h, cxy, cxy
            elif i == 1:  # top right
                x1, y1, x2, y2 = cxy, cxy - h, cxy + w, cxy
            elif i == 2:  # bottom left
                x1, y1, x2, y2 = cxy - w, cxy, cxy, cxy + h
            elif i == 3:  # bottom right
                x1, y1, x2, y2 = cxy, cxy, cxy + w, cxy + h

            for key in mosaic_results[0].get("img_fields", []):
                canvas[key][y1:y2, x1:x2] = results[key]

            for key in results.get("bbox_fields", []):
                bboxes = results[key]
                bboxes[:, 0::2] = bboxes[:, 0::2] + x1
                bboxes[:, 1::2] = bboxes[:, 1::2] + y1
                results[key] = bboxes

        output_results = input_results
        output_results["file_name"] = None
        output_results["img_fields"] = mosaic_results[0].get("img_fields", [])
        output_results["bbox_fields"] = mosaic_results[0].get("bbox_fields", [])
        for key in output_results["img_fields"]:
            output_results[key] = canvas[key]

        for key in output_results["bbox_fields"]:
            output_results[key] = np.concatenate(
                [r[key] for r in mosaic_results], axis=0
            )

        output_results["gt_labels"] = np.concatenate(
            [r["gt_labels"] for r in mosaic_results], axis=0
        )

        output_results["img_shape"] = canvas_shape
        output_results["ori_shape"] = canvas_shape
        output_results["flip"] = False
        output_results["flip_direction"] = None

        return output_results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"individual_pipeline={self.individual_pipeline}, "
            f"pad_val={self.pad_val})"
        )
        return repr_str
