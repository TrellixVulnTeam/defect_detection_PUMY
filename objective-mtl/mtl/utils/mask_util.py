import cv2
import numpy as np
import torch
from abc import ABCMeta, abstractmethod
from torch.nn.modules.utils import _pair
import pycocotools.mask as mask_util
import pycocotools.mask as maskUtils

from mtl.cores.ops import roi_align
from mtl.utils.misc_util import slice_list
from mtl.utils.geometric_util import (
    rescale_size,
    imrescale,
    imresize,
    imflip,
    impad,
    imshear,
    imtranslate,
    imrotate,
)


class BaseInstanceMasks(metaclass=ABCMeta):
    """Base class for instance masks."""

    @abstractmethod
    def rescale(self, scale, interpolation="nearest"):
        """Rescale masks as large as possible while keeping the aspect ratio.
        For details can refer to `imrescale`.

        Args:
            scale (tuple[int]): The maximum size (h, w) of rescaled mask.
            interpolation (str): Same as :func:`imrescale`.

        Returns:
            BaseInstanceMasks: The rescaled masks.
        """
        pass

    @abstractmethod
    def resize(self, out_shape, interpolation="nearest"):
        """Resize masks to the given out_shape.

        Args:
            out_shape: Target (h, w) of resized mask.
            interpolation (str): See :func:`imresize`.

        Returns:
            BaseInstanceMasks: The resized masks.
        """
        pass

    @abstractmethod
    def flip(self, flip_direction="horizontal"):
        """Flip masks alone the given direction.

        Args:
            flip_direction (str): Either 'horizontal' or 'vertical'.

        Returns:
            BaseInstanceMasks: The flipped masks.
        """
        pass

    @abstractmethod
    def pad(self, out_shape, pad_val):
        """Pad masks to the given size of (h, w).

        Args:
            out_shape (tuple[int]): Target (h, w) of padded mask.
            pad_val (int): The padded value.

        Returns:
            BaseInstanceMasks: The padded masks.
        """
        pass

    @abstractmethod
    def crop(self, bbox):
        """Crop each mask by the given bbox.

        Args:
            bbox (ndarray): Bbox in format [x1, y1, x2, y2], shape (4, ).

        Return:
            BaseInstanceMasks: The cropped masks.
        """
        pass

    @abstractmethod
    def crop_and_resize(
        self, bboxes, out_shape, inds, device, interpolation="bilinear"
    ):
        """Crop and resize masks by the given bboxes.

        This function is mainly used in mask targets computation.
        It firstly align mask to bboxes by assigned_inds, then crop mask by the
        assigned bbox and resize to the size of (mask_h, mask_w)

        Args:
            bboxes (Tensor): Bboxes in format [x1, y1, x2, y2], shape (N, 4)
            out_shape (tuple[int]): Target (h, w) of resized mask
            inds (ndarray): Indexes to assign masks to each bbox
            device (str): Device of bboxes
            interpolation (str): See `imresize`

        Return:
            BaseInstanceMasks: the cropped and resized masks.
        """
        pass

    @abstractmethod
    def expand(self, expanded_h, expanded_w, top, left):
        """see :class:`Expand`."""
        pass

    @property
    @abstractmethod
    def areas(self):
        """ndarray: areas of each instance."""
        pass

    @abstractmethod
    def to_ndarray(self):
        """Convert masks to the format of ndarray.

        Return:
            ndarray: Converted masks in the format of ndarray.
        """
        pass

    @abstractmethod
    def to_tensor(self, dtype, device):
        """Convert masks to the format of Tensor.

        Args:
            dtype (str): Dtype of converted mask.
            device (torch.device): Device of converted masks.

        Returns:
            Tensor: Converted masks in the format of Tensor.
        """
        pass

    @abstractmethod
    def translate(
        self,
        out_shape,
        offset,
        direction="horizontal",
        fill_val=0,
        interpolation="bilinear",
    ):
        """Translate the masks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            offset (int | float): The offset for translate.
            direction (str): The translate direction, either "horizontal"
                or "vertical".
            fill_val (int | float): Border value. Default 0.
            interpolation (str): Same as :func:`imtranslate`.

        Returns:
            Translated masks.
        """
        pass

    def shear(
        self,
        out_shape,
        magnitude,
        direction="horizontal",
        border_value=0,
        interpolation="bilinear",
    ):
        """Shear the masks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            magnitude (int | float): The magnitude used for shear.
            direction (str): The shear direction, either "horizontal"
                or "vertical".
            border_value (int | tuple[int]): Value used in case of a
                constant border. Default 0.
            interpolation (str): Same as in :func:`imshear`.

        Returns:
            ndarray: Sheared masks.
        """
        pass

    @abstractmethod
    def rotate(self, out_shape, angle, center=None, scale=1.0, fill_val=0):
        """Rotate the masks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            angle (int | float): Rotation angle in degrees. Positive values
                mean counter-clockwise rotation.
            center (tuple[float], optional): Center point (w, h) of the
                rotation in source image. If not specified, the center of
                the image will be used.
            scale (int | float): Isotropic scale factor.
            fill_val (int | float): Border value. Default 0 for masks.

        Returns:
            Rotated masks.
        """
        pass


class BitmapMasks(BaseInstanceMasks):
    """This class represents masks in the form of bitmaps.

    Args:
        masks (ndarray): ndarray of masks in shape (N, H, W), where N is
            the number of objects.
        height (int): height of masks
        width (int): width of masks
    """

    def __init__(self, masks, height, width):
        self.height = height
        self.width = width
        if len(masks) == 0:
            self.masks = np.empty((0, self.height, self.width), dtype=np.uint8)
        else:
            assert isinstance(masks, (list, np.ndarray))
            if isinstance(masks, list):
                assert isinstance(masks[0], np.ndarray)
                assert masks[0].ndim == 2  # (H, W)
            else:
                assert masks.ndim == 3  # (N, H, W)

            self.masks = np.stack(masks).reshape(-1, height, width)
            assert self.masks.shape[1] == self.height
            assert self.masks.shape[2] == self.width

    def __getitem__(self, index):
        """Index the BitmapMask.

        Args:
            index (int | ndarray): Indices in the format of integer or ndarray.

        Returns:
            :obj:`BitmapMasks`: Indexed bitmap masks.
        """
        masks = self.masks[index].reshape(-1, self.height, self.width)
        return BitmapMasks(masks, self.height, self.width)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += f"num_masks={len(self.masks)}, "
        s += f"height={self.height}, "
        s += f"width={self.width})"
        return s

    def __len__(self):
        """Number of masks."""
        return len(self.masks)

    def rescale(self, scale, interpolation="nearest"):
        """See :func:`BaseInstanceMasks.rescale`."""
        if len(self.masks) == 0:
            new_w, new_h = rescale_size((self.width, self.height), scale)
            rescaled_masks = np.empty((0, new_h, new_w), dtype=np.uint8)
        else:
            rescaled_masks = np.stack(
                [
                    imrescale(mask, scale, interpolation=interpolation)
                    for mask in self.masks
                ]
            )
        height, width = rescaled_masks.shape[1:]
        return BitmapMasks(rescaled_masks, height, width)

    def resize(self, out_shape, interpolation="nearest"):
        """See :func:`BaseInstanceMasks.resize`."""
        if len(self.masks) == 0:
            resized_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            resized_masks = np.stack(
                [
                    imresize(mask, out_shape, interpolation=interpolation)
                    for mask in self.masks
                ]
            )
        return BitmapMasks(resized_masks, *out_shape)

    def flip(self, flip_direction="horizontal"):
        """See :func:`BaseInstanceMasks.flip`."""
        assert flip_direction in ("horizontal", "vertical", "diagonal")

        if len(self.masks) == 0:
            flipped_masks = self.masks
        else:
            flipped_masks = np.stack(
                [imflip(mask, direction=flip_direction) for mask in self.masks]
            )
        return BitmapMasks(flipped_masks, self.height, self.width)

    def pad(self, out_shape, pad_val=0):
        """See :func:`BaseInstanceMasks.pad`."""
        if len(self.masks) == 0:
            padded_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            padded_masks = np.stack(
                [impad(mask, shape=out_shape, pad_val=pad_val) for mask in self.masks]
            )
        return BitmapMasks(padded_masks, *out_shape)

    def crop(self, bbox):
        """See :func:`BaseInstanceMasks.crop`."""
        assert isinstance(bbox, np.ndarray)
        assert bbox.ndim == 1

        # clip the boundary
        bbox = bbox.copy()
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.height)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1, 1)
        h = np.maximum(y2 - y1, 1)

        if len(self.masks) == 0:
            cropped_masks = np.empty((0, h, w), dtype=np.uint8)
        else:
            cropped_masks = self.masks[:, y1 : y1 + h, x1 : x1 + w]
        return BitmapMasks(cropped_masks, h, w)

    def crop_and_resize(
        self, bboxes, out_shape, inds, device="cpu", interpolation="bilinear"
    ):
        """See :func:`BaseInstanceMasks.crop_and_resize`."""
        if len(self.masks) == 0:
            empty_masks = np.empty((0, *out_shape), dtype=np.uint8)
            return BitmapMasks(empty_masks, *out_shape)

        # convert bboxes to tensor
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes).to(device=device)
        if isinstance(inds, np.ndarray):
            inds = torch.from_numpy(inds).to(device=device)

        num_bbox = bboxes.shape[0]
        fake_inds = torch.arange(num_bbox, device=device).to(dtype=bboxes.dtype)[
            :, None
        ]
        rois = torch.cat([fake_inds, bboxes], dim=1)  # Nx5
        rois = rois.to(device=device)
        if num_bbox > 0:
            gt_masks_th = (
                torch.from_numpy(self.masks)
                .to(device)
                .index_select(0, inds)
                .to(dtype=rois.dtype)
            )
            targets = roi_align(
                gt_masks_th[:, None, :, :], rois, out_shape, 1.0, -1, True
            ).squeeze(1)
            resized_masks = (targets >= 0.5).cpu().numpy()
        else:
            resized_masks = []
        return BitmapMasks(resized_masks, *out_shape)

    def expand(self, expanded_h, expanded_w, top, left):
        """See :func:`BaseInstanceMasks.expand`."""
        if len(self.masks) == 0:
            expanded_mask = np.empty((0, expanded_h, expanded_w), dtype=np.uint8)
        else:
            expanded_mask = np.zeros(
                (len(self), expanded_h, expanded_w), dtype=np.uint8
            )
            expanded_mask[
                :, top : top + self.height, left : left + self.width
            ] = self.masks
        return BitmapMasks(expanded_mask, expanded_h, expanded_w)

    def translate(
        self,
        out_shape,
        offset,
        direction="horizontal",
        fill_val=0,
        interpolation="bilinear",
    ):
        """Translate the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            offset (int | float): The offset for translate.
            direction (str): The translate direction, either "horizontal"
                or "vertical".
            fill_val (int | float): Border value. Default 0 for masks.
            interpolation (str): Same as :func:`imtranslate`.

        Returns:
            BitmapMasks: Translated BitmapMasks.
        """
        if len(self.masks) == 0:
            translated_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            translated_masks = imtranslate(
                self.masks.transpose((1, 2, 0)),
                offset,
                direction,
                border_value=fill_val,
                interpolation=interpolation,
            )
            if translated_masks.ndim == 2:
                translated_masks = translated_masks[:, :, None]
            translated_masks = translated_masks.transpose((2, 0, 1)).astype(
                self.masks.dtype
            )
        return BitmapMasks(translated_masks, *out_shape)

    def shear(
        self,
        out_shape,
        magnitude,
        direction="horizontal",
        border_value=0,
        interpolation="bilinear",
    ):
        """Shear the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            magnitude (int | float): The magnitude used for shear.
            direction (str): The shear direction, either "horizontal"
                or "vertical".
            border_value (int | tuple[int]): Value used in case of a
                constant border.
            interpolation (str): Same as in :func:`imshear`.

        Returns:
            BitmapMasks: The sheared masks.
        """
        if len(self.masks) == 0:
            sheared_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            sheared_masks = imshear(
                self.masks.transpose((1, 2, 0)),
                magnitude,
                direction,
                border_value=border_value,
                interpolation=interpolation,
            )
            if sheared_masks.ndim == 2:
                sheared_masks = sheared_masks[:, :, None]
            sheared_masks = sheared_masks.transpose((2, 0, 1)).astype(self.masks.dtype)
        return BitmapMasks(sheared_masks, *out_shape)

    def rotate(self, out_shape, angle, center=None, scale=1.0, fill_val=0):
        """Rotate the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            angle (int | float): Rotation angle in degrees. Positive values
                mean counter-clockwise rotation.
            center (tuple[float], optional): Center point (w, h) of the
                rotation in source image. If not specified, the center of
                the image will be used.
            scale (int | float): Isotropic scale factor.
            fill_val (int | float): Border value. Default 0 for masks.

        Returns:
            BitmapMasks: Rotated BitmapMasks.
        """
        if len(self.masks) == 0:
            rotated_masks = np.empty((0, *out_shape), dtype=self.masks.dtype)
        else:
            rotated_masks = imrotate(
                self.masks.transpose((1, 2, 0)),
                angle,
                center=center,
                scale=scale,
                border_value=fill_val,
            )
            if rotated_masks.ndim == 2:
                # case when only one mask, (h, w)
                rotated_masks = rotated_masks[:, :, None]  # (h, w, 1)
            rotated_masks = rotated_masks.transpose((2, 0, 1)).astype(self.masks.dtype)
        return BitmapMasks(rotated_masks, *out_shape)

    @property
    def areas(self):
        """See :py:attr:`BaseInstanceMasks.areas`."""
        return self.masks.sum((1, 2))

    def to_ndarray(self):
        """See :func:`BaseInstanceMasks.to_ndarray`."""
        return self.masks

    def to_tensor(self, dtype, device):
        """See :func:`BaseInstanceMasks.to_tensor`."""
        return torch.tensor(self.masks, dtype=dtype, device=device)


class PolygonMasks(BaseInstanceMasks):
    """This class represents masks in the form of polygons.

    Polygons is a list of three levels. The first level of the list
    corresponds to objects, the second level to the polys that compose the
    object, the third level to the poly coordinates

    Args:
        masks (list[list[ndarray]]): The first level of the list
            corresponds to objects, the second level to the polys that
            compose the object, the third level to the poly coordinates
        height (int): height of masks
        width (int): width of masks
    """

    def __init__(self, masks, height, width):
        assert isinstance(masks, list)
        if len(masks) > 0:
            assert isinstance(masks[0], list)
            assert isinstance(masks[0][0], np.ndarray)

        self.height = height
        self.width = width
        self.masks = masks

    def __getitem__(self, index):
        """Index the polygon masks.

        Args:
            index (ndarray | List): The indices.

        Returns:
            :obj:`PolygonMasks`: The indexed polygon masks.
        """
        if isinstance(index, np.ndarray):
            index = index.tolist()
        if isinstance(index, list):
            masks = [self.masks[i] for i in index]
        else:
            try:
                masks = self.masks[index]
            except Exception:
                raise ValueError(
                    f"Unsupported input of type {type(index)} for indexing!"
                )
        if len(masks) and isinstance(masks[0], np.ndarray):
            masks = [masks]  # ensure a list of three levels
        return PolygonMasks(masks, self.height, self.width)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += f"num_masks={len(self.masks)}, "
        s += f"height={self.height}, "
        s += f"width={self.width})"
        return s

    def __len__(self):
        """Number of masks."""
        return len(self.masks)

    def rescale(self, scale, interpolation=None):
        """see :func:`BaseInstanceMasks.rescale`"""
        new_w, new_h = rescale_size((self.width, self.height), scale)
        if len(self.masks) == 0:
            rescaled_masks = PolygonMasks([], new_h, new_w)
        else:
            rescaled_masks = self.resize((new_h, new_w))
        return rescaled_masks

    def resize(self, out_shape, interpolation=None):
        """see :func:`BaseInstanceMasks.resize`"""
        if len(self.masks) == 0:
            resized_masks = PolygonMasks([], *out_shape)
        else:
            h_scale = out_shape[0] / self.height
            w_scale = out_shape[1] / self.width
            resized_masks = []
            for poly_per_obj in self.masks:
                resized_poly = []
                for p in poly_per_obj:
                    p = p.copy()
                    p[0::2] *= w_scale
                    p[1::2] *= h_scale
                    resized_poly.append(p)
                resized_masks.append(resized_poly)
            resized_masks = PolygonMasks(resized_masks, *out_shape)
        return resized_masks

    def flip(self, flip_direction="horizontal"):
        """see :func:`BaseInstanceMasks.flip`"""
        assert flip_direction in ("horizontal", "vertical", "diagonal")
        if len(self.masks) == 0:
            flipped_masks = PolygonMasks([], self.height, self.width)
        else:
            flipped_masks = []
            for poly_per_obj in self.masks:
                flipped_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    if flip_direction == "horizontal":
                        p[0::2] = self.width - p[0::2]
                    elif flip_direction == "vertical":
                        p[1::2] = self.height - p[1::2]
                    else:
                        p[0::2] = self.width - p[0::2]
                        p[1::2] = self.height - p[1::2]
                    flipped_poly_per_obj.append(p)
                flipped_masks.append(flipped_poly_per_obj)
            flipped_masks = PolygonMasks(flipped_masks, self.height, self.width)
        return flipped_masks

    def crop(self, bbox):
        """see :func:`BaseInstanceMasks.crop`"""
        assert isinstance(bbox, np.ndarray)
        assert bbox.ndim == 1

        # clip the boundary
        bbox = bbox.copy()
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.height)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1, 1)
        h = np.maximum(y2 - y1, 1)

        if len(self.masks) == 0:
            cropped_masks = PolygonMasks([], h, w)
        else:
            cropped_masks = []
            for poly_per_obj in self.masks:
                cropped_poly_per_obj = []
                for p in poly_per_obj:
                    # pycocotools will clip the boundary
                    p = p.copy()
                    p[0::2] -= bbox[0]
                    p[1::2] -= bbox[1]
                    cropped_poly_per_obj.append(p)
                cropped_masks.append(cropped_poly_per_obj)
            cropped_masks = PolygonMasks(cropped_masks, h, w)
        return cropped_masks

    def pad(self, out_shape, pad_val=0):
        """padding has no effect on polygons`"""
        return PolygonMasks(self.masks, *out_shape)

    def expand(self, *args, **kwargs):
        """TODO: Add expand for polygon"""
        raise NotImplementedError

    def crop_and_resize(
        self, bboxes, out_shape, inds, device="cpu", interpolation="bilinear"
    ):
        """see :func:`BaseInstanceMasks.crop_and_resize`"""
        out_h, out_w = out_shape
        if len(self.masks) == 0:
            return PolygonMasks([], out_h, out_w)

        resized_masks = []
        for i in range(len(bboxes)):
            mask = self.masks[inds[i]]
            bbox = bboxes[i, :]
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1, 1)
            h = np.maximum(y2 - y1, 1)
            h_scale = out_h / max(h, 0.1)  # avoid too large scale
            w_scale = out_w / max(w, 0.1)

            resized_mask = []
            for p in mask:
                p = p.copy()
                # crop
                # pycocotools will clip the boundary
                p[0::2] -= bbox[0]
                p[1::2] -= bbox[1]

                # resize
                p[0::2] *= w_scale
                p[1::2] *= h_scale
                resized_mask.append(p)
            resized_masks.append(resized_mask)
        return PolygonMasks(resized_masks, *out_shape)

    def translate(
        self,
        out_shape,
        offset,
        direction="horizontal",
        fill_val=None,
        interpolation=None,
    ):
        """Translate the PolygonMasks."""
        assert fill_val is None or fill_val == 0, (
            "Here fill_val is not "
            f"used, and defaultly should be None or 0. got {fill_val}."
        )
        if len(self.masks) == 0:
            translated_masks = PolygonMasks([], *out_shape)
        else:
            translated_masks = []
            for poly_per_obj in self.masks:
                translated_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    if direction == "horizontal":
                        p[0::2] = np.clip(p[0::2] + offset, 0, out_shape[1])
                    elif direction == "vertical":
                        p[1::2] = np.clip(p[1::2] + offset, 0, out_shape[0])
                    translated_poly_per_obj.append(p)
                translated_masks.append(translated_poly_per_obj)
            translated_masks = PolygonMasks(translated_masks, *out_shape)
        return translated_masks

    def shear(
        self,
        out_shape,
        magnitude,
        direction="horizontal",
        border_value=0,
        interpolation="bilinear",
    ):
        """See :func:`BaseInstanceMasks.shear`."""
        if len(self.masks) == 0:
            sheared_masks = PolygonMasks([], *out_shape)
        else:
            sheared_masks = []
            if direction == "horizontal":
                shear_matrix = np.stack([[1, magnitude], [0, 1]]).astype(np.float32)
            elif direction == "vertical":
                shear_matrix = np.stack([[1, 0], [magnitude, 1]]).astype(np.float32)
            for poly_per_obj in self.masks:
                sheared_poly = []
                for p in poly_per_obj:
                    p = np.stack([p[0::2], p[1::2]], axis=0)  # [2, n]
                    new_coords = np.matmul(shear_matrix, p)  # [2, n]
                    new_coords[0, :] = np.clip(new_coords[0, :], 0, out_shape[1])
                    new_coords[1, :] = np.clip(new_coords[1, :], 0, out_shape[0])
                    sheared_poly.append(new_coords.transpose((1, 0)).reshape(-1))
                sheared_masks.append(sheared_poly)
            sheared_masks = PolygonMasks(sheared_masks, *out_shape)
        return sheared_masks

    def rotate(self, out_shape, angle, center=None, scale=1.0, fill_val=0):
        """See :func:`BaseInstanceMasks.rotate`."""
        if len(self.masks) == 0:
            rotated_masks = PolygonMasks([], *out_shape)
        else:
            rotated_masks = []
            rotate_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
            for poly_per_obj in self.masks:
                rotated_poly = []
                for p in poly_per_obj:
                    p = p.copy()
                    coords = np.stack([p[0::2], p[1::2]], axis=1)  # [n, 2]
                    # pad 1 to convert from format [x, y] to homogeneous
                    # coordinates format [x, y, 1]
                    coords = np.concatenate(
                        (coords, np.ones((coords.shape[0], 1), coords.dtype)), axis=1
                    )  # [n, 3]
                    rotated_coords = np.matmul(
                        rotate_matrix[None, :, :], coords[:, :, None]
                    )[
                        ..., 0
                    ]  # [n, 2, 1] -> [n, 2]
                    rotated_coords[:, 0] = np.clip(
                        rotated_coords[:, 0], 0, out_shape[1]
                    )
                    rotated_coords[:, 1] = np.clip(
                        rotated_coords[:, 1], 0, out_shape[0]
                    )
                    rotated_poly.append(rotated_coords.reshape(-1))
                rotated_masks.append(rotated_poly)
            rotated_masks = PolygonMasks(rotated_masks, *out_shape)
        return rotated_masks

    def to_bitmap(self):
        """convert polygon masks to bitmap masks."""
        bitmap_masks = self.to_ndarray()
        return BitmapMasks(bitmap_masks, self.height, self.width)

    @property
    def areas(self):
        """Compute areas of masks.

        This func is modified from `detectron2
        <https://github.com/facebookresearch/detectron2/blob/ffff8acc35ea88ad1cb1806ab0f00b4c1c5dbfd9/detectron2/structures/masks.py#L387>`_.
        The function only works with Polygons using the shoelace formula.

        Return:
            ndarray: areas of each instance
        """  # noqa: W501
        area = []
        for polygons_per_obj in self.masks:
            area_per_obj = 0
            for p in polygons_per_obj:
                area_per_obj += self._polygon_area(p[0::2], p[1::2])
            area.append(area_per_obj)
        return np.asarray(area)

    def _polygon_area(self, x, y):
        """Compute the area of a component of a polygon.

        Using the shoelace formula:
        https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

        Args:
            x (ndarray): x coordinates of the component
            y (ndarray): y coordinates of the component

        Return:
            float: the are of the component
        """  # noqa: 501
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def to_ndarray(self):
        """Convert masks to the format of ndarray."""
        if len(self.masks) == 0:
            return np.empty((0, self.height, self.width), dtype=np.uint8)
        bitmap_masks = []
        for poly_per_obj in self.masks:
            bitmap_masks.append(
                polygon_to_bitmap(poly_per_obj, self.height, self.width)
            )
        return np.stack(bitmap_masks)

    def to_tensor(self, dtype, device):
        """See :func:`BaseInstanceMasks.to_tensor`."""
        if len(self.masks) == 0:
            return torch.empty((0, self.height, self.width), dtype=dtype, device=device)
        ndarray_masks = self.to_ndarray()
        return torch.tensor(ndarray_masks, dtype=dtype, device=device)


def polygon_to_bitmap(polygons, height, width):
    """Convert masks from the form of polygons to bitmaps.

    Args:
        polygons (list[ndarray]): masks in polygon representation
        height (int): mask height
        width (int): mask width

    Return:
        ndarray: the converted masks in bitmap representation
    """
    rles = maskUtils.frPyObjects(polygons, height, width)
    rle = maskUtils.merge(rles)
    bitmap_mask = maskUtils.decode(rle).astype(np.bool)
    return bitmap_mask


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list, cfg):
    """Compute mask target for positive proposals in multiple images.

    Args:
        pos_proposals_list (list[Tensor]): Positive proposals in multiple
            images.
        pos_assigned_gt_inds_list (list[Tensor]): Assigned GT indices for each
            positive proposals.
        gt_masks_list (list[:obj:`BaseInstanceMasks`]): Ground truth masks of
            each image.
        cfg (dict): Config dict that specifies the mask size.

    Returns:
        list[Tensor]: Mask target of each image.
    """
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(
        mask_target_single,
        pos_proposals_list,
        pos_assigned_gt_inds_list,
        gt_masks_list,
        cfg_list,
    )
    mask_targets = list(mask_targets)
    if len(mask_targets) > 0:
        mask_targets = torch.cat(mask_targets)
    return mask_targets


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    """Compute mask target for each positive proposal in the image.

    Args:
        pos_proposals (Tensor): Positive proposals.
        pos_assigned_gt_inds (Tensor): Assigned GT inds of positive proposals.
        gt_masks (:obj:`BaseInstanceMasks`): GT masks in the format of Bitmap
            or Polygon.
        cfg (dict): Config dict that indicate the mask size.

    Returns:
        Tensor: Mask target of each positive proposals in the image.
    """
    device = pos_proposals.device
    mask_size = _pair(cfg["mask_size"])
    num_pos = pos_proposals.size(0)
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        maxh, maxw = gt_masks.height, gt_masks.width
        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

        mask_targets = gt_masks.crop_and_resize(
            proposals_np, mask_size, inds=pos_assigned_gt_inds, device=device
        ).to_ndarray()

        mask_targets = torch.from_numpy(mask_targets).float().to(device)
    else:
        mask_targets = pos_proposals.new_zeros((0,) + mask_size)

    return mask_targets


def split_combined_polys(polys, poly_lens, polys_per_mask):
    """Split the combined 1-D polys into masks.
    A mask is represented as a list of polys, and a poly is represented as
    a 1-D array. In dataset, all masks are concatenated into a single 1-D
    tensor. Here we need to split the tensor into original representations.
    Args:
        polys (list): a list (length = image num) of 1-D tensors
        poly_lens (list): a list (length = image num) of poly length
        polys_per_mask (list): a list (length = image num) of poly number
            of each mask
    Returns:
        list: a list (length = image num) of list (length = mask num) of \
            list (length = poly num) of numpy array.
    """
    mask_polys_list = []
    for img_ind in range(len(polys)):
        polys_single = polys[img_ind]
        polys_lens_single = poly_lens[img_ind].tolist()
        polys_per_mask_single = polys_per_mask[img_ind].tolist()

        split_polys = slice_list(polys_single, polys_lens_single)
        mask_polys = slice_list(split_polys, polys_per_mask_single)
        mask_polys_list.append(mask_polys)
    return mask_polys_list


def encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code.
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    if isinstance(mask_results, tuple):  # mask scoring
        cls_segms, cls_mask_scores = mask_results
    else:
        cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = [[] for _ in range(num_classes)]
    for i in range(len(cls_segms)):
        for cls_segm in cls_segms[i]:
            encoded_mask_results[i].append(
                mask_util.encode(
                    np.array(cls_segm[:, :, np.newaxis], order="F", dtype="uint8")
                )[0]
            )  # encoded with RLE
    if isinstance(mask_results, tuple):
        return encoded_mask_results, cls_mask_scores
    else:
        return encoded_mask_results


def get_map_from_lines(width, height, concat_type, concat_lines, ignore_index=255):
    """get map from split lines"""
    gt_seg_map = np.zeros(shape=(height, width))
    if concat_type == 0:
        # concat vertically
        for concat_line in concat_lines:
            concat_index = int(concat_line)
            gt_seg_map[concat_index, :] = 1
            gt_seg_map[concat_index + 1, :] = 1
            for i in range(1, 11):
                if concat_index - i >= 0:
                    gt_seg_map[concat_index - i, :] = ignore_index
                if concat_index + 1 + i < height - 1:
                    gt_seg_map[concat_index + 1 + i, :] = ignore_index
    else:
        # concat horizontally
        for concat_line in concat_lines:
            concat_index = int(concat_line)
            gt_seg_map[:, concat_index] = 2
            gt_seg_map[:, concat_index + 1] = 2
            for i in range(1, 11):
                if concat_index - i >= 0:
                    gt_seg_map[:, concat_index - i] = ignore_index
                if concat_index + 1 + i < width - 1:
                    gt_seg_map[:, concat_index + 1 + i] = ignore_index
    return gt_seg_map
