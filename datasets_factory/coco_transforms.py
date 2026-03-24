"""Augmentation utilities for COCO-style detection/segmentation samples.

All transforms follow the contract used in torchvision detection datasets:
`transform(image, target) -> (image, target)` where:
- `image` is a PIL image during geometric preprocessing, then a tensor at output.
- `target` is a dict with keys like `boxes`, `labels`, and optionally `masks`.

Key rules implemented here:
- Keep aspect ratio with deterministic letterbox padding.
- Apply the exact same geometric transform to image and labels.
- Use separate interpolation: bilinear for image, nearest for masks/segmentation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def _as_hw(size: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    if len(size) != 2:
        raise ValueError(f"size must be int or (h, w), got {size!r}")
    h, w = size
    return int(h), int(w)


def _clip_boxes_to_bounds(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0].clamp(0.0, float(width))
    boxes[:, 2] = boxes[:, 2].clamp(0.0, float(width))
    boxes[:, 1] = boxes[:, 1].clamp(0.0, float(height))
    boxes[:, 3] = boxes[:, 3].clamp(0.0, float(height))
    return boxes


def _filter_degenerate_boxes(target: dict) -> dict:
    boxes = target["boxes"]
    if boxes.numel() == 0:
        if "area" in target:
            target["area"] = torch.zeros((0,), dtype=torch.float32, device=boxes.device)
        return target

    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)
    keep = (widths > 1e-6) & (heights > 1e-6)

    target["boxes"] = boxes[keep]
    target["labels"] = target["labels"][keep]
    target["iscrowd"] = target["iscrowd"][keep]
    if "area" in target:
        filtered_boxes = target["boxes"]
        target["area"] = (
            (filtered_boxes[:, 2] - filtered_boxes[:, 0]).clamp(min=0.0)
            * (filtered_boxes[:, 3] - filtered_boxes[:, 1]).clamp(min=0.0)
        )

    if "masks" in target and isinstance(target["masks"], torch.Tensor):
        target["masks"] = target["masks"][keep]

    return target


class _Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, image: Image.Image, target: dict):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class Letterbox:
    """Resize while preserving aspect ratio, then pad to fixed output size."""

    def __init__(
        self,
        output_size: int | Tuple[int, int],
        fill: int | Tuple[int, int, int] = (114, 114, 114),
        image_interpolation: TF.InterpolationMode = TF.InterpolationMode.BILINEAR,
        antialias: bool = True,
    ) -> None:
        self.output_size = output_size
        self.fill = fill
        self.image_interpolation = image_interpolation
        self.antialias = antialias

    def __call__(self, image: Image.Image, target: dict):
        target = dict(target)
        out_h, out_w = _as_hw(self.output_size)
        in_w, in_h = image.size

        scale = min(out_w / in_w, out_h / in_h)
        new_w = max(1, int(round(in_w * scale)))
        new_h = max(1, int(round(in_h * scale)))

        pad_w = out_w - new_w
        pad_h = out_h - new_h
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top

        image = TF.resize(image, size=[new_h, new_w], interpolation=self.image_interpolation, antialias=self.antialias)
        image = TF.pad(image, padding=[left, top, right, bottom], fill=self.fill, padding_mode="constant")

        boxes = target["boxes"]
        if boxes.numel() > 0:
            boxes = boxes * scale
            boxes[:, 0::2] += torch.tensor([left, left], dtype=boxes.dtype, device=boxes.device)
            boxes[:, 1::2] += torch.tensor([top, top], dtype=boxes.dtype, device=boxes.device)
            boxes = _clip_boxes_to_bounds(boxes, out_w, out_h)
            target["boxes"] = boxes

        masks = target.get("masks")
        if isinstance(masks, torch.Tensor) and masks.numel() > 0:
            # Nearest-neighbor for class masks / segmentation logits
            masks = F.interpolate(
                masks.unsqueeze(1).float(),
                size=(new_h, new_w),
                mode="nearest",
            )
            masks = masks.squeeze(1)
            masks = F.pad(masks, (left, right, top, bottom), mode="constant", value=0)
            target["masks"] = masks.to(torch.long)

        target["size"] = torch.tensor([out_h, out_w], dtype=torch.int64)
        target["orig_size"] = torch.tensor([in_h, in_w], dtype=torch.int64)
        target["scale"] = torch.tensor([scale], dtype=torch.float32)
        target["pad"] = torch.tensor([top, left], dtype=torch.int64)

        return image, _filter_degenerate_boxes(target)


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, target: dict):
        if random.random() > self.p:
            return image, target

        w = image.width
        image = TF.hflip(image)

        target = dict(target)
        boxes = target["boxes"]
        if boxes.numel() > 0:
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            boxes[:, 0] = w - x2
            boxes[:, 2] = w - x1
            target["boxes"] = boxes

        masks = target.get("masks")
        if isinstance(masks, torch.Tensor) and masks.numel() > 0:
            target["masks"] = torch.flip(masks, dims=[2])

        return image, target


class RandomPhotometric:
    """Safe color jitter for image-only augmentation."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        p: float = 0.5,
    ) -> None:
        self.p = p
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, image: Image.Image, target: dict):
        if random.random() < self.p:
            image = self.jitter(image)
        return image, target


class MultiScaleLetterbox:
    """Pick a random square output side and apply letterbox per batch/sample."""

    def __init__(
        self,
        min_size: int = 320,
        max_size: int = 640,
        stride: int = 32,
        fill: int | Tuple[int, int, int] = (114, 114, 114),
    ) -> None:
        if min_size > max_size:
            raise ValueError("min_size must be <= max_size")
        if stride <= 0:
            raise ValueError("stride must be > 0")
        self.min_size = min_size
        self.max_size = max_size
        self.stride = stride
        self.fill = fill

    def __call__(self, image: Image.Image, target: dict):
        min_v = self.min_size
        max_v = self.max_size
        stride = self.stride
        choices = list(range(min_v, max_v + 1, stride))
        if not choices:
            choices = [min_v]
        size = random.choice(choices)
        return Letterbox((size, size), fill=self.fill)(image, target)


class ToTensorNormalize:
    """Convert image to CHW float tensor and normalize for training."""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.normalize = T.Normalize(mean=mean, std=std)

    def __call__(self, image: Image.Image, target: dict):
        image_t = TF.to_tensor(image)
        image_t = self.normalize(image_t)
        return image_t, target


@dataclass
class CocoTransformConfig:
    multiscale: bool = False
    fixed_size: int = 640
    min_size: int = 320
    max_size: int = 640
    stride: int = 32
    enable_flip: bool = True
    enable_color_jitter: bool = True
    fill: int | Tuple[int, int, int] = (114, 114, 114)


def build_train_transform(config: CocoTransformConfig):
    transforms: List[Callable] = []

    if config.enable_color_jitter:
        transforms.append(RandomPhotometric(p=0.6))

    if config.multiscale:
        transforms.append(MultiScaleLetterbox(config.min_size, config.max_size, config.stride, fill=config.fill))
    else:
        transforms.append(Letterbox(config.fixed_size, fill=config.fill))

    if config.enable_flip:
        transforms.append(RandomHorizontalFlip(p=0.5))

    transforms.append(ToTensorNormalize())
    return _Compose(transforms)
