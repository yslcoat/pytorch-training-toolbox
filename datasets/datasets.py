import logging

from typing import Protocol
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from datasets.dummy_dataset import DummyDataset
from datasets.TumorSegmentationDataset import TumorSegmentationDataset
from datasets.data_utils import create_train_val_subsets
from utils.configs import (
    DummyDatasetConfig,
    MnistDatasetConfig,
    TumorSegmentationDatasetConfig,
    TrainingConfig,
)


class DatasetBuilder(Protocol):
    def build(self, configs: TrainingConfig, partition: str = "train") -> Dataset:
        ...


class DummyDatasetBuilder(DatasetBuilder):
    def has_partition(
        self,
        dataset_config: DummyDatasetConfig,
        partition: str,
    ) -> bool:
        if partition == "train":
            return dataset_config.n_samples > 0
        if partition == "val":
            return dataset_config.val_n_samples > 0
        return False

    def build(self, configs: TrainingConfig, partition: str = "train") -> Dataset:
        if not isinstance(configs.dataset_config, DummyDatasetConfig):
            raise TypeError(
                "DummyDatasetBuilder expects DummyDatasetConfig, "
                f"got {type(configs.dataset_config)!r}"
            )
        dataset_config = configs.dataset_config

        if not self.has_partition(dataset_config, partition):
            raise ValueError(
                f"Partition '{partition}' is not available for dataset '{configs.dataset}'."
            )

        if partition == "train":
            n_samples = dataset_config.n_samples
        elif partition == "val":
            n_samples = dataset_config.val_n_samples
        else:
            raise ValueError(
                f"Unsupported dataset partition '{partition}' for {configs.dataset}"
            )

        return DummyDataset(
            n_samples=n_samples,
            inputs_tensor_shape=dataset_config.inputs_tensor_shape,
            num_classes=dataset_config.num_classes,
        )


class MnistDatasetBuilder(DatasetBuilder):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            nn.Flatten(start_dim=0),
        ]
    )

    def build(self, configs: TrainingConfig, partition: str = "train") -> Dataset:
        if not isinstance(configs.dataset_config, MnistDatasetConfig):
            raise TypeError(
                "MnistDatasetBuilder expects MnistDatasetConfig, "
                f"got {type(configs.dataset_config)!r}"
            )
        if partition not in {"train", "val", "test"}:
            raise ValueError(
                f"Unsupported dataset partition '{partition}' for {configs.dataset}"
            )

        return torchvision.datasets.MNIST(
            root=configs.dataset_config.root,
            train=partition == "train",
            transform=self.transform,
            target_transform=None,
            download=configs.dataset_config.download,
        )


class _TumorSegmentationEvalTransform:
    def __init__(self, image_height: int, image_width: int):
        self.image_size = [image_height, image_width]

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = TF.resize(
            image,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            self.image_size,
            interpolation=InterpolationMode.NEAREST,
        )
        mask = (mask > 0.5).float()
        return image, mask


class _TumorSegmentationTrainTransform(_TumorSegmentationEvalTransform):
    def __init__(self, dataset_config: TumorSegmentationDatasetConfig):
        super().__init__(
            image_height=dataset_config.image_height,
            image_width=dataset_config.image_width,
        )
        self.hflip_prob = dataset_config.hflip_prob
        self.vflip_prob = dataset_config.vflip_prob
        self.affine_prob = dataset_config.affine_prob
        self.rotate_degrees = dataset_config.rotate_degrees
        self.translate_ratio = dataset_config.translate_ratio
        self.scale_min = dataset_config.scale_min
        self.scale_max = dataset_config.scale_max
        self.color_jitter_prob = dataset_config.color_jitter_prob
        self.brightness = dataset_config.brightness
        self.contrast = dataset_config.contrast

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = super().__call__(image, mask)

        if torch.rand(1).item() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if torch.rand(1).item() < self.vflip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if torch.rand(1).item() < self.affine_prob:
            _, height, width = image.shape
            max_dx = int(round(self.translate_ratio * width))
            max_dy = int(round(self.translate_ratio * height))

            tx = (
                int(torch.randint(-max_dx, max_dx + 1, (1,)).item())
                if max_dx > 0
                else 0
            )
            ty = (
                int(torch.randint(-max_dy, max_dy + 1, (1,)).item())
                if max_dy > 0
                else 0
            )
            angle = float(
                (torch.rand(1).item() * 2.0 - 1.0) * self.rotate_degrees
            )
            scale = float(
                self.scale_min
                + (self.scale_max - self.scale_min) * torch.rand(1).item()
            )

            image = TF.affine(
                image,
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            )

        if torch.rand(1).item() < self.color_jitter_prob:
            if self.brightness > 0.0:
                brightness_factor = float(
                    1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self.brightness
                )
                image = TF.adjust_brightness(image, brightness_factor)
            if self.contrast > 0.0:
                contrast_factor = float(
                    1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self.contrast
                )
                image = TF.adjust_contrast(image, contrast_factor)
            image = image.clamp(0.0, 1.0)

        mask = (mask > 0.5).float()
        return image, mask


class TumorSegmentationDatasetBuilder(DatasetBuilder):
    def build(self, configs: TrainingConfig, partition: str = "train") -> Dataset:
        if not isinstance(configs.dataset_config, TumorSegmentationDatasetConfig):
            raise TypeError(
                "TumorSegmentationDatasetBuilder expects "
                "TumorSegmentationDatasetConfig, "
                f"got {type(configs.dataset_config)!r}"
            )
        if partition not in {"train", "val"}:
            raise ValueError(
                f"Unsupported dataset partition '{partition}' for {configs.dataset}"
            )

        dataset_config = configs.dataset_config
        if partition == "train" and dataset_config.enable_augmentations:
            image_mask_transform = _TumorSegmentationTrainTransform(dataset_config)
        else:
            image_mask_transform = _TumorSegmentationEvalTransform(
                image_height=dataset_config.image_height,
                image_width=dataset_config.image_width,
            )

        full_dataset = TumorSegmentationDataset(
            data_root_dir_path=dataset_config.root,
            image_mask_transform=image_mask_transform,
        )

        train_subset, val_subset = create_train_val_subsets(
            full_dataset,
            val_split=dataset_config.val_split,
            split_seed=dataset_config.split_seed,
        )
        if partition == "train":
            return train_subset
        return val_subset



DATASET_REGISTRY: dict[str, DatasetBuilder] = {
    "DummyDataset": DummyDatasetBuilder(),
    "MNIST": MnistDatasetBuilder(),
    "TumorSegmentation": TumorSegmentationDatasetBuilder(),
}


def create_dataset(
    configs: TrainingConfig,
    partition: str = "train",
) -> Dataset:
    logging.info(
        f"=> creating {partition} dataset '{configs.dataset}'",
    )
    builder = DATASET_REGISTRY.get(configs.dataset)

    if not builder:
        raise ValueError(f"Dataset {configs.dataset} not supported.")

    return builder.build(configs, partition=partition)
