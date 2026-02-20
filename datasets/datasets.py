import logging

from typing import Protocol
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

from datasets.dummy_dataset import DummyDataset
from datasets.TumorSegmentationDataset import (
    TumorSegmentationDataset,
    TumorSegmentationEvalTransform,
    TumorSegmentationTrainTransform,
)
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
            image_mask_transform = TumorSegmentationTrainTransform(
                image_height=dataset_config.image_height,
                image_width=dataset_config.image_width,
                hflip_prob=dataset_config.hflip_prob,
                vflip_prob=dataset_config.vflip_prob,
                affine_prob=dataset_config.affine_prob,
                rotate_degrees=dataset_config.rotate_degrees,
                translate_ratio=dataset_config.translate_ratio,
                scale_min=dataset_config.scale_min,
                scale_max=dataset_config.scale_max,
                color_jitter_prob=dataset_config.color_jitter_prob,
                brightness=dataset_config.brightness,
                contrast=dataset_config.contrast,
            )
        else:
            image_mask_transform = TumorSegmentationEvalTransform(
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
