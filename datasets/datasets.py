import logging

from typing import Protocol
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

from datasets.dummy_dataset import DummyDataset
from utils.configs import (
    DummyDatasetConfig,
    MnistDatasetConfig,
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


DATASET_REGISTRY: dict[str, DatasetBuilder] = {
    "DummyDataset": DummyDatasetBuilder(),
    "MNIST": MnistDatasetBuilder()
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
