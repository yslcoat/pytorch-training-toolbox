import logging

from typing import Protocol
from torch.utils.data import Dataset

from datasets.dummy_dataset import DummyDataset
from utils.configs import TrainingConfig


class DatasetBuilder(Protocol):
    def has_partition(self, configs: TrainingConfig, partition: str) -> bool:
        ...

    def build(self, configs: TrainingConfig, partition: str = "train") -> Dataset:
        ...


class DummyDatasetBuilder(DatasetBuilder):
    def has_partition(self, configs: TrainingConfig, partition: str) -> bool:
        if partition == "train":
            return configs.dataset_config.n_samples > 0
        if partition == "val":
            return configs.dataset_config.val_n_samples > 0
        return False

    def build(self, configs: TrainingConfig, partition: str = "train") -> Dataset:
        if not self.has_partition(configs, partition):
            raise ValueError(
                f"Partition '{partition}' is not available for dataset '{configs.dataset}'."
            )

        if partition == "train":
            n_samples = configs.dataset_config.n_samples
        elif partition == "val":
            n_samples = configs.dataset_config.val_n_samples
        else:
            raise ValueError(
                f"Unsupported dataset partition '{partition}' for {configs.dataset}"
            )

        return DummyDataset(
            n_samples=n_samples,
            inputs_tensor_shape=configs.dataset_config.inputs_tensor_shape,
            num_classes=configs.dataset_config.num_classes,
        )


DATASET_REGISTRY: dict[str, DatasetBuilder] = {
    "DummyDataset": DummyDatasetBuilder(),
}


def create_dataset(
    configs: TrainingConfig,
    partition: str = "train",
) -> Dataset:
    logging.info(
        "=> creating %s dataset '%s'",
        partition,
        configs.dataset,
    )
    builder = DATASET_REGISTRY.get(configs.dataset)

    if not builder:
        raise ValueError(f"Dataset {configs.dataset} not supported.")

    return builder.build(configs, partition=partition)


def has_dataset_partition(configs: TrainingConfig, partition: str) -> bool:
    builder = DATASET_REGISTRY.get(configs.dataset)
    if not builder:
        raise ValueError(f"Dataset {configs.dataset} not supported.")
    return builder.has_partition(configs, partition)
