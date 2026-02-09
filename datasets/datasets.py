import logging

from typing import Protocol
import torch.nn as nn
from torch.utils.data import Dataset

from datasets.dummy_dataset import DummyDataset
from utils.configs import TrainingConfig

class DatasetBuilder(Protocol):
    def build(self, configs: TrainingConfig) -> Dataset:
        ...


class DummyDatasetBuilder(DatasetBuilder):
    def build(self, configs: TrainingConfig) -> Dataset:
        return DummyDataset(
            n_samples=configs.dataset_config.n_samples,
            inputs_tensor_shape=configs.dataset_config.inputs_tensor_shape,
            num_classes=configs.dataset_config.num_classes,
        )


DATASET_REGISTRY: dict[str, DatasetBuilder] = {
    "DummyDataset": DummyDatasetBuilder(),
}


def create_dataset(configs: TrainingConfig) -> Dataset:
    logging.info("=> creating dataset '{}'".format(configs.dataset))
    builder = DATASET_REGISTRY.get(configs.dataset)

    if not builder:
        raise ValueError(f"Dataset {configs.dataset} not supported.")

    return builder.build(configs)

