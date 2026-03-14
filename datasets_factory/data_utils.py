import torch
from torch.utils.data import DataLoader, Dataset
from collections.abc import Callable

from utils.configs import TrainingConfig
from .collate_functions import MixUpCollator


def _infer_num_classes(configs: TrainingConfig) -> int:
    return configs.num_classes


def _build_mixup_collator(configs: TrainingConfig) -> MixUpCollator:
    return MixUpCollator(num_classes=_infer_num_classes(configs))


COLLATE_FN_REGISTRY: dict[str, Callable[[TrainingConfig], object]] = {
    "mixup": _build_mixup_collator
}


def create_dataloader(
    dataset: Dataset,
    configs: TrainingConfig,
    partition: str = "train",
) -> DataLoader:

    if configs.dist.distributed:
        if partition == "train":
            sampler = torch.utils.data.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None

    shuffle = (
        False
        if sampler is not None
        else (configs.dataloader.shuffle if partition == "train" else False)
    )

    collate_fn_name = configs.dataloader.collate_fn
    collate_fn_builder = (
        COLLATE_FN_REGISTRY[collate_fn_name]
        if collate_fn_name in COLLATE_FN_REGISTRY
        else None
    )
    collate_fn = (
        collate_fn_builder(configs) if collate_fn_builder is not None else None
    )

    return DataLoader(
        dataset,
        batch_size=configs.optim.batch_size,
        shuffle=shuffle,
        num_workers=configs.dataloader.num_workers,
        pin_memory=configs.dataloader.pin_memory,
        collate_fn=collate_fn if collate_fn is not None else None,
        sampler=sampler if sampler is not None else None,
    )
