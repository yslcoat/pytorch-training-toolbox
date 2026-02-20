import torch
from torch.utils.data import DataLoader, Dataset, Subset

from utils.configs import TrainingConfig


def create_dataloader(
    dataset: Dataset,
    configs: TrainingConfig,
    collate_fn=None,
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

    return DataLoader(
        dataset,
        batch_size=configs.optim.batch_size,
        shuffle=shuffle,
        num_workers=configs.dataloader.num_workers,
        pin_memory=configs.dataloader.pin_memory,
        collate_fn=collate_fn if collate_fn is not None else None,
        sampler=sampler if sampler is not None else None,
    )


def create_train_val_subsets(
    dataset: Dataset,
    val_split: float,
    split_seed: int,
) -> tuple[Subset, Subset]:
    if not 0.0 < val_split < 1.0:
        raise ValueError(
            f"val_split must be in (0.0, 1.0), got {val_split}"
        )

    n_total = len(dataset)
    n_val = int(round(n_total * val_split))
    n_val = max(1, min(n_total - 1, n_val))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(split_seed)
    train_subset, val_subset = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=generator,
    )
    return train_subset, val_subset
