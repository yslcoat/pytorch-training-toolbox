import torch
from torch.utils.data import DataLoader

from utils.configs import TrainingConfig


def create_dataloader(dataset, configs: TrainingConfig, collate_fn=None, partition='train') -> DataLoader:
    if configs.dist.distributed:
        if partition == 'train':
            sampler = torch.utils.data.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False, drop_last=True)
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=configs.optim.batch_size,
        shuffle=False if sampler is not None else configs.dataloader.shuffle,
        num_workers=configs.dataloader.num_workers,
        pin_memory=configs.dataloader.pin_memory,
        collate_fn=collate_fn if collate_fn is not None else None,
        sampler=sampler if sampler is not None else None,
    )