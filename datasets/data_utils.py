import torch
from torch.utils.data import DataLoader


def create_dataloader(dataset, configs, collate_fn=None, partition='train') -> DataLoader:
    if configs.distributed:
        if partition == 'train':
            sampler = torch.utils.data.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False, drop_last=True)
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=configs.batch_size,
        shuffle=configs.shuffle,
        num_workers=configs.num_workers,
        pin_memory=configs.pin_memory
    )