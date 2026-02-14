import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from trainer import TrainingManager
from models.models import create_model
from datasets.datasets import create_dataset
from datasets.data_utils import create_dataloader
from utils.configs_parser import (
    TrainingConfig,
    parse_training_configs
)
from utils.torch_utils import (
    configure_training_device,
    initialize_distributed_mode,
    enable_manual_seed,
    configure_ddp
)
from metrics.metrics_engine import MetricsEngine


def initialize_training(configs: TrainingConfig):
    if configs.logging.seed is not None:
        enable_manual_seed(configs.logging.seed)

    ngpus_per_node = configure_ddp(configs)

    if configs.dist.multiprocessing_distributed:
        configs.dist.world_size = ngpus_per_node * configs.dist.world_size
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, configs))
    else:
        main(configs.dist.gpu, ngpus_per_node, configs)


def main(gpu, ngpus_per_node: int, configs: TrainingConfig):
    configs.dist.gpu = gpu

    device = configure_training_device(configs)

    if configs.dist.distributed:
        initialize_distributed_mode(gpu, ngpus_per_node, configs)

    model = create_model(configs, device, ngpus_per_node)

    train_dataset = create_dataset(configs)
    val_dataset = create_dataset(configs)

    train_loader = create_dataloader(train_dataset, configs)
    val_loader = create_dataloader(val_dataset, configs, partition='val')

    criterion = nn.CrossEntropyLoss().to(device) # Maybe create util function with a registry of different loss functions instead of hardcoding, we'll see.

    optimizer = torch.optim.AdamW(
        model.parameters(), configs.optim.lr, weight_decay=configs.optim.weight_decay
    ) # Same here as criterion

    if configs.optim.scheduler_step_unit == "epoch":
        total_scheduler_iters = configs.optim.epochs
    else:
        total_scheduler_iters = configs.optim.epochs * len(train_loader)

    warmup_iters = configs.optim.warmup_iters
    if warmup_iters < 0:
        raise ValueError(f"warmup_iters must be >= 0, got {warmup_iters}")
    if total_scheduler_iters > 0 and warmup_iters >= total_scheduler_iters:
        raise ValueError(
            "warmup_iters must be smaller than total scheduler iterations. "
            f"Got warmup_iters={warmup_iters}, total_iters={total_scheduler_iters}, "
            f"scheduler_step_unit={configs.optim.scheduler_step_unit}"
        )

    scheduler = None
    if total_scheduler_iters > 0:
        cosine_iters = total_scheduler_iters - warmup_iters
        if warmup_iters == 0:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=cosine_iters, eta_min=1e-6
            )
        else:
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_iters
            )
            main_scheduler = CosineAnnealingLR(
                optimizer, T_max=cosine_iters, eta_min=1e-6
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_iters],
            )

    metrics_engine = MetricsEngine(configs)
    
    training_manager = TrainingManager(
        configs=configs,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        metrics_engine=metrics_engine,
        local_rank=configs.dist.gpu if configs.dist.gpu is not None else 0,
        device=device,
        scheduler=scheduler
    )

    if configs.logging.resume:
        training_manager.load_checkpoint()
    
    training_manager.train()


if __name__ == "__main__":
    configs = parse_training_configs()
    initialize_training(configs)
