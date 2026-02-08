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
    if configs.seed is not None:
        enable_manual_seed(configs.seed)

    ngpus_per_node = configure_ddp(configs)

    if configs.multiprocessing_distributed:
        configs.world_size = ngpus_per_node * configs.world_size
        mp.spawn(main, nprocs=ngpus_per_node, configs=(ngpus_per_node, configs))
    else:
        main(configs.gpu, ngpus_per_node, configs)


def main(gpu, ngpus_per_node: int, configs: TrainingConfig):
    configs.gpu = gpu

    use_accel = not configs.no_accel and torch.accelerator.is_available()

    device = configure_training_device(configs)

    if configs.distributed:
        initialize_distributed_mode(gpu, ngpus_per_node, configs)

    model = create_model(configs, device, ngpus_per_node)

    train_dataset = create_dataset(configs)
    val_dataset = create_dataset(configs)

    train_loader = create_dataloader(train_dataset, configs)
    val_loader = create_dataloader(val_dataset, configs, partition='val')

    criterion = nn.CrossEntropyLoss().to(device) # Maybe create util function with a registry of different loss functions instead of hardcoding, we'll see.

    optimizer = torch.optim.AdamW(
        model.parameters(), configs.lr, weight_decay=configs.weight_decay
    ) # Same here as criterion

    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * configs.epochs - configs.warmup_period, eta_min=1e-6
    ) # Same here as criterion

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, total_iters=configs.warmup_period
    ) # Same here as criterion

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[configs.warmup_period],
    ) # Here too, should have some logic to simplify the entire optimizer setup tbh. 

    metrics_engine = MetricsEngine(configs) # Wonder if use_accel can be set in the config class. Need utility function for setting list of metric functions.

    training_manager = TrainingManager(
        configs=configs,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        metrics_engine=metrics_engine,
        local_rank=device,
        scheduler=scheduler
    )

    if configs.resume:
        training_manager.load_checkpoint()
    
    training_manager.train()


if __name__ == "__main__":
    configs = parse_training_configs()
    initialize_training(configs)