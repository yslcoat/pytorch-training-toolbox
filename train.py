import logging
import torch

from trainer import TrainingManager
from criterions.criterions_factory import create_criterion
from models.models import create_model
from optimization.optimizers import create_optimizer
from optimization.schedulers import create_scheduler
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
from utils.utils import configure_process_logging
from metrics.metrics_engine import MetricsEngine


def initialize_training(configs: TrainingConfig):
    if configs.logging.seed is not None:
        enable_manual_seed(configs.logging.seed)

    ngpus_per_node = configure_ddp(configs)

    if configs.dist.multiprocessing_distributed:
        configs.dist.world_size = ngpus_per_node * configs.dist.world_size
        torch.multiprocessing.spawn(
            main,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, configs),
        )
    else:
        main(configs.dist.gpu, ngpus_per_node, configs)


def main(gpu: int | None, ngpus_per_node: int, configs: TrainingConfig):
    if gpu is not None:
        configs.dist.gpu = gpu

    if configs.dist.distributed:
        initialize_distributed_mode(gpu, ngpus_per_node, configs)

    configure_process_logging(configs)

    device = configure_training_device(configs)
    if device is None:
        raise RuntimeError("configure_training_device returned None.")

    model = create_model(configs, device, ngpus_per_node)

    train_dataset = create_dataset(configs, partition="train")
    train_loader = create_dataloader(train_dataset, configs, partition="train")

    val_dataset = create_dataset(configs, partition="val")
    val_loader = create_dataloader(val_dataset, configs, partition="val")

    criterion = create_criterion(configs).to(device)

    optimizer = create_optimizer(configs, model)

    scheduler = None
    if configs.scheduler != "none":
        scheduler = create_scheduler(configs, optimizer, train_loader)

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
