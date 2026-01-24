import random
import warnings
import logging
import os
import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import Subset

from trainer import TrainingManager
from models.models import create_model
from utils.configs_parser import (
    TrainingConfig,
    parse_training_configs
)
from utils.torch_utils import (
    configure_multi_gpu_model,
    configure_training_device,
    enable_manual_seed,
    configure_ddp
)


def configure_training(configs):
    if configs.seed_val is not None:
        enable_manual_seed(configs.seed_val)

    ngpus_per_node = configure_ddp(configs)

    if configs.multiprocessing_distributed:
        configs.world_size = ngpus_per_node * configs.world_size
        mp.spawn(main, nprocs=ngpus_per_node, configs=(ngpus_per_node, configs))
    else:
        main(configs.gpu, ngpus_per_node, configs)


def main(gpu, ngpus_per_node, configs):
    # TODO: Improve implementation of training_id generation.
    unique_training_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{configs.arch}_{unique_training_id}"
    global best_acc1
    configs.gpu = gpu

    use_accel = not configs.no_accel and torch.accelerator.is_available()

    if use_accel:
        if configs.gpu is not None:
            torch.accelerator.set_device_index(configs.gpu)
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    if configs.distributed:
        if configs.dist_url == "env://" and configs.rank == -1:
            configs.rank = int(os.environ["RANK"])
        if configs.multiprocessing_distributed:
            configs.rank = configs.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=configs.dist_backend,
            init_method=configs.dist_url,
            world_size=configs.world_size,
            rank=configs.rank,
        )

    logging.info("=> creating model '{}'".format(configs.arch))
    model = create_model(configs)

    if not use_accel:
        logging.info("using CPU, this will be slow")
    else:
        configure_multi_gpu_model(configs, model, device, ngpus_per_node)

    train_loader, val_loader, train_sampler, _ = build_data_loaders(configs)

    criterion = nn.CrossEntropyLoss().to(device)

    total_steps = len(train_loader) * configs.epochs

    optimizer = torch.optim.AdamW(
        model.parameters(), configs.lr, weight_decay=configs.weight_decay
    )

    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - configs.warmup_period, eta_min=1e-6
    )

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, total_iters=configs.warmup_period
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[configs.warmup_period],
    )

    metrics_engine = MetricsEngine(use_accel)

    training_manager = TrainingManager()

    if configs.resume:
        load_checkpoint(configs, device, model, optimizer, scheduler, metrics_engine)

    if configs.evaluate:
        validate(val_loader, model, criterion, metrics_engine, configs)
        return
    
    training_manager.train()


if __name__ == "__main__":
    configs = parse_training_configs()
    configure_training(configs)