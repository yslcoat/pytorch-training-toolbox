import random
import warnings
import os
import logging

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from .configs import *


def configure_training_device(configs: TrainingConfig):
    use_accel = not configs.dist.no_accel and torch.accelerator.is_available()
    if use_accel:
        if configs.dist.gpu is not None:
            torch.accelerator.set_device_index(configs.dist.gpu)
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    return device


def initialize_distributed_mode(gpu, ngpus_per_node, configs: TrainingConfig):
    if configs.dist.dist_url == "env://" and configs.dist.rank == -1:
            configs.dist.rank = int(os.environ["RANK"])
    if configs.dist.multiprocessing_distributed:
        configs.dist.rank = configs.dist.rank * ngpus_per_node + gpu
    dist.init_process_group(
        backend=configs.dist.dist_backend,
        init_method=configs.dist.dist_url,
        world_size=configs.dist.world_size,
        rank=configs.dist.rank,
    )
    dist.barrier()


def configure_multi_gpu_model(configs: TrainingConfig, model, device, ngpus_per_node):
    if configs.dist.distributed:
        if device.type == "cuda":
            if configs.dist.gpu is not None:
                torch.cuda.set_device(configs.dist.gpu)
                model.cuda(device)
                configs.optim.batch_size = int(configs.optim.batch_size / ngpus_per_node)
                configs.dataloader.num_workers = int((configs.dataloader.num_workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[configs.dist.gpu]
                )
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif device.type == "cuda":
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.to(device)

    return model


def enable_manual_seed(seed_val):
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    cudnn.deterministic = True
    cudnn.benchmark = False
    warnings.warn(
        "You have chosen to seed training. "
        "This will turn on the CUDNN deterministic setting, "
        "which can slow down your training considerably! "
        "You may see unexpected behavior when restarting "
        "from checkpoints."
    )


def configure_ddp(configs: TrainingConfig):
    if configs.dist.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if configs.dist.dist_url == "env://" and configs.dist.world_size == -1:
        configs.dist.world_size = int(os.environ["WORLD_SIZE"])

    configs.dist.distributed = configs.dist.world_size > 1 or configs.dist.multiprocessing_distributed

    use_accel = not configs.dist.no_accel and torch.accelerator.is_available()

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    logging.info(f"Using device: {device}")

    if device.type == "cuda":
        ngpus_per_node = torch.accelerator.device_count()
        if ngpus_per_node == 1 and configs.dist.dist_backend == "nccl":
            warnings.warn(
                "nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'"
            )
    else:
        ngpus_per_node = 1

    return ngpus_per_node