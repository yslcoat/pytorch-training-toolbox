import random
import warnings
import os
import logging

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist


def configure_training_device(configs):
    use_accel = not configs.no_accel and torch.accelerator.is_available()
    if use_accel:
        if configs.gpu is not None:
            torch.accelerator.set_device_index(configs.gpu)
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    return device


def initialize_distributed_mode(gpu, ngpus_per_node, configs):
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
    dist.barrier()


def configure_multi_gpu_model(configs, model, device, ngpus_per_node):
    if configs.distributed:
        if device.type == "cuda":
            if configs.gpu is not None:
                torch.cuda.set_device(configs.gpu)
                model.cuda(device)
                configs.batch_size = int(configs.batch_size / ngpus_per_node)
                configs.workers = int((configs.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[configs.gpu]
                )
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif device.type == "cuda":
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.to(device)


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


def configure_ddp(configs):
    if configs.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if configs.dist_url == "env://" and configs.world_size == -1:
        configs.world_size = int(os.environ["WORLD_SIZE"])

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed

    use_accel = not configs.no_accel and torch.accelerator.is_available()

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    logging.info(f"Using device: {device}")

    if device.type == "cuda":
        ngpus_per_node = torch.accelerator.device_count()
        if ngpus_per_node == 1 and configs.dist_backend == "nccl":
            warnings.warn(
                "nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'"
            )
    else:
        ngpus_per_node = 1

    return ngpus_per_node