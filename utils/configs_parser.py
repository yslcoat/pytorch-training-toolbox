import argparse
import datetime
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    training_id: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_parent_dir: Path = Path("./outputs")
    output_dir: Path = field(init=False)
    output_filename: str = "checkpoint.pth.tar"
    seed: Optional[int] = None
    print_freq: int = 10
    resume: str = ""
    evaluate: bool = False

    dummy_data: bool = False
    data_dir: Optional[Path] = None
    workers: int = 4

    epochs: int = 90
    start_epoch: int = 0
    batch_size: int = 256
    lr: float = 5e-4
    momentum: float = 0.9
    warmup_period: int = 10000
    weight_decay: float = 0.05

    arch: str = "resnet18"
    pretrained: bool = False

    image_size: int = 224
    patch_size: int = 16
    num_classes: int = 1000
    dim: int = 1024
    depth: int = 6
    heads: int = 16
    mlp_dim: int = 2048
    dropout: float = 0.1
    emb_dropout: float = 0.1

    mixup: bool = False
    randaug_num_ops: int = 2
    randaug_magnitude: int = 9

    gpu: Optional[int] = None
    no_accel: bool = False
    world_size: int = -1
    rank: int = -1
    dist_url: str = "tcp://224.66.41.62:23456"
    dist_backend: str = "nccl"
    multiprocessing_distributed: bool = False
    distributed = world_size > 1 or multiprocessing_distributed

    def __post_init__(self):
        """
        Might use this for validation of inputs, will see
        """
        self.output_dir = self.output_parent_dir / self.training_id

        if not self.dummy_data and self.data_dir is None:
            pass


def parse_training_configs() -> TrainingConfig:
    parser = argparse.ArgumentParser("Configuration parser for model training")

    model_names = ["resnet18", "vit", "feedforward"]

    # Data configs
    data_group = parser.add_argument_group("Data Settings")
    source_group = data_group.add_mutually_exclusive_group()
    source_group.add_argument(
        "--dummy_data", action="store_true", help="Specify to use dummy data."
    )
    source_group.add_argument(
        "-d", "--data_dir", type=Path, help="Directory containing training data."
    )
    data_group.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )

    # Model configs
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="resnet18",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
    )
    model_group.add_argument(
        "--pretrained", action="store_true", help="use pre-trained model"
    )

    # ViT configs
    model_group.add_argument(
        "--image_size", type=int, default=224, help="size of input image to vit."
    )
    model_group.add_argument(
        "--patch_size", type=int, default=16, help="size of patches."
    )
    model_group.add_argument(
        "--num_classes", type=int, default=1000, help="number of classes in dataset."
    )
    model_group.add_argument(
        "--dim", type=int, default=1024, help="last dimension of output tensor."
    )
    model_group.add_argument(
        "--depth", type=int, default=6, help="number of transformer blocks."
    )
    model_group.add_argument(
        "--heads", type=int, default=16, help="number of heads in multi-head attention."
    )
    model_group.add_argument(
        "--mlp_dim", type=int, default=2048, help="dimension of mlp."
    )
    model_group.add_argument("--dropout", type=float, default=0.1, help="dropout rate.")
    model_group.add_argument(
        "--emb_dropout", type=float, default=0.1, help="embedding dropout rate."
    )

    # Optimization/training configs
    optim_group = parser.add_argument_group("Optimization & Training")
    optim_group.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    optim_group.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="manual epoch number"
    )
    optim_group.add_argument(
        "-b", "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )
    optim_group.add_argument(
        "--lr",
        "--learning-rate",
        default=5e-4,
        type=float,
        metavar="LR",
        dest="lr",
        help="initial learning rate",
    )
    optim_group.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    optim_group.add_argument(
        "--warmup_period", type=int, default=10000, help="number of warmup steps"
    )
    optim_group.add_argument(
        "--wd",
        "--weight-decay",
        default=0.05,
        type=float,
        metavar="W",
        dest="weight_decay",
        help="weight decay",
    )
    optim_group.add_argument(
        "--mixup", action="store_true", help="applies mixup augmentation"
    )
    optim_group.add_argument(
        "--randaug_num_ops", type=int, default=2, help="RandAugment number of ops"
    )
    optim_group.add_argument(
        "--randaug_magnitude", type=int, default=9, help="RandAugment magnitude"
    )

    # Logging configs
    log_group = parser.add_argument_group("Logging & Output")
    log_group.add_argument(
        "-p", "--print-freq", default=10, type=int, metavar="N", help="print frequency"
    )
    log_group.add_argument(
        "-o",
        "--output_parent_dir",
        default=Path("./outputs"),
        type=Path,
        help="parent dir for storage",
    )
    log_group.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint",
    )
    log_group.add_argument(
        "-e", "--evaluate", action="store_true", help="evaluate model on validation set"
    )

    # Distributed training configs
    dist_group = parser.add_argument_group("Distributed Training")
    dist_group.add_argument(
        "--world-size", default=-1, type=int, help="number of nodes"
    )
    dist_group.add_argument("--rank", default=-1, type=int, help="node rank")
    dist_group.add_argument(
        "--dist-url",
        default="tcp://224.66.41.62:23456",
        type=str,
        help="url for distributed training",
    )
    dist_group.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    dist_group.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training."
    )
    dist_group.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    dist_group.add_argument(
        "--no-accel", action="store_true", help="disables accelerator"
    )
    dist_group.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training",
    )

    args = parser.parse_args()

    return TrainingConfig(**vars(args))
