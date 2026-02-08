import argparse
import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict, Any
from dataclasses import dataclass, field

from models.models import MODEL_REGISTRY
from datasets.datasets import DATASET_REGISTRY
from metrics.metrics import METRICS_REGISTRY


@dataclass
class FeedForwardNetworkConfig:
    input_size: int = 784
    n_layers: int = 3
    hidden_dim: int = 256
    output_dim: int = 10
    dropout: float = 0.1


@dataclass
class DummyDatasetConfig:
    n_samples: int = 10000
    inputs_tensor_shape: List[int] = field(default_factory=lambda: [784])
    num_classes: int = 10


@dataclass
class TopKAccuracyConfig:
    top_k: List[int] = field(default_factory=lambda: [1, 5])


@dataclass
class OptimizationConfig:
    epochs: int = 90
    start_epoch: int = 0
    batch_size: int = 256
    lr: float = 5e-4
    momentum: float = 0.9
    warmup_period: int = 10000
    weight_decay: float = 0.05
    mixup: bool = False


@dataclass
class DistributedConfig:
    world_size: int = -1
    rank: int = -1
    dist_url: str = "tcp://224.66.41.62:23456"
    dist_backend: str = "nccl"
    gpu: Optional[int] = None
    no_accel: bool = False
    multiprocessing_distributed: bool = False

    distributed: bool = field(init=False)

    def __post_init__(self):
        self.distributed = self.world_size > 1 or self.multiprocessing_distributed


@dataclass
class LoggingConfig:
    training_id: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_parent_dir: Path = Path("./outputs")
    output_filename: str = "checkpoint.pth.tar"
    print_freq: int = 10
    resume: str = ""
    evaluate: bool = False
    seed: Optional[int] = None

    active_metrics: List[str] = field(default_factory=lambda: ["top_k_accuracy"])

    output_dir: Path = field(init=False)

    def __post_init__(self):
        self.output_dir = self.output_parent_dir / self.training_id


@dataclass
class TrainingConfig:
    optim: OptimizationConfig
    dist: DistributedConfig
    logging: LoggingConfig

    model_config: FeedForwardNetworkConfig
    dataset_config: DummyDatasetConfig

    metrics_config: Dict[str, Any]


def parse_training_configs() -> TrainingConfig:
    parser = argparse.ArgumentParser("Configuration parser for model training")

    optim_group = parser.add_argument_group("Optimization")
    optim_group.add_argument("--epochs", default=90, type=int)
    optim_group.add_argument("--batch-size", default=256, type=int)
    optim_group.add_argument("--lr", default=5e-4, type=float)

    dist_group = parser.add_argument_group("Distributed")
    dist_group.add_argument("--world-size", default=-1, type=int)
    dist_group.add_argument("--gpu", default=None, type=int)

    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--metrics", nargs="+", default=["top_k_accuracy"])
    log_group.add_argument("--resume", default="", type=str)

    selection_group = parser.add_argument_group("Component Selection")
    selection_group.add_argument(
        "--arch", default="FeedForwardNeuralNetwork", choices=MODEL_REGISTRY.keys()
    )
    selection_group.add_argument(
        "--dataset", default="DummyDataset", choices=DATASET_REGISTRY.keys()
    )

    ff_group = parser.add_argument_group("Model: FeedForward")
    ff_group.add_argument("--ff-input-size", default=784, type=int)
    ff_group.add_argument("--ff-n-layers", default=3, type=int)
    ff_group.add_argument("--ff-hidden-dim", default=256, type=int)
    ff_group.add_argument("--ff-output-dim", default=10, type=int)

    dummy_group = parser.add_argument_group("Dataset: Dummy")
    dummy_group.add_argument("--dummy-n-samples", default=10000, type=int)
    dummy_group.add_argument("--dummy-input-shape", default=[784], nargs="+", type=int)

    topk_group = parser.add_argument_group("Metric: TopK")
    topk_group.add_argument("--topk-values", default=[1, 5], nargs="+", type=int)

    args = parser.parse_args()

    if args.arch == "FeedForwardNeuralNetwork":
        model_config = FeedForwardNetworkConfig(
            input_size=args.ff_input_size,
            n_layers=args.ff_n_layers,
            hidden_dim=args.ff_hidden_dim,
            output_dim=args.ff_output_dim,
        )
    else:
        raise ValueError(f"No config defined for arch: {args.arch}")

    if args.dataset == "DummyDataset":
        dataset_config = DummyDatasetConfig(
            n_samples=args.dummy_n_samples,
            inputs_tensor_shape=args.dummy_input_shape,
            num_classes=args.ff_output_dim,
        )
    else:
        raise ValueError(f"No config defined for dataset: {args.dataset}")

    metrics_config_map = {}
    if "top_k_accuracy" in args.metrics:
        metrics_config_map["top_k_accuracy"] = TopKAccuracyConfig(
            top_k=args.topk_values
        )

    return TrainingConfig(
        optim=OptimizationConfig(
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
        ),
        dist=DistributedConfig(world_size=args.world_size, gpu=args.gpu),
        logging=LoggingConfig(resume=args.resume, active_metrics=args.metrics),
        model_config=model_config,
        dataset_config=dataset_config,
        metrics_config=metrics_config_map,
    )
