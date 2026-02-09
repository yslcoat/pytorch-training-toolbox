import argparse
import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict, Any
from dataclasses import dataclass, field


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
class DataLoaderConfig:
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True


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

    arch: str 
    dataset: str
    dataloader: DataLoaderConfig

    model_config: FeedForwardNetworkConfig
    dataset_config: DummyDatasetConfig

    metrics_config: Dict[str, Any]