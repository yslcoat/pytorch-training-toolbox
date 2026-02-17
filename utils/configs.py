import datetime
from pathlib import Path
from typing import Optional, Dict, Any
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
    inputs_tensor_shape: list[int] = field(default_factory=lambda: [784])
    num_classes: int = 10


@dataclass
class TopKAccuracyConfig:
    top_k: list[int] = field(default_factory=lambda: [1, 5])


@dataclass
class CriterionConfigs:
    pass


@dataclass
class CrossEntropyLossConfigs(CriterionConfigs):
    label_smoothing: float = 0.0
    ignore_index: int = -100
    reduction: str = "mean"

    def __post_init__(self):
        if not 0.0 <= self.label_smoothing <= 1.0:
            raise ValueError(
                f"label_smoothing must be in [0.0, 1.0], got {self.label_smoothing}"
            )
        if self.reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"reduction must be one of ['none', 'mean', 'sum'], got {self.reduction}"
            )


@dataclass
class OptimizationConfig:
    epochs: int = 90
    start_epoch: int = 0
    batch_size: int = 256
    lr: float = 5e-4
    momentum: float = 0.9
    warmup_iters: int = 0
    scheduler_step_unit: str = "step"
    weight_decay: float = 0.05
    mixup: bool = False

    def __post_init__(self):
        if self.epochs < 0:
            raise ValueError(f"epochs must be >= 0, got {self.epochs}")
        if self.start_epoch < 0:
            raise ValueError(f"start_epoch must be >= 0, got {self.start_epoch}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.warmup_iters < 0:
            raise ValueError(f"warmup_iters must be >= 0, got {self.warmup_iters}")
        if self.scheduler_step_unit not in {"step", "epoch"}:
            raise ValueError(
                f"scheduler_step_unit must be one of ['step', 'epoch'], got {self.scheduler_step_unit}"
            )


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

    active_metrics: list[str] = field(default_factory=lambda: ["top_1_accuracy", "top_5_accuracy"])

    output_dir: Path = field(init=False)

    def __post_init__(self):
        self.output_dir = self.output_parent_dir / self.training_id


@dataclass
class TrainingConfig:
    optim: OptimizationConfig
    dist: DistributedConfig
    logging: LoggingConfig

    criterion: str

    arch: str 
    dataset: str
    dataloader: DataLoaderConfig

    model_config: FeedForwardNetworkConfig
    dataset_config: DummyDatasetConfig

    criterion_config: CriterionConfigs | None
    metrics_config: Dict[str, Any]
