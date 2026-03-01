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
class VisionTransformerConfig:
    image_size: tuple[int, int] | int = 224
    n_channels: int = 3
    patch_size: int = 16
    num_classes: int = 1000
    emb_dim: int = 768
    n_heads: int = 12
    n_blocks: int = 12
    attn_head_dim: int = 64
    dropout: float = 0.1
    emb_dropout: float = 0.1


@dataclass
class DummyDatasetConfig:
    n_samples: int = 10000
    val_n_samples: int = 0
    inputs_tensor_shape: list[int] = field(default_factory=lambda: [784])
    num_classes: int = 10

    def __post_init__(self):
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be > 0, got {self.n_samples}")
        if self.val_n_samples < 0:
            raise ValueError(
                f"val_n_samples must be >= 0, got {self.val_n_samples}"
            )


@dataclass
class MnistDatasetConfig:
    root: Path = Path("/home/yslcoat/data")
    download: bool = True


@dataclass
class ImageNetDatasetConfig:
    root: Path = Path("./data")
    object_detection: bool = False


@dataclass
class DataAugmentationConfig:
    randaug_num_ops: int = 2
    randaug_magnitude: int = 9

    def __post_init__(self):
        if self.randaug_num_ops < 0:
            raise ValueError(
                f"randaug_num_ops must be >= 0, got {self.randaug_num_ops}"
            )
        if not 0 <= self.randaug_magnitude <= 30:
            raise ValueError(
                "randaug_magnitude must be in [0, 30], "
                f"got {self.randaug_magnitude}"
            )


@dataclass
class TopKAccuracyConfig:
    top_k: list[int] = field(default_factory=lambda: [1, 5])


@dataclass
class DiceScoreConfig:
    smooth: float = 1e-6
    from_logits: bool = True
    threshold: float = 0.5

    def __post_init__(self):
        if self.smooth < 0.0:
            raise ValueError(f"smooth must be >= 0.0, got {self.smooth}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                f"threshold must be in [0.0, 1.0], got {self.threshold}"
            )


@dataclass
class BBoxIoUScoreConfig:
    smooth: float = 1e-6
    from_logits: bool = False
    box_format: str = "xyxy"
    reduction: str = "mean"

    def __post_init__(self):
        if self.smooth < 0.0:
            raise ValueError(f"smooth must be >= 0.0, got {self.smooth}")
        if self.box_format not in {"xyxy", "cxcywh"}:
            raise ValueError(
                "box_format must be one of ['xyxy', 'cxcywh'], "
                f"got {self.box_format}"
            )
        if self.reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                "reduction must be one of ['none', 'mean', 'sum'], "
                f"got {self.reduction}"
            )


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
class DiceLossConfigs(CriterionConfigs):
    smooth: float = 1e-6
    from_logits: bool = True
    reduction: str = "mean"

    def __post_init__(self):
        if self.smooth < 0.0:
            raise ValueError(f"smooth must be >= 0.0, got {self.smooth}")
        if self.reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"reduction must be one of ['none', 'mean', 'sum'], got {self.reduction}"
            )


@dataclass
class OptimizerConfigs:
    pass


@dataclass
class AdamWConfigs(OptimizerConfigs):
    lr: float = 5e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.05
    amsgrad: bool = False

    def __post_init__(self):
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0.0, got {self.lr}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0.0, got {self.eps}")
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be >= 0.0, got {self.weight_decay}"
            )

        beta1, beta2 = self.betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0.0, 1.0), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0.0, 1.0), got {beta2}")


@dataclass
class SchedulerConfigs:
    pass


@dataclass
class LinearLRConfigs(SchedulerConfigs):
    start_factor: float = 0.01
    end_factor: float = 1.0
    total_iters: int | None = None

    def __post_init__(self):
        if self.start_factor <= 0.0:
            raise ValueError(
                f"start_factor must be > 0.0, got {self.start_factor}"
            )
        if self.end_factor <= 0.0:
            raise ValueError(f"end_factor must be > 0.0, got {self.end_factor}")
        if self.total_iters is not None and self.total_iters <= 0:
            raise ValueError(
                f"total_iters must be > 0 when provided, got {self.total_iters}"
            )


@dataclass
class CosineAnnealingLRConfigs(SchedulerConfigs):
    eta_min: float = 1e-6
    t_max: int | None = None

    def __post_init__(self):
        if self.eta_min < 0.0:
            raise ValueError(f"eta_min must be >= 0.0, got {self.eta_min}")
        if self.t_max is not None and self.t_max <= 0:
            raise ValueError(f"t_max must be > 0 when provided, got {self.t_max}")


@dataclass
class LinearThenCosineAnnealingLRConfigs(SchedulerConfigs):
    linear_start_factor: float = 0.01
    linear_end_factor: float = 1.0
    warmup_iters: int | None = None
    cosine_eta_min: float = 1e-6
    cosine_t_max: int | None = None

    def __post_init__(self):
        if self.linear_start_factor <= 0.0:
            raise ValueError(
                "linear_start_factor must be > 0.0, "
                f"got {self.linear_start_factor}"
            )
        if self.linear_end_factor <= 0.0:
            raise ValueError(
                f"linear_end_factor must be > 0.0, got {self.linear_end_factor}"
            )
        if self.warmup_iters is not None and self.warmup_iters <= 0:
            raise ValueError(
                f"warmup_iters must be > 0 when provided, got {self.warmup_iters}"
            )
        if self.cosine_eta_min < 0.0:
            raise ValueError(
                f"cosine_eta_min must be >= 0.0, got {self.cosine_eta_min}"
            )
        if self.cosine_t_max is not None and self.cosine_t_max <= 0:
            raise ValueError(
                f"cosine_t_max must be > 0 when provided, got {self.cosine_t_max}"
            )


@dataclass
class OptimizationConfig:
    epochs: int = 90
    start_epoch: int = 0
    batch_size: int = 256
    momentum: float = 0.9
    warmup_iters: int = 0
    scheduler_step_unit: str = "step"
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
    optimizer: str
    scheduler: str

    arch: str
    dataset: str
    dataloader: DataLoaderConfig

    model_config: (
        FeedForwardNetworkConfig
        | VisionTransformerConfig
    )
    dataset_config: (
        DummyDatasetConfig
        | MnistDatasetConfig
        | ImageNetDatasetConfig
    )
    data_augmentation: DataAugmentationConfig

    criterion_config: CriterionConfigs | None
    optimizer_config: OptimizerConfigs | None
    scheduler_config: SchedulerConfigs | None
    metrics_config: Dict[str, Any]
