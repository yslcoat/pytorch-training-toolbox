import argparse

from models.models import MODEL_REGISTRY
from datasets.datasets import DATASET_REGISTRY
from metrics.metrics import METRICS_REGISTRY
from criterions.criterions_factory import CRITERIONS_REGISTRY


from utils.configs import (
    TrainingConfig, 
    FeedForwardNetworkConfig, 
    DummyDatasetConfig, 
    TopKAccuracyConfig,
    CriterionConfigs,
    CrossEntropyLossConfigs,
    OptimizationConfig,
    DistributedConfig,
    DataLoaderConfig,
    LoggingConfig
)


def parse_training_configs() -> TrainingConfig:
    parser = argparse.ArgumentParser("Configuration parser for model training")

    optim_group = parser.add_argument_group("Optimization")
    optim_group.add_argument("--epochs", default=90, type=int)
    optim_group.add_argument("--batch-size", default=256, type=int)
    optim_group.add_argument("--lr", default=5e-4, type=float)
    optim_group.add_argument("--warmup-iters", default=0, type=int)
    optim_group.add_argument(
        "--scheduler-step-unit",
        default="step",
        choices=["step", "epoch"],
    )

    dist_group = parser.add_argument_group("Distributed")
    dist_group.add_argument("--world-size", default=-1, type=int)
    dist_group.add_argument("--rank", default=-1, type=int)
    dist_group.add_argument("--dist-url", default="tcp://224.66.41.62:23456", type=str)
    dist_group.add_argument("--dist-backend", default="nccl", type=str)
    dist_group.add_argument("--multiprocessing-distributed", action="store_true")
    dist_group.add_argument("--gpu", default=None, type=int)

    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--metrics", nargs="+", default=["top_1_accuracy", "top_5_accuracy"], choices=METRICS_REGISTRY.keys())
    log_group.add_argument("--resume", default="", type=str)

    selection_group = parser.add_argument_group("Component Selection")
    selection_group.add_argument(
        "--arch", default="FeedForwardNeuralNetwork", choices=MODEL_REGISTRY.keys()
    )
    selection_group.add_argument(
        "--dataset", default="DummyDataset", choices=DATASET_REGISTRY.keys()
    )
    selection_group.add_argument(
        "--criterion", default="cross_entropy_loss", choices=CRITERIONS_REGISTRY.keys()
    )

    dataloader_group = parser.add_argument_group("DataLoader")
    dataloader_group.add_argument(
        "--dataloader-shuffle",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    dataloader_group.add_argument("--dataloader-num-workers", default=4, type=int)
    dataloader_group.add_argument(
        "--dataloader-pin-memory",
        default=True,
        action=argparse.BooleanOptionalAction,
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

    ce_group = parser.add_argument_group("Criterion: CrossEntropyLoss")
    ce_group.add_argument("--ce-label-smoothing", default=0.0, type=float)
    ce_group.add_argument("--ce-ignore-index", default=-100, type=int)
    ce_group.add_argument(
        "--ce-reduction",
        default="mean",
        choices=["none", "mean", "sum"],
    )

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
    
    dataloader_config = DataLoaderConfig(
            shuffle=args.dataloader_shuffle,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory
        )

    if args.dataset == "DummyDataset":
        dataset_config = DummyDatasetConfig(
            n_samples=args.dummy_n_samples,
            inputs_tensor_shape=args.dummy_input_shape,
            num_classes=args.ff_output_dim,
        )
    else:
        raise ValueError(f"No config defined for dataset: {args.dataset}")

    metrics_config_map = {}
    if "top_1_accuracy" in args.metrics:
        metrics_config_map["top_1_accuracy"] = TopKAccuracyConfig(
            top_k=list(args.topk_values)
        )
    if "top_5_accuracy" in args.metrics:
        metrics_config_map["top_5_accuracy"] = TopKAccuracyConfig(
            top_k=list(args.topk_values)
        )

    criterion_config: CriterionConfigs | None
    if args.criterion == "cross_entropy_loss":
        criterion_config = CrossEntropyLossConfigs(
            label_smoothing=args.ce_label_smoothing,
            ignore_index=args.ce_ignore_index,
            reduction=args.ce_reduction,
        )
    else:
        raise ValueError(f"No config defined for criterion: {args.criterion}")

    return TrainingConfig(
        optim=OptimizationConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_iters=args.warmup_iters,
            scheduler_step_unit=args.scheduler_step_unit,
        ),
        dist=DistributedConfig(
            world_size=args.world_size,
            rank=args.rank,
            dist_url=args.dist_url,
            dist_backend=args.dist_backend,
            multiprocessing_distributed=args.multiprocessing_distributed,
            gpu=args.gpu,
        ),
        logging=LoggingConfig(resume=args.resume, active_metrics=args.metrics),
        criterion=args.criterion,
        arch=args.arch,
        dataset=args.dataset,
        model_config=model_config,
        dataloader=dataloader_config,
        dataset_config=dataset_config,
        criterion_config=criterion_config,
        metrics_config=metrics_config_map,
    )
