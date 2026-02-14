import argparse

from models.models import MODEL_REGISTRY
from datasets.datasets import DATASET_REGISTRY
from metrics.metrics import METRICS_REGISTRY


from utils.configs import (
    TrainingConfig, 
    FeedForwardNetworkConfig, 
    DummyDatasetConfig, 
    TopKAccuracyConfig,
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

    dataloader_group = parser.add_argument_group("DataLoader")
    dataloader_group.add_argument("--dataloader-shuffle", default=True, type=bool)
    dataloader_group.add_argument("--dataloader-num-workers", default=4, type=int)
    dataloader_group.add_argument("--dataloader-pin-memory", default=True, type=bool)

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
    if "top_k_accuracy" in args.metrics:
        metrics_config_map["top_k_accuracy"] = TopKAccuracyConfig(
            top_k=args.topk_values
        )

    return TrainingConfig(
        optim=OptimizationConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_iters=args.warmup_iters,
            scheduler_step_unit=args.scheduler_step_unit,
        ),
        dist=DistributedConfig(world_size=args.world_size, gpu=args.gpu),
        logging=LoggingConfig(resume=args.resume, active_metrics=args.metrics),
        arch=args.arch,
        dataset=args.dataset,
        model_config=model_config,
        dataloader=dataloader_config,
        dataset_config=dataset_config,
        metrics_config=metrics_config_map,
    )
