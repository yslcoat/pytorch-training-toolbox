import argparse
from dataclasses import MISSING, fields
from pathlib import Path

from models.models import MODEL_REGISTRY
from datasets_factory.datasets import DATASET_REGISTRY
from metrics.metrics import METRICS_REGISTRY
from criterions.criterions_factory import CRITERIONS_REGISTRY
from optimization.optimizers import OPTIMIZER_REGISTRY
from optimization.schedulers import SCHEDULER_REGISTRY


from utils.configs import (
    TrainingConfig,
    FeedForwardNetworkConfig,
    DummyDatasetConfig,
    MnistDatasetConfig,
    ImageNetDatasetConfig,
    DataAugmentationConfig,
    BBoxIoUScoreConfig,
    TopKAccuracyConfig,
    DiceScoreConfig,
    CriterionConfigs,
    CrossEntropyLossConfigs,
    DiceLossConfigs,
    OptimizerConfigs,
    AdamWConfigs,
    SchedulerConfigs,
    LinearLRConfigs,
    CosineAnnealingLRConfigs,
    LinearThenCosineAnnealingLRConfigs,
    OptimizationConfig,
    DistributedConfig,
    DataLoaderConfig,
    LoggingConfig
)


def config_field_default(config_cls: type, field_name: str):
    config_field = next(
        (field for field in fields(config_cls) if field.name == field_name),
        None,
    )
    if config_field is None:
        raise ValueError(
            f"Field '{field_name}' not found on config class {config_cls.__name__}"
        )
    if config_field.default is not MISSING:
        return config_field.default
    if config_field.default_factory is not MISSING:
        return config_field.default_factory()
    raise ValueError(
        f"Field '{field_name}' on config class {config_cls.__name__} has no default"
    )


def parse_training_configs() -> TrainingConfig:
    parser = argparse.ArgumentParser("Configuration parser for model training")

    optim_group = parser.add_argument_group("Optimization")
    optim_group.add_argument(
        "--epochs",
        default=config_field_default(OptimizationConfig, "epochs"),
        type=int,
    )
    optim_group.add_argument(
        "--start-epoch",
        default=config_field_default(OptimizationConfig, "start_epoch"),
        type=int,
    )
    optim_group.add_argument(
        "--batch-size",
        default=config_field_default(OptimizationConfig, "batch_size"),
        type=int,
    )
    optim_group.add_argument(
        "--warmup-iters",
        default=config_field_default(OptimizationConfig, "warmup_iters"),
        type=int,
    )
    optim_group.add_argument(
        "--scheduler-step-unit",
        default=config_field_default(OptimizationConfig, "scheduler_step_unit"),
        choices=["step", "epoch"],
    )

    dist_group = parser.add_argument_group("Distributed")
    dist_group.add_argument(
        "--world-size",
        default=config_field_default(DistributedConfig, "world_size"),
        type=int,
    )
    dist_group.add_argument(
        "--rank",
        default=config_field_default(DistributedConfig, "rank"),
        type=int,
    )
    dist_group.add_argument(
        "--dist-url",
        default=config_field_default(DistributedConfig, "dist_url"),
        type=str,
    )
    dist_group.add_argument(
        "--dist-backend",
        default=config_field_default(DistributedConfig, "dist_backend"),
        type=str,
    )
    dist_group.add_argument(
        "--multiprocessing-distributed",
        default=config_field_default(DistributedConfig, "multiprocessing_distributed"),
        action="store_true",
    )
    dist_group.add_argument(
        "--gpu",
        default=config_field_default(DistributedConfig, "gpu"),
        type=int,
    )
    dist_group.add_argument(
        "--no-accel",
        default=config_field_default(DistributedConfig, "no_accel"),
        action="store_true",
    )

    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--metrics",
        nargs="+",
        default=config_field_default(LoggingConfig, "active_metrics"),
        choices=METRICS_REGISTRY.keys(),
    )
    log_group.add_argument(
        "--resume",
        default=config_field_default(LoggingConfig, "resume"),
        type=str,
    )
    log_group.add_argument(
        "--print-freq",
        default=config_field_default(LoggingConfig, "print_freq"),
        type=int,
    )
    log_group.add_argument(
        "--seed",
        default=config_field_default(LoggingConfig, "seed"),
        type=int,
    )

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
    selection_group.add_argument(
        "--optimizer", default="adamw", choices=OPTIMIZER_REGISTRY.keys()
    )
    selection_group.add_argument(
        "--scheduler",
        default="none",
        choices=["none", *SCHEDULER_REGISTRY.keys()],
    )

    dataloader_group = parser.add_argument_group("DataLoader")
    dataloader_group.add_argument(
        "--dataloader-shuffle",
        default=config_field_default(DataLoaderConfig, "shuffle"),
        action=argparse.BooleanOptionalAction,
    )
    dataloader_group.add_argument(
        "--dataloader-num-workers",
        default=config_field_default(DataLoaderConfig, "num_workers"),
        type=int,
    )
    dataloader_group.add_argument(
        "--dataloader-pin-memory",
        default=config_field_default(DataLoaderConfig, "pin_memory"),
        action=argparse.BooleanOptionalAction,
    )

    ff_group = parser.add_argument_group("Model: FeedForward")
    ff_group.add_argument(
        "--ff-input-size",
        default=config_field_default(FeedForwardNetworkConfig, "input_size"),
        type=int,
    )
    ff_group.add_argument(
        "--ff-n-layers",
        default=config_field_default(FeedForwardNetworkConfig, "n_layers"),
        type=int,
    )
    ff_group.add_argument(
        "--ff-hidden-dim",
        default=config_field_default(FeedForwardNetworkConfig, "hidden_dim"),
        type=int,
    )
    ff_group.add_argument(
        "--ff-output-dim",
        default=config_field_default(FeedForwardNetworkConfig, "output_dim"),
        type=int,
    )

    dummy_group = parser.add_argument_group("Dataset: Dummy")
    dummy_group.add_argument(
        "--dummy-n-samples",
        default=config_field_default(DummyDatasetConfig, "n_samples"),
        type=int,
    )
    dummy_group.add_argument(
        "--dummy-val-n-samples",
        default=config_field_default(DummyDatasetConfig, "val_n_samples"),
        type=int,
    )
    dummy_group.add_argument(
        "--dummy-input-shape",
        default=config_field_default(DummyDatasetConfig, "inputs_tensor_shape"),
        nargs="+",
        type=int,
    )

    mnist_group = parser.add_argument_group("Dataset: MNIST")
    mnist_group.add_argument(
        "--mnist-root",
        default=str(config_field_default(MnistDatasetConfig, "root")),
        type=str,
    )
    mnist_group.add_argument(
        "--mnist-download",
        default=config_field_default(MnistDatasetConfig, "download"),
        action=argparse.BooleanOptionalAction,
    )

    imagenet_group = parser.add_argument_group("Dataset: ImageNet")
    imagenet_group.add_argument(
        "--imagenet-root",
        default=str(config_field_default(ImageNetDatasetConfig, "root")),
        type=str,
    )
    imagenet_group.add_argument(
        "--imagenet-object-detection",
        default=config_field_default(ImageNetDatasetConfig, "object_detection"),
        action=argparse.BooleanOptionalAction,
    )

    aug_group = parser.add_argument_group("Data Augmentation")
    aug_group.add_argument(
        "--randaug-num-ops",
        default=config_field_default(DataAugmentationConfig, "randaug_num_ops"),
        type=int,
    )
    aug_group.add_argument(
        "--randaug-magnitude",
        default=config_field_default(DataAugmentationConfig, "randaug_magnitude"),
        type=int,
    )


    topk_group = parser.add_argument_group("Metric: TopK")
    topk_group.add_argument(
        "--topk-values",
        default=config_field_default(TopKAccuracyConfig, "top_k"),
        nargs="+",
        type=int,
    )

    dice_score_group = parser.add_argument_group("Metric: DiceScore")
    dice_score_group.add_argument(
        "--dice-score-smooth",
        default=config_field_default(DiceScoreConfig, "smooth"),
        type=float,
    )
    dice_score_group.add_argument(
        "--dice-score-from-logits",
        default=config_field_default(DiceScoreConfig, "from_logits"),
        action=argparse.BooleanOptionalAction,
    )
    dice_score_group.add_argument(
        "--dice-score-threshold",
        default=config_field_default(DiceScoreConfig, "threshold"),
        type=float,
    )

    bbox_iou_group = parser.add_argument_group("Metric: BBoxIoUScore")
    bbox_iou_group.add_argument(
        "--bbox-iou-smooth",
        default=config_field_default(BBoxIoUScoreConfig, "smooth"),
        type=float,
    )
    bbox_iou_group.add_argument(
        "--bbox-iou-from-logits",
        default=config_field_default(BBoxIoUScoreConfig, "from_logits"),
        action=argparse.BooleanOptionalAction,
    )
    bbox_iou_group.add_argument(
        "--bbox-iou-box-format",
        default=config_field_default(BBoxIoUScoreConfig, "box_format"),
        choices=["xyxy", "cxcywh"],
    )
    bbox_iou_group.add_argument(
        "--bbox-iou-reduction",
        default=config_field_default(BBoxIoUScoreConfig, "reduction"),
        choices=["none", "mean", "sum"],
    )

    ce_group = parser.add_argument_group("Criterion: CrossEntropyLoss")
    ce_group.add_argument(
        "--ce-label-smoothing",
        default=config_field_default(CrossEntropyLossConfigs, "label_smoothing"),
        type=float,
    )
    ce_group.add_argument(
        "--ce-ignore-index",
        default=config_field_default(CrossEntropyLossConfigs, "ignore_index"),
        type=int,
    )
    ce_group.add_argument(
        "--ce-reduction",
        default=config_field_default(CrossEntropyLossConfigs, "reduction"),
        choices=["none", "mean", "sum"],
    )

    dice_group = parser.add_argument_group("Criterion: DiceLoss")
    dice_group.add_argument(
        "--dice-smooth",
        default=config_field_default(DiceLossConfigs, "smooth"),
        type=float,
    )
    dice_group.add_argument(
        "--dice-from-logits",
        default=config_field_default(DiceLossConfigs, "from_logits"),
        action=argparse.BooleanOptionalAction,
    )
    dice_group.add_argument(
        "--dice-reduction",
        default=config_field_default(DiceLossConfigs, "reduction"),
        choices=["none", "mean", "sum"],
    )

    adamw_group = parser.add_argument_group("Optimizer: AdamW")
    adamw_group.add_argument(
        "--adamw-lr",
        "--lr",
        dest="adamw_lr",
        default=config_field_default(AdamWConfigs, "lr"),
        type=float,
    )
    adamw_group.add_argument(
        "--adamw-weight-decay",
        "--weight-decay",
        dest="adamw_weight_decay",
        default=config_field_default(AdamWConfigs, "weight_decay"),
        type=float,
    )
    adamw_group.add_argument(
        "--adamw-betas",
        default=config_field_default(AdamWConfigs, "betas"),
        nargs=2,
        type=float,
    )
    adamw_group.add_argument(
        "--adamw-eps",
        default=config_field_default(AdamWConfigs, "eps"),
        type=float,
    )
    adamw_group.add_argument(
        "--adamw-amsgrad",
        default=config_field_default(AdamWConfigs, "amsgrad"),
        action=argparse.BooleanOptionalAction,
    )

    linear_group = parser.add_argument_group("Scheduler: LinearLR")
    linear_group.add_argument(
        "--linear-start-factor",
        default=config_field_default(LinearLRConfigs, "start_factor"),
        type=float,
    )
    linear_group.add_argument(
        "--linear-end-factor",
        default=config_field_default(LinearLRConfigs, "end_factor"),
        type=float,
    )
    linear_group.add_argument(
        "--linear-total-iters",
        default=config_field_default(LinearLRConfigs, "total_iters"),
        type=int,
    )

    cosine_group = parser.add_argument_group("Scheduler: CosineAnnealingLR")
    cosine_group.add_argument(
        "--cosine-eta-min",
        default=config_field_default(CosineAnnealingLRConfigs, "eta_min"),
        type=float,
    )
    cosine_group.add_argument(
        "--cosine-t-max",
        default=config_field_default(CosineAnnealingLRConfigs, "t_max"),
        type=int,
    )

    linear_cosine_group = parser.add_argument_group("Scheduler: Linear + Cosine")
    linear_cosine_group.add_argument(
        "--linear-cosine-start-factor",
        default=config_field_default(
            LinearThenCosineAnnealingLRConfigs,
            "linear_start_factor",
        ),
        type=float,
    )
    linear_cosine_group.add_argument(
        "--linear-cosine-end-factor",
        default=config_field_default(
            LinearThenCosineAnnealingLRConfigs,
            "linear_end_factor",
        ),
        type=float,
    )
    linear_cosine_group.add_argument(
        "--linear-cosine-warmup-iters",
        default=config_field_default(
            LinearThenCosineAnnealingLRConfigs,
            "warmup_iters",
        ),
        type=int,
    )
    linear_cosine_group.add_argument(
        "--linear-cosine-eta-min",
        default=config_field_default(
            LinearThenCosineAnnealingLRConfigs,
            "cosine_eta_min",
        ),
        type=float,
    )
    linear_cosine_group.add_argument(
        "--linear-cosine-t-max",
        default=config_field_default(
            LinearThenCosineAnnealingLRConfigs,
            "cosine_t_max",
        ),
        type=int,
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
            val_n_samples=args.dummy_val_n_samples,
            inputs_tensor_shape=args.dummy_input_shape,
            num_classes=args.ff_output_dim,
        )
    elif args.dataset == "MNIST":
        dataset_config = MnistDatasetConfig(
            root=Path(args.mnist_root),
            download=args.mnist_download,
        )
    elif args.dataset == "ImageNet":
        dataset_config = ImageNetDatasetConfig(
            root=Path(args.imagenet_root),
            object_detection=args.imagenet_object_detection,
        )
    else:
        raise ValueError(f"No config defined for dataset: {args.dataset}")

    data_augmentation_config = DataAugmentationConfig(
        randaug_num_ops=args.randaug_num_ops,
        randaug_magnitude=args.randaug_magnitude,
    )

    metrics_config_map = {}
    if "top_1_accuracy" in args.metrics:
        metrics_config_map["top_1_accuracy"] = TopKAccuracyConfig(
            top_k=list(args.topk_values)
        )
    if "top_5_accuracy" in args.metrics:
        metrics_config_map["top_5_accuracy"] = TopKAccuracyConfig(
            top_k=list(args.topk_values)
        )
    if "dice_score" in args.metrics:
        metrics_config_map["dice_score"] = DiceScoreConfig(
            smooth=args.dice_score_smooth,
            from_logits=args.dice_score_from_logits,
            threshold=args.dice_score_threshold,
        )
    if (
        "bbox_iou_score" in args.metrics
        or "intersection_over_union" in args.metrics
    ):
        bbox_iou_config = BBoxIoUScoreConfig(
            smooth=args.bbox_iou_smooth,
            from_logits=args.bbox_iou_from_logits,
            box_format=args.bbox_iou_box_format,
            reduction=args.bbox_iou_reduction,
        )
        if "bbox_iou_score" in args.metrics:
            metrics_config_map["bbox_iou_score"] = bbox_iou_config
        if "intersection_over_union" in args.metrics:
            metrics_config_map["intersection_over_union"] = bbox_iou_config

    criterion_config: CriterionConfigs | None
    if args.criterion == "cross_entropy_loss":
        criterion_config = CrossEntropyLossConfigs(
            label_smoothing=args.ce_label_smoothing,
            ignore_index=args.ce_ignore_index,
            reduction=args.ce_reduction,
        )
    elif args.criterion == "dice_loss":
        criterion_config = DiceLossConfigs(
            smooth=args.dice_smooth,
            from_logits=args.dice_from_logits,
            reduction=args.dice_reduction,
        )
    else:
        raise ValueError(f"No config defined for criterion: {args.criterion}")

    optimizer_config: OptimizerConfigs | None
    if args.optimizer == "adamw":
        optimizer_config = AdamWConfigs(
            lr=args.adamw_lr,
            weight_decay=args.adamw_weight_decay,
            betas=tuple(args.adamw_betas),
            eps=args.adamw_eps,
            amsgrad=args.adamw_amsgrad,
        )
    else:
        raise ValueError(f"No config defined for optimizer: {args.optimizer}")

    scheduler_config: SchedulerConfigs | None
    if args.scheduler == "none":
        scheduler_config = None
    elif args.scheduler == "linear_lr":
        scheduler_config = LinearLRConfigs(
            start_factor=args.linear_start_factor,
            end_factor=args.linear_end_factor,
            total_iters=args.linear_total_iters,
        )
    elif args.scheduler == "cosine_annealing_lr":
        scheduler_config = CosineAnnealingLRConfigs(
            eta_min=args.cosine_eta_min,
            t_max=args.cosine_t_max,
        )
    elif args.scheduler == "linear_then_cosine_annealing_lr":
        scheduler_config = LinearThenCosineAnnealingLRConfigs(
            linear_start_factor=args.linear_cosine_start_factor,
            linear_end_factor=args.linear_cosine_end_factor,
            warmup_iters=args.linear_cosine_warmup_iters,
            cosine_eta_min=args.linear_cosine_eta_min,
            cosine_t_max=args.linear_cosine_t_max,
        )
    else:
        raise ValueError(f"No config defined for scheduler: {args.scheduler}")

    return TrainingConfig(
        optim=OptimizationConfig(
            epochs=args.epochs,
            start_epoch=args.start_epoch,
            batch_size=args.batch_size,
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
            no_accel=args.no_accel,
        ),
        logging=LoggingConfig(
            resume=args.resume,
            print_freq=args.print_freq,
            seed=args.seed,
            active_metrics=args.metrics,
        ),
        criterion=args.criterion,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        arch=args.arch,
        dataset=args.dataset,
        model_config=model_config,
        dataloader=dataloader_config,
        dataset_config=dataset_config,
        data_augmentation=data_augmentation_config,
        criterion_config=criterion_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        metrics_config=metrics_config_map,
    )
