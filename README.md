# PyTorch Training Toolbox

A small, registry-driven training framework for PyTorch experiments.

It currently supports:
- Model: `FeedForwardNeuralNetwork`
- Datasets: `DummyDataset`, `MNIST`
- Criterion: `cross_entropy_loss`
- Optimizer: `adamw`
- Schedulers: `none`, `linear_lr`, `cosine_annealing_lr`, `linear_then_cosine_annealing_lr`
- Metrics: `top_1_accuracy`, `top_5_accuracy`

## Repository layout

- `train.py`: training entrypoint
- `trainer.py`: train/validate loop and checkpointing
- `utils/configs.py`: dataclass configs
- `utils/configs_parser.py`: CLI argument parsing into `TrainingConfig`
- `models/models.py`: model builders and `MODEL_REGISTRY`
- `datasets/datasets.py`: dataset builders and `DATASET_REGISTRY`
- `criterions/criterions_factory.py`: criterion builders and registry
- `optimization/optimizers.py`: optimizer builders and registry
- `optimization/schedulers.py`: scheduler builders and registry
- `metrics/metrics.py`: metric builders and `METRICS_REGISTRY`

## Basic usage

Inspect all available CLI options:

```bash
python train.py --help
```

Run a minimal dummy training job:

```bash
python train.py \
  --dataset DummyDataset \
  --arch FeedForwardNeuralNetwork \
  --epochs 3 \
  --batch-size 64 \
  --dummy-n-samples 2048 \
  --dummy-val-n-samples 256 \
  --dummy-input-shape 784 \
  --ff-input-size 784 \
  --ff-output-dim 10
```

Run MNIST:

```bash
python train.py \
  --dataset MNIST \
  --arch FeedForwardNeuralNetwork \
  --epochs 5 \
  --batch-size 128 \
  --mnist-root ./data \
  --mnist-download \
  --ff-input-size 784 \
  --ff-output-dim 10
```

Outputs are written to `outputs/<timestamp>/`:
- `checkpoint.pth.tar`
- `model_best.pth.tar` (when best validation loss improves)
- `training.log`

Resume from a checkpoint:

```bash
python train.py --resume outputs/<timestamp>/checkpoint.pth.tar
```

## Current dataset transform behavior

Transforms are currently hardcoded in dataset builders:

- `DummyDatasetBuilder`: no transforms.
- `MnistDatasetBuilder`: `ToTensor -> Normalize((0.1307,), (0.3081,)) -> Flatten`.

This is implemented in `datasets/datasets.py`.

## How the framework works

1. `utils/configs_parser.py` parses CLI args into a `TrainingConfig`.
2. `train.py` calls `create_model`, `create_dataset`, `create_criterion`, `create_optimizer`, `create_scheduler`.
3. Each `create_*` function dispatches via its registry.
4. `trainer.py` runs train/val loops, logging, metric aggregation, and checkpointing.

## Adding a new model

1. Add the model module under `models/` (for example `models/MyModel.py`).
2. Add a model config dataclass in `utils/configs.py` (for example `MyModelConfig`).
3. Add a builder in `models/models.py` implementing `ModelBuilder`.
4. Register it in `MODEL_REGISTRY`.
5. Add parser args in `utils/configs_parser.py` and instantiate `model_config` in the `if args.arch == ...` branch.
6. Update `TrainingConfig.model_config` type in `utils/configs.py` to include your config type.

Checklist:
- Forward output shape should be `[batch_size, num_classes]` for current criterion/metrics.
- Ensure your model input shape matches your dataset output shape.

## Adding a new criterion

1. Add a criterion config dataclass in `utils/configs.py` if needed.
2. Implement a builder in `criterions/criterions_factory.py` implementing `CriterionBuilder`.
3. Register it in `CRITERIONS_REGISTRY`.
4. Add CLI args and creation branch for `criterion_config` in `utils/configs_parser.py`.

## Adding a new optimizer

1. Add optimizer config dataclass in `utils/configs.py`.
2. Implement a builder in `optimization/optimizers.py` implementing `OptimizerBuilder`.
3. Register in `OPTIMIZER_REGISTRY`.
4. Add CLI args and `optimizer_config` parsing branch in `utils/configs_parser.py`.

## Adding a new metric

1. Implement the metric module under `metrics/` as an `nn.Module` returning a scalar.
2. Add a builder in `metrics/metrics.py` implementing `MetricsBuilder`.
3. Register it in `METRICS_REGISTRY`.
4. Add any metric-specific config dataclass to `utils/configs.py` if needed.
5. Update `utils/configs_parser.py`:
   - add parser args for the metric
   - add logic that inserts the metric config into `metrics_config_map`

Notes:
- `MetricsEngine` builds metrics from `configs.logging.active_metrics`.
- Top-k metrics assume classification logits with class dimension at `dim=1`.

## Adding a new dataset

1. Add dataset implementation under `datasets/`.
2. Add dataset config dataclass in `utils/configs.py`.
3. Implement a builder in `datasets/datasets.py` implementing `DatasetBuilder`.
4. Register it in `DATASET_REGISTRY`.
5. Add parser args and dataset config creation branch in `utils/configs_parser.py`.
6. Update `TrainingConfig.dataset_config` union in `utils/configs.py`.

Important:
- `train.py` currently always constructs both `train` and `val` datasets.
- Your dataset builder should support `partition="val"` (or you must update `train.py` behavior).
- If your dataset needs transforms, define them in the dataset builder (current project convention).

## Common pitfalls

- `DummyDataset` needs `--dummy-val-n-samples > 0` with the current training flow.
- `top_5_accuracy` requires model output dimension of at least 5.
- MNIST is flattened in the dataset builder; if you add CNN models, adapt transforms accordingly.
