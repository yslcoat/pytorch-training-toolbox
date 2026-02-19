from typing import Protocol, Sized

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    LinearLR,
    SequentialLR,
)

from utils.configs import (
    CosineAnnealingLRConfigs,
    LinearLRConfigs,
    LinearThenCosineAnnealingLRConfigs,
    SchedulerConfigs,
    TrainingConfig,
)


def _calculate_total_scheduler_iters(
    configs: TrainingConfig,
    train_loader: Sized,
) -> int:
    if configs.optim.scheduler_step_unit == "epoch":
        return configs.optim.epochs
    return configs.optim.epochs * len(train_loader)


class SchedulerBuilder(Protocol):
    def build(
        self,
        configs: TrainingConfig,
        optimizer: Optimizer,
        train_loader: Sized,
        scheduler_config: SchedulerConfigs | None,
    ) -> LRScheduler:
        ...


class LinearLRBuilder(SchedulerBuilder):
    def build(
        self,
        configs: TrainingConfig,
        optimizer: Optimizer,
        train_loader: Sized,
        scheduler_config: SchedulerConfigs | None,
    ) -> LRScheduler:
        if scheduler_config is None:
            linear_config = LinearLRConfigs()
        elif not isinstance(scheduler_config, LinearLRConfigs):
            raise TypeError(
                "LinearLRBuilder expects LinearLRConfigs or None, "
                f"got {type(scheduler_config)!r}"
            )
        else:
            linear_config = scheduler_config

        total_iters = linear_config.total_iters
        if total_iters is None:
            total_iters = _calculate_total_scheduler_iters(configs, train_loader)
        if total_iters <= 0:
            raise ValueError(
                "LinearLR requires total scheduler iterations > 0. "
                f"Got total_iters={total_iters}."
            )

        return LinearLR(
            optimizer,
            start_factor=linear_config.start_factor,
            end_factor=linear_config.end_factor,
            total_iters=total_iters,
        )


class CosineAnnealingLRBuilder(SchedulerBuilder):
    def build(
        self,
        configs: TrainingConfig,
        optimizer: Optimizer,
        train_loader: Sized,
        scheduler_config: SchedulerConfigs | None,
    ) -> LRScheduler:
        if scheduler_config is None:
            cosine_config = CosineAnnealingLRConfigs()
        elif not isinstance(scheduler_config, CosineAnnealingLRConfigs):
            raise TypeError(
                "CosineAnnealingLRBuilder expects CosineAnnealingLRConfigs or None, "
                f"got {type(scheduler_config)!r}"
            )
        else:
            cosine_config = scheduler_config

        t_max = cosine_config.t_max
        if t_max is None:
            t_max = _calculate_total_scheduler_iters(configs, train_loader)
        if t_max <= 0:
            raise ValueError(
                "CosineAnnealingLR requires T_max > 0. "
                f"Got T_max={t_max}."
            )

        return CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=cosine_config.eta_min,
        )


class LinearThenCosineAnnealingLRBuilder(SchedulerBuilder):
    def build(
        self,
        configs: TrainingConfig,
        optimizer: Optimizer,
        train_loader: Sized,
        scheduler_config: SchedulerConfigs | None,
    ) -> LRScheduler:
        if scheduler_config is None:
            combo_config = LinearThenCosineAnnealingLRConfigs()
        elif not isinstance(scheduler_config, LinearThenCosineAnnealingLRConfigs):
            raise TypeError(
                "LinearThenCosineAnnealingLRBuilder expects "
                "LinearThenCosineAnnealingLRConfigs or None, "
                f"got {type(scheduler_config)!r}"
            )
        else:
            combo_config = scheduler_config

        total_scheduler_iters = _calculate_total_scheduler_iters(configs, train_loader)
        if total_scheduler_iters <= 1:
            raise ValueError(
                "Linear+Cosine scheduler requires at least 2 total scheduler iterations. "
                f"Got total_scheduler_iters={total_scheduler_iters}."
            )

        warmup_iters = combo_config.warmup_iters
        if warmup_iters is None:
            if configs.optim.warmup_iters > 0:
                warmup_iters = configs.optim.warmup_iters
            else:
                warmup_iters = max(1, total_scheduler_iters // 10)

        if warmup_iters <= 0:
            raise ValueError(
                "warmup_iters must be > 0 for Linear+Cosine scheduler. "
                f"Got warmup_iters={warmup_iters}."
            )
        if warmup_iters >= total_scheduler_iters:
            raise ValueError(
                "warmup_iters must be smaller than total scheduler iterations for "
                "Linear+Cosine scheduler. "
                f"Got warmup_iters={warmup_iters}, "
                f"total_scheduler_iters={total_scheduler_iters}."
            )

        cosine_t_max = combo_config.cosine_t_max
        if cosine_t_max is None:
            cosine_t_max = total_scheduler_iters - warmup_iters
        if cosine_t_max <= 0:
            raise ValueError(
                "cosine_t_max must be > 0 for Linear+Cosine scheduler. "
                f"Got cosine_t_max={cosine_t_max}."
            )

        linear_scheduler = LinearLR(
            optimizer,
            start_factor=combo_config.linear_start_factor,
            end_factor=combo_config.linear_end_factor,
            total_iters=warmup_iters,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max,
            eta_min=combo_config.cosine_eta_min,
        )

        return SequentialLR(
            optimizer,
            schedulers=[linear_scheduler, cosine_scheduler],
            milestones=[warmup_iters],
        )


SCHEDULER_REGISTRY: dict[str, SchedulerBuilder] = {
    "linear_lr": LinearLRBuilder(),
    "cosine_annealing_lr": CosineAnnealingLRBuilder(),
    "linear_then_cosine_annealing_lr": LinearThenCosineAnnealingLRBuilder(),
}


def create_scheduler(
    configs: TrainingConfig,
    optimizer: Optimizer,
    train_loader: Sized,
) -> LRScheduler:
    if configs.scheduler == "none":
        raise ValueError(
            "create_scheduler called with scheduler='none'. "
            "Guard this in the caller and skip scheduler creation."
        )

    builder = SCHEDULER_REGISTRY.get(configs.scheduler)
    if builder is None:
        raise ValueError(f"Unknown scheduler: {configs.scheduler}")

    return builder.build(configs, optimizer, train_loader, configs.scheduler_config)
