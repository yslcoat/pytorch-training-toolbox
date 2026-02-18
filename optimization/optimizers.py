from typing import Protocol

import torch
import torch.nn as nn

from utils.configs import (
    AdamWConfigs,
    OptimizerConfigs,
    TrainingConfig,
)


class OptimizerBuilder(Protocol):
    def build(
        self,
        model: nn.Module,
        optimizer_config: OptimizerConfigs | None,
    ) -> torch.optim.Optimizer:
        ...


class AdamWBuilder(OptimizerBuilder):
    def build(
        self,
        model: nn.Module,
        optimizer_config: OptimizerConfigs | None,
    ) -> torch.optim.Optimizer:
        if optimizer_config is None:
            adamw_config = AdamWConfigs()
        elif not isinstance(optimizer_config, AdamWConfigs):
            raise TypeError(
                "AdamWBuilder expects AdamWConfigs or None, "
                f"got {type(optimizer_config)!r}"
            )
        else:
            adamw_config = optimizer_config

        return torch.optim.AdamW(
            model.parameters(),
            lr=adamw_config.lr,
            betas=adamw_config.betas,
            eps=adamw_config.eps,
            weight_decay=adamw_config.weight_decay,
            amsgrad=adamw_config.amsgrad,
        )


OPTIMIZER_REGISTRY = {
    "adamw": AdamWBuilder(),
}


def create_optimizer(
    configs: TrainingConfig,
    model: nn.Module,
) -> torch.optim.Optimizer:
    builder = OPTIMIZER_REGISTRY.get(configs.optimizer)
    if builder is None:
        raise ValueError(f"Unknown optimizer: {configs.optimizer}")
    return builder.build(model, configs.optimizer_config)
