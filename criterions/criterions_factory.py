from typing import Protocol
import torch.nn as nn

from utils.configs import (
    CriterionConfigs,
    CrossEntropyLossConfigs,
    TrainingConfig,
)

class CriterionBuilder(Protocol):
    def build(self, criterion_config: CriterionConfigs | None) -> nn.Module:
        ...


class CrossEntropyLossBuilder(CriterionBuilder):
    def build(self, cross_entropy_configs: CrossEntropyLossConfigs | None) -> nn.Module:
        if cross_entropy_configs is None:
            cross_entropy_configs = CrossEntropyLossConfigs()
        elif not isinstance(cross_entropy_configs, CrossEntropyLossConfigs):
            raise TypeError(
                "CrossEntropyLossBuilder expects CrossEntropyLossConfigs or None, "
                f"got {type(cross_entropy_configs)!r}"
            )

        return nn.CrossEntropyLoss(
            label_smoothing=cross_entropy_configs.label_smoothing,
            ignore_index=cross_entropy_configs.ignore_index,
            reduction=cross_entropy_configs.reduction,
        )


CRITERIONS_REGISTRY = {
    "cross_entropy_loss": CrossEntropyLossBuilder(),
}


def create_criterion(configs: TrainingConfig) -> nn.Module:
    builder = CRITERIONS_REGISTRY.get(configs.criterion)
    if builder is None:
        raise ValueError(f"Unknown criterion: {configs.criterion}")
    return builder.build(configs.criterion_config)
