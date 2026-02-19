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
    def build(self, criterion_config: CriterionConfigs | None) -> nn.Module:
        if criterion_config is None:
            criterion_config = CrossEntropyLossConfigs()
        elif not isinstance(criterion_config, CrossEntropyLossConfigs):
            raise TypeError(
                "CrossEntropyLossBuilder expects CrossEntropyLossConfigs or None, "
                f"got {type(criterion_config)!r}"
            )

        return nn.CrossEntropyLoss(
            label_smoothing=criterion_config.label_smoothing,
            ignore_index=criterion_config.ignore_index,
            reduction=criterion_config.reduction,
        )


CRITERIONS_REGISTRY = {
    "cross_entropy_loss": CrossEntropyLossBuilder(),
}


def create_criterion(configs: TrainingConfig) -> nn.Module:
    builder = CRITERIONS_REGISTRY.get(configs.criterion)
    if builder is None:
        raise ValueError(f"Unknown criterion: {configs.criterion}")
    return builder.build(configs.criterion_config)
