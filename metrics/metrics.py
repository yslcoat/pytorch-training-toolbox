from typing import Protocol
import torch.nn as nn

from .dice_score import DiceScore
from .top_k_accuracy import Top1Accuracy, Top5Accuracy
from utils.configs import DiceScoreConfig, TopKAccuracyConfig

class MetricsBuilder(Protocol):
    def build(self, metric_config: object | None) -> nn.Module:
        ...


def _validate_top_k_metric_config(
    *,
    metric_name: str,
    expected_k: int,
    metric_config: object | None,
) -> None:
    if metric_config is None:
        return
    if not isinstance(metric_config, TopKAccuracyConfig):
        raise TypeError(
            f"{metric_name} expects TopKAccuracyConfig or None, got {type(metric_config)!r}"
        )
    if expected_k not in metric_config.top_k:
        raise ValueError(
            f"{metric_name} requires k={expected_k}, got top_k={metric_config.top_k}"
        )
    

class Top1Builder(MetricsBuilder):
    def build(self, metric_config: object | None) -> nn.Module:
        _validate_top_k_metric_config(
            metric_name="top_1_accuracy",
            expected_k=1,
            metric_config=metric_config,
        )
        return Top1Accuracy()
    

class Top5Builder(MetricsBuilder):
    def build(self, metric_config: object | None) -> nn.Module:
        _validate_top_k_metric_config(
            metric_name="top_5_accuracy",
            expected_k=5,
            metric_config=metric_config,
        )
        return Top5Accuracy()


class DiceScoreBuilder(MetricsBuilder):
    def build(self, metric_config: object | None) -> nn.Module:
        if metric_config is None:
            metric_config = DiceScoreConfig()
        elif not isinstance(metric_config, DiceScoreConfig):
            raise TypeError(
                "dice_score expects DiceScoreConfig or None, "
                f"got {type(metric_config)!r}"
            )

        return DiceScore(
            smooth=metric_config.smooth,
            from_logits=metric_config.from_logits,
            threshold=metric_config.threshold,
        )
    

METRICS_REGISTRY = {
    "top_1_accuracy": Top1Builder(),
    "top_5_accuracy": Top5Builder(),
    "dice_score": DiceScoreBuilder(),
}
