import logging

from typing import Protocol
import torch.nn as nn

from .top_k_accuracy import Top1Accuracy, Top5Accuracy

class MetricsBuilder(Protocol):
    def build(self, _) -> nn.Module:
        ...

class Top1Builder(MetricsBuilder):
    def build(self, _) -> nn.Module:
        return Top1Accuracy()

class Top5Builder(MetricsBuilder):
    def build(self, _) -> nn.Module:
        return Top5Accuracy()

METRICS_REGISTRY = {
    "top_1_accuracy": Top1Builder(),
    "top_5_accuracy": Top5Builder(),
}


def create_model(configs, device, ngpus_per_node):
    logging.info("=> creating model '{}'".format(configs.arch))
    builder = METRICS_REGISTRY.get(configs.metrics)

    if not builder:
        raise ValueError(f"Metric {configs.metrics} not supported.")

    metrics = builder.build(configs)

    return metrics