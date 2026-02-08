import logging

from typing import Protocol
import torch.nn as nn

from top_k_accuracy import TopKAccuracy


class MetricsBuilder(Protocol):
    def build(self, configs) -> nn.Module:
        ...


class TopKAccuracyBuilder(MetricsBuilder):
    def build(self, configs) -> nn.Module:
        return TopKAccuracy(
            top_k=configs.top_k,
        )


METRICS_REGISTRY: dict[str, MetricsBuilder] = {
    "top_k_accuracy": TopKAccuracyBuilder(),
}


def create_model(configs, device, ngpus_per_node):
    logging.info("=> creating model '{}'".format(configs.arch))
    builder = METRICS_REGISTRY.get(configs.metrics)

    if not builder:
        raise ValueError(f"Metric {configs.metrics} not supported.")

    metrics = builder.build(configs)


    return metrics