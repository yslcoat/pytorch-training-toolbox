
import logging

from typing import Protocol
import torch.nn as nn

from models.FeedForwardNeuralNetwork import FeedForwardNeuralNetwork
from utils.torch_utils import (
    configure_multi_gpu_model,
)


class ModelBuilder(Protocol):
    def build(self, configs) -> nn.Module:
        ...


class FeedForwardNeuralNetworkBuilder(ModelBuilder):
    def build(self, configs) -> nn.Module:
        return FeedForwardNeuralNetwork(
            configs.input_size,
            configs.n_layers,
            configs.hidden_dim,
            configs.output_size,
        )


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "FeedForwardNeuralNetwork": FeedForwardNeuralNetworkBuilder(),
}


def create_model(configs, device, ngpus_per_node):
    logging.info("=> creating model '{}'".format(configs.arch))
    builder = MODEL_REGISTRY.get(configs.arch)

    if not builder:
        raise ValueError(f"Model {configs.arch} not supported.")
    
    model = builder.build(configs)

    if not configs.use_accel:
        logging.info("using CPU, this will be slow")
    else:
        configure_multi_gpu_model(configs, model, device, ngpus_per_node)

    return model