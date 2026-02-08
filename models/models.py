import logging

from typing import Protocol
import torch.nn as nn

from models.FeedForwardNeuralNetwork import FeedForwardNeuralNetwork
from utils.torch_utils import (
    configure_multi_gpu_model,
)
from utils.configs_parser import TrainingConfig, FeedForwardNetworkConfig


class ModelBuilder(Protocol):
    def build(self, configs: TrainingConfig) -> nn.Module: ...


class FeedForwardNeuralNetworkBuilder(ModelBuilder):
    def build(self, configs: TrainingConfig) -> nn.Module:
        model_conf = configs.model_config

        if not isinstance(model_conf, FeedForwardNetworkConfig):
            raise ValueError("Incorrect config type for FeedForwardNetwork")

        return FeedForwardNeuralNetwork(
            input_size=model_conf.input_size,
            n_hidden_layers=model_conf.n_layers,
            hidden_dim=model_conf.hidden_dim,
            output_dim=model_conf.output_dim,
            dropout=model_conf.dropout,
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
        model = configure_multi_gpu_model(configs, model, device, ngpus_per_node)

    return model
