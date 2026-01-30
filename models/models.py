
from typing import Protocol
import torch.nn as nn

from models.FeedForwardNeuralNetwork import FeedForwardNeuralNetwork


class ModelBuilder(Protocol):
    def build(self, model_configs) -> nn.Module:
        ...


class FeedForwardNeuralNetworkBuilder(ModelBuilder):
    def build(self, model_configs) -> nn.Module:
        return FeedForwardNeuralNetwork(
            model_configs.input_size,
            model_configs.n_layers,
            model_configs.hidden_dim,
            model_configs.output_size,
        )


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "FeedForwardNeuralNetwork": FeedForwardNeuralNetworkBuilder(),
}


def create_model(model_configs):
    builder = MODEL_REGISTRY.get(model_configs.arch)

    if not builder:
        raise ValueError(f"Model {model_configs.arch} not supported.")

    return builder.build(model_configs)