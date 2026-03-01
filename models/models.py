import logging

from typing import Protocol
import torch.nn as nn

from models.FeedForwardNeuralNetwork import FeedForwardNeuralNetwork
from models.vision_transformers.vision_transformer import VisionTransformer

from utils.torch_utils import (
    configure_multi_device_model,
)
from utils.configs import TrainingConfig, FeedForwardNetworkConfig, VisionTransformerConfig


class ModelBuilder(Protocol):
    def build(self, configs: TrainingConfig) -> nn.Module: ...


class FeedForwardNeuralNetworkBuilder(ModelBuilder):
    def build(self, configs: TrainingConfig) -> nn.Module:
        model_config = configs.model_config

        if not isinstance(model_config, FeedForwardNetworkConfig):
            raise ValueError("Incorrect config type for FeedForwardNetwork")

        return FeedForwardNeuralNetwork(
            input_size=model_config.input_size,
            n_hidden_layers=model_config.n_layers,
            hidden_dim=model_config.hidden_dim,
            output_dim=model_config.output_dim,
            dropout=model_config.dropout,
        )


class VisionTransformerBuilder(ModelBuilder):
    def build(self, configs: TrainingConfig) -> nn.Module:
        model_config = configs.model_config

        if not isinstance(model_config, VisionTransformerConfig):
            raise ValueError("Incorrect config type for VisionTransformer")

        return VisionTransformer(
            image_size=model_config.image_size,
            n_channels=model_config.n_channels,
            patch_size=model_config.patch_size,
            num_classes=model_config.num_classes,
            emb_dim=model_config.emb_dim,
            n_heads=model_config.n_heads,
            n_blocks=model_config.n_blocks,
            attn_head_dim=model_config.attn_head_dim,
            dropout=model_config.dropout,
            emb_dropout=model_config.emb_dropout,
        )


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "FeedForwardNeuralNetwork": FeedForwardNeuralNetworkBuilder(),
    "VisionTransformer": VisionTransformerBuilder(),
}


def create_model(configs: TrainingConfig, device, ngpus_per_node):
    logging.info("=> creating model '{}'".format(configs.arch))
    builder = MODEL_REGISTRY.get(configs.arch)

    if not builder:
        raise ValueError(f"Model {configs.arch} not supported.")

    model = builder.build(configs)

    if configs.dist.no_accel or device.type == "cpu":
        logging.info("using CPU, this will be slow")

    model = configure_multi_device_model(configs, model, device, ngpus_per_node)

    return model
