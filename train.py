import torch
import torch.nn as nn

from trainer import TrainingManager
from models.models import create_model
from utils.configs_parser import (
    TrainingConfig,
    parse_training_configs
)



def main(configs: TrainingConfig):
    raise NotImplementedError


if __name__ == "__main__":
    configs = parse_training_configs()
    main(configs)