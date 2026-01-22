import torch
import torch.nn as nn

from utils.configs_parser import (
    TrainingConfig,
    parse_training_configs
)

def main(configs: TrainingConfig):
    raise NotImplementedError


if __name__ == "__main__":
    configs = parse_training_configs()
    main(configs)