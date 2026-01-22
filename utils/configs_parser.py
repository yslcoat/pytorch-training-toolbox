import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    dummy_data: bool
    data_dir: Optional[Path] = None


def parse_training_configs():
    parser = argparse.ArgumentParser("Configuration parser for model training")
    data_configs = parser.add_mutually_exclusive_group()
    data_configs.add_argument(
        "--dummy_data",
        action="store_true",
        help="Specify to use dummy data."
    )
    data_configs.add_argument(
        "-d",
        "--data_dir",
        help="Directory containing training data.",
        type=Path
    )

    configs = parser.parse_args()
    return TrainingConfig(**vars(configs))