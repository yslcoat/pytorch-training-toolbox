import logging

import torch

from utils.configs_parser import (
    TrainingConfig,
)


def configure_process_logging(configs: TrainingConfig) -> None:
    if (
        configs.dist.distributed
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        is_main_process = torch.distributed.get_rank() == 0
    else:
        is_main_process = True

    handlers: list[logging.Handler] = []
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    if is_main_process:
        configs.logging.output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            configs.logging.output_dir / "training.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.INFO if is_main_process else logging.WARNING,
        handlers=handlers,
        force=True,
    )