import argparse
from pathlib import Path
from typing import Optional, cast
import logging
import shutil
import os

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from metrics.metrics_engine import MetricsEngine


logger = logging.getLogger(__name__)
logging.basicConfig(filename='training_log.log', encoding='utf-8', level=logging.DEBUG)


class TrainingManager():
    def __init__(
            self,
            *,
            configs,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler | None,
            criterion,
            train_dataloader: torch.utils.data.DataLoader,
            val_dataloader: torch.utils.data.DataLoader | None,
            metrics_engine: MetricsEngine,
            local_rank: int,
            ):
        
            self.configs = configs
            self.model = model

            if isinstance(local_rank, torch.device):
                self.device = local_rank
                self.local_rank = 0
            else:
                self.local_rank = local_rank
                self.device = torch.device(f"cuda:{local_rank}")

            self.optimizer = optimizer
            self.scheduler = scheduler
            self.criterion = criterion

            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
            
            self.train_sampler = train_dataloader.sampler            

            self.metrics_engine = metrics_engine
    
    def save_checkpoint(self, state, path, is_best, filename="checkpoint.pth.tar"):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(state, Path(path, filename))
        if is_best:
            shutil.copyfile(
                Path(path, filename), Path(path, "model_best.pth.tar")
            )

    def load_checkpoint(self) -> None:
        if os.path.isfile(self.configs.resume):
            logging.info("=> loading checkpoint '{}'".format(self.configs.resume))
            if self.configs.gpu is None:
                checkpoint = torch.load(self.configs.resume, weights_only=False)
            else:
                loc = f"{self.device.type}:{self.configs.gpu}"
                checkpoint = torch.load(self.configs.resume, map_location=loc, weights_only=False)
            self.configs.start_epoch = checkpoint["epoch"]
            self.best_loss = checkpoint.get("best_loss", float('inf'))
            if self.configs.gpu is not None:
                best_loss = best_loss.to(self.configs.gpu)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.scheduler and "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.metrics_engine.batch_history = checkpoint["batch_history"]
            self.metrics_engine.epoch_history = checkpoint["epoch_history"]
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    self.configs.resume, checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(self.configs.resume))
    
    def process_batch(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        if self.model.training:
            self.optimizer.zero_grad()        
            loss.backward()
            self.optimizer.step()

        self.metrics_engine.process_batch_metrics(outputs, targets, loss.item())
    
    def process_epoch(self, epoch, dataloader, is_training):
        if is_training and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(epoch)

        for inputs, targets in dataloader:
            self.process_batch(inputs, targets)

        self.metrics_engine.process_epoch_metrics()

    def validate(self, epoch):
        pass

    
    def train(self):
        for epoch in range(self.configs.n_epochs):
            if self.local_rank == 0:
                logger.info(f"Epoch: {epoch}")

            self.model.train()
            self.process_epoch(epoch, self.train_dataloader, self.model.training)

            if self.scheduler:
                self.scheduler.step()
            
            if self.val_dataloader:
                self.model.eval()
                with torch.no_grad():
                    self.process_epoch(epoch, self.val_dataloader, self.model.training)

        self.save_checkpoint({}, self.configs.output_dir, self.configs.output_filename)