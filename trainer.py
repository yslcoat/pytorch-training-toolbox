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
from utils.configs import TrainingConfig


logger = logging.getLogger(__name__)
logging.basicConfig(filename='training_log.log', encoding='utf-8', level=logging.DEBUG)


class TrainingManager():
    def __init__(
            self,
            *,
            configs: TrainingConfig,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler | None,
            criterion,
            train_dataloader: torch.utils.data.DataLoader,
            val_dataloader: torch.utils.data.DataLoader | None,
            metrics_engine: MetricsEngine,
            local_rank: int,
            device: torch.device,
            ):
        
            self.configs = configs
            self.model = model

            self.local_rank = local_rank
            self.device = device

            if (
                self.configs.dist.distributed
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                self.global_rank = torch.distributed.get_rank()
            else:
                self.global_rank = 0

            self.is_main_process = self.global_rank == 0

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
        if os.path.isfile(self.configs.logging.resume):
            logging.info("=> loading checkpoint '{}'".format(self.configs.logging.resume))
            if self.configs.dist.gpu is None:
                checkpoint = torch.load(self.configs.logging.resume, weights_only=False)
            else:
                loc = f"{self.device.type}:{self.configs.dist.gpu}"
                checkpoint = torch.load(self.configs.logging.resume, map_location=loc, weights_only=False)
            self.configs.optim.start_epoch = checkpoint["epoch"]
            best_loss = checkpoint.get("best_loss", float("inf"))
            if isinstance(best_loss, torch.Tensor):
                best_loss = best_loss.item()
            self.metrics_engine.best_loss = float(best_loss)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.scheduler and "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.metrics_engine.batch_history = checkpoint["batch_history"]
            self.metrics_engine.epoch_history = checkpoint["epoch_history"]
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    self.configs.logging.resume, checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(self.configs.logging.resume))

    def process_batch(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        if self.model.training:
            self.optimizer.zero_grad()        
            loss.backward()
            self.optimizer.step()
            if self.scheduler and self.configs.optim.scheduler_step_unit == "step":
                self.scheduler.step()

        self.metrics_engine.process_batch_metrics(outputs, targets, loss.item())

    def process_epoch(self, epoch, dataloader, is_training):
        if is_training and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(epoch)

        for i, (inputs, targets) in enumerate(dataloader):
            self.process_batch(inputs, targets)

            if i % self.configs.logging.print_freq == 0:
                self.metrics_engine.progress.display(i + 1)

        return self.metrics_engine.process_epoch_metrics()

    def validate(self, epoch):
        pass

    
    def train(self):
        start_epoch = self.configs.optim.start_epoch
        if start_epoch < 0:
            raise ValueError(f"start_epoch must be >= 0, got {start_epoch}")
        if start_epoch >= self.configs.optim.epochs:
            if self.is_main_process:
                logger.info(
                    "start_epoch (%s) >= configured epochs (%s); nothing to train.",
                    start_epoch,
                    self.configs.optim.epochs,
                )
            return

        for epoch in range(start_epoch, self.configs.optim.epochs):
            if self.is_main_process:
                logger.info(f"Epoch: {epoch}")

            self.metrics_engine.set_mode("train")
            self.metrics_engine.reset_metrics()
            self.metrics_engine.configure_progress_meter(len(self.train_dataloader), epoch)
            
            self.model.train()
            train_epoch_metrics = self.process_epoch(epoch, self.train_dataloader, is_training=True)

            if self.scheduler and self.configs.optim.scheduler_step_unit == "epoch":
                self.scheduler.step()

            is_best = False
            
            if self.val_dataloader:
                self.metrics_engine.set_mode("validate")
                self.metrics_engine.reset_metrics()
                self.metrics_engine.configure_progress_meter(len(self.val_dataloader), epoch)
                self.model.eval()
                with torch.no_grad():
                    val_epoch_metrics = self.process_epoch(epoch, self.val_dataloader, is_training=False)
                is_best = self.metrics_engine.update_best_loss(val_epoch_metrics)
            else:
                is_best = self.metrics_engine.update_best_loss(train_epoch_metrics)

            if self.is_main_process:
                state = {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_loss": self.metrics_engine.best_loss,
                    "batch_history": self.metrics_engine.batch_history,
                    "epoch_history": self.metrics_engine.epoch_history,
                }
                if self.scheduler:
                    state["scheduler"] = self.scheduler.state_dict()
                    
                self.save_checkpoint(
                    state, 
                    Path(self.configs.logging.output_dir), 
                    is_best, 
                    self.configs.logging.output_filename
                )
