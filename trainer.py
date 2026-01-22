import argparse
from typing import Optional, cast
import logging

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


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
            metrics_engine,
            local_rank: int,
            ):
        
            self.configs = configs
            self.local_rank = local_rank

            self.device = torch.device(f"cuda:{local_rank}")
            self.model = DDP(model.to(self.device), device_ids=[local_rank])

            self.optimizer = optimizer
            self.scheduler = scheduler
            self.criterion = criterion

            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
            
            train_sampler = train_dataloader.sampler
            if isinstance(train_sampler, DistributedSampler):
                self.train_sampler = cast(DistributedSampler, train_dataloader.sampler)
            else:
                raise TypeError("DDP Trainer requires a DistributedSampler")
            

            self.metrics_engine = metrics_engine
    
    def save_model(self):
        if self.local_rank == 0:
            state_dict = self.model.module.state_dict()
            torch.save(state_dict, f"{self.configs.save_path}/model.pt")
            logger.info(f"Model saved to {self.configs.save_path}/model.pt")
    
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
        if is_training:
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

        self.save_model()