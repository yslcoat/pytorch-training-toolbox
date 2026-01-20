import argparse
from typing import Optional

import torch
import torch.nn as nn


class Trainer():
    def __init__(
            self,
            *,
            configs,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler | None,
            train_dataloader: torch.utils.data.DataLoader,
            val_dataloader: torch.utils.data.DataLoader | None,
            metrics_engine,
            ):
        
            self.configs = configs
            self.train_dataloader = train_dataloader
            raise NotImplementedError
    
    def save_model(self):
        raise NotImplementedError
    
    def train_batch(self, inputs, targets):
        raise NotImplementedError
    
    def train_epoch(self):
        for epoch in range(self.configs.n_epochs):
            for inputs, targets in self.train_dataloader:
                self.train_batch(inputs, targets)

        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError