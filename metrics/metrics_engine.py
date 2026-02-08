import torch
import logging
from typing import Dict, List, Callable

import torch.distributed as dist

from metrics.metrics_functions import METRICS_FUNC_REGISTRY


class AverageMeter:
    def __init__(self, use_accel: bool = False):
        self.use_accel = use_accel
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if self.use_accel: # TODO: Wanna make this smarter. Could this be configured in the configs? 
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")

        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        
        if dist.is_initialized():
            dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
            
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count


class MetricsEngine:
    def __init__(self, metrics_configs):
        self.metric_functions: Dict[str, Callable] = {
            name: METRICS_FUNC_REGISTRY[name] 
            for name in metrics_configs.metrics 
            if name in METRICS_FUNC_REGISTRY
        }
        self.use_accel = not metrics_configs.no_accel and torch.accelerator.is_available()
        self.batch_history = []
        self.epoch_history = []
        
        self.meters: Dict[str, AverageMeter] = {
            name: AverageMeter(use_accel=self.use_accel) 
            for name in self.metric_functions.keys()
        }
        self.meters["loss"] = AverageMeter(use_accel=self.use_accel)

    def process_batch_metrics(self, outputs: torch.Tensor, targets: torch.Tensor, loss_val: float):
        batch_results = {"loss": loss_val}
        batch_size = targets.size(0)

        self.meters["loss"].update(loss_val, batch_size)

        for name, fn in self.metric_functions.items():
            val = fn(outputs, targets)
            self.meters[name].update(val, batch_size)
            batch_results[name] = val

        self.batch_history.append(batch_results)

    def process_epoch_metrics(self):
        for meter in self.meters.values():
            meter.all_reduce()

        epoch_results = {name: meter.avg for name, meter in self.meters.items()}
        self.epoch_history.append(epoch_results)
        
        for meter in self.meters.values():
            meter.reset()
            
        return epoch_results

    def get_latest_stats(self):
        return self.epoch_history[-1] if self.epoch_history else {}
