import torch
from enum import Enum
from typing import Dict, List, Callable

import torch.nn as nn
import torch.distributed as dist

from utils.configs import TrainingConfig
from metrics.metrics import METRICS_REGISTRY


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    def __init__(self, name, use_accel: bool = False, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.use_accel = use_accel
        self.fmt = fmt
        self.summary_type = summary_type
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class MetricsEngine:
    def __init__(self, configs: TrainingConfig):
        self.metric_functions: Dict[str, nn.Module] = {}
        for name in configs.logging.active_metrics:
            if name in METRICS_REGISTRY:
                builder = METRICS_REGISTRY[name]
                metric_config = configs.metrics_config.get(name)
                self.metric_functions[name] = builder.build(metric_config)

        self.use_accel = not configs.dist.no_accel and torch.accelerator.is_available()
        self.batch_history = []
        self.epoch_history = []
        self.best_loss = float("inf")
        if (
            configs.dist.distributed
            and dist.is_available()
            and dist.is_initialized()
        ):
            self.is_main_process = dist.get_rank() == 0
        else:
            self.is_main_process = True
        
        self.meters: Dict[str, AverageMeter] = {}
        for name in self.metric_functions.keys():
            self.meters[name] = AverageMeter(name=name, use_accel=self.use_accel)            
        if "loss" not in self.meters:
            self.meters["loss"] = AverageMeter(name="loss", use_accel=self.use_accel, fmt=":.4e")

    def set_mode(self, mode):
        self.mode = mode
        target_summary_type = Summary.AVERAGE if mode == "validate" else Summary.NONE

        for name, meter in self.meters.items():
            if name in self.metric_functions:
                meter.summary_type = target_summary_type

    def reset_metrics(self):
        for meter in self.meters.values():
            meter.reset()

    def configure_progress_meter(self, n_samples, epoch=None):
        if self.mode == "train":
            prefix = "Training Epoch: [{}]".format(epoch)
        elif self.mode == "validate":
            prefix = "Validate Epoch: [{}]".format(epoch)
        else:
            prefix = "Test: "

        meter_list = []
        
        if "loss" in self.meters:
            meter_list.append(self.meters["loss"])
            
        for name, meter in self.meters.items():
            if name != "loss":
                meter_list.append(meter)

        self.progress = ProgressMeter(
            n_samples,
            meter_list,
            prefix=prefix,
        )

    def process_batch_metrics(self, outputs: torch.Tensor, targets: torch.Tensor, loss_val: float):
        batch_results = {"loss": loss_val}
        batch_size = targets.size(0)

        if "loss" in self.meters:
            self.meters["loss"].update(loss_val, batch_size)

        for name, fn in self.metric_functions.items():
            val = fn(outputs, targets)
            if isinstance(val, torch.Tensor):
                val = val.item()
                
            self.meters[name].update(val, batch_size)
            batch_results[name] = val

        self.batch_history.append(batch_results)

    def process_epoch_metrics(self):
        for meter in self.meters.values():
            meter.all_reduce()

        epoch_results = {name: meter.avg for name, meter in self.meters.items()}
        self.epoch_history.append(epoch_results)
        
        if self.is_main_process and hasattr(self, 'progress'):
            self.progress.display_summary()

        self.reset_metrics()
            
        return epoch_results

    def update_best_loss(self, epoch_results: Dict[str, float], metric_name: str = "loss") -> bool:
        metric_val = epoch_results.get(metric_name)
        if metric_val is None:
            return False

        metric_val = float(metric_val)
        if metric_val < self.best_loss:
            self.best_loss = metric_val
            return True

        return False

    def get_latest_stats(self):
        return self.epoch_history[-1] if self.epoch_history else {}
