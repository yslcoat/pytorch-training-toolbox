import torch
import torch.nn as nn


class TopKBase(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            batch_size = targets.size(0)
            _, pred = outputs.topk(self.k, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            correct_k = correct[:self.k].reshape(-1).float().sum(0, keepdim=True)
            return correct_k.mul_(100.0 / batch_size).item()


class Top1Accuracy(TopKBase):
    def __init__(self):
        super().__init__(k=1)

class Top5Accuracy(TopKBase):
    def __init__(self):
        super().__init__(k=5)