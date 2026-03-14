import torch
import torch.nn as nn


class TopKBase(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be > 0 for TopK metric, got {k}")
        self.k = k

    def _resolve_targets(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 1:
            return targets.long()

        if targets.ndim == 2 and targets.size(1) == 1:
            return targets.squeeze(1).long()

        if targets.ndim == 2 and targets.size(1) == outputs.size(1):
            return targets.float().argmax(dim=1)

        raise ValueError(
            "Top-k accuracy expects classification targets of shape "
            "[batch_size] or [batch_size, num_classes] (for mixed/soft labels). "
            f"Got shape {tuple(targets.shape)}."
        )

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            if outputs.ndim < 2:
                raise ValueError(
                    f"Expected model outputs with shape [batch_size, num_classes], got {tuple(outputs.shape)}"
                )

            num_classes = outputs.size(1)
            if self.k > num_classes:
                raise ValueError(
                    f"Top-{self.k} accuracy is invalid for outputs with {num_classes} classes. "
                    "Reduce k or increase model output dimension."
                )

            hard_targets = self._resolve_targets(outputs, targets)
            if hard_targets.shape[0] != outputs.size(0):
                raise ValueError(
                    f"Mismatched batch for outputs/targets: {outputs.size(0)} vs {hard_targets.size(0)}"
                )
            batch_size = hard_targets.size(0)
            _, pred = outputs.topk(self.k, 1, True, True)
            pred = pred.t()
            correct = pred.eq(hard_targets.view(1, -1).expand_as(pred))
            correct_k = correct[:self.k].reshape(-1).float().sum(0, keepdim=True)
            return correct_k.mul_(100.0 / batch_size).item()


class Top1Accuracy(TopKBase):
    def __init__(self):
        super().__init__(k=1)

class Top5Accuracy(TopKBase):
    def __init__(self):
        super().__init__(k=5)
