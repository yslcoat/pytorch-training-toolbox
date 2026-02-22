import torch
import torch.nn as nn


class DiceScore(nn.Module):
    def __init__(
        self,
        smooth: float = 1e-6,
        from_logits: bool = True,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits
        self.threshold = threshold

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if outputs.dim() == targets.dim() + 1 and outputs.shape[1] == 1:
                targets = targets.unsqueeze(1)

            if outputs.shape != targets.shape:
                raise ValueError(
                    "DiceScore expects outputs and targets with matching shape, got "
                    f"{tuple(outputs.shape)} and {tuple(targets.shape)}."
                )

            preds = outputs
            if self.from_logits:
                preds = torch.sigmoid(preds)

            preds = (preds >= self.threshold).to(dtype=preds.dtype)
            targets = targets.to(dtype=preds.dtype)

            preds = preds.reshape(preds.shape[0], -1)
            targets = targets.reshape(targets.shape[0], -1)

            intersection = (preds * targets).sum(dim=1)
            union = preds.sum(dim=1) + targets.sum(dim=1)
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return dice.mean()
