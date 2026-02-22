import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceScore(nn.Module):
    def __init__(
        self,
        smooth: float = 1e-6,
        from_logits: bool = True,
        threshold: float = 0.5,
    ):
        super().__init__()
        if smooth < 0.0:
            raise ValueError(f"smooth must be >= 0.0, got {smooth}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"threshold must be in [0.0, 1.0], got {threshold}"
            )
        self.smooth = smooth
        self.from_logits = from_logits
        self.threshold = threshold

    @staticmethod
    def _to_one_hot(
        targets: torch.Tensor,
        num_classes: int,
        spatial_shape: tuple[int, ...],
    ) -> torch.Tensor:
        if targets.shape[1:] != spatial_shape:
            raise ValueError(
                "Class-index targets must match model spatial shape. "
                f"Expected {spatial_shape}, got {tuple(targets.shape[1:])}."
            )
        return F.one_hot(targets.long(), num_classes=num_classes).movedim(-1, 1)

    @staticmethod
    def _expected_target_shapes(outputs: torch.Tensor) -> str:
        batch_size = outputs.shape[0]
        num_classes = outputs.shape[1]
        spatial_shape = tuple(outputs.shape[2:])

        if num_classes == 1:
            return (
                "Expected targets for binary DiceScore (C=1): "
                f"{(batch_size, *spatial_shape)} or {(batch_size, 1, *spatial_shape)}."
            )
        return (
            f"Expected targets for multiclass DiceScore (C={num_classes}): "
            f"{(batch_size, *spatial_shape)} for class indices in [0, {num_classes - 1}], "
            f"or {(batch_size, num_classes, *spatial_shape)} for one-hot/probabilities."
        )

    def _prepare_targets(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if outputs.dim() < 2:
            raise ValueError(
                "outputs must have at least 2 dimensions (B x C x ...), got "
                f"{tuple(outputs.shape)}."
            )
        if targets.dim() < 1:
            raise ValueError(
                "targets must have at least 1 dimension (batch-first), got "
                f"{tuple(targets.shape)}."
            )

        batch_size = outputs.shape[0]
        num_classes = outputs.shape[1]
        spatial_shape = tuple(outputs.shape[2:])

        if targets.shape[0] != batch_size:
            raise ValueError(
                "Batch size mismatch for DiceScore. "
                f"outputs={tuple(outputs.shape)}, targets={tuple(targets.shape)}. "
                f"Expected batch size {batch_size}, got {targets.shape[0]}."
            )

        if num_classes == 1:
            if targets.dim() == outputs.dim() and targets.shape == outputs.shape:
                return outputs, targets
            if (
                targets.dim() == outputs.dim() - 1
                and tuple(targets.shape) == (batch_size, *spatial_shape)
            ):
                return outputs, targets.unsqueeze(1)
        else:
            if targets.dim() == outputs.dim() and targets.shape == outputs.shape:
                return outputs, targets
            if (
                targets.dim() == outputs.dim() - 1
                and tuple(targets.shape) == (batch_size, *spatial_shape)
            ):
                return outputs, self._to_one_hot(
                    targets=targets,
                    num_classes=num_classes,
                    spatial_shape=spatial_shape,
                )

        raise ValueError(
            "Incompatible shapes for DiceScore. "
            f"outputs={tuple(outputs.shape)}, targets={tuple(targets.shape)}. "
            f"{self._expected_target_shapes(outputs)}"
        )

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if outputs.device != targets.device:
                raise ValueError(
                    "DiceScore expects outputs and targets on the same device, got "
                    f"outputs={outputs.device}, targets={targets.device}."
                )

            outputs, targets = self._prepare_targets(outputs, targets)

            preds = outputs
            if self.from_logits:
                if outputs.shape[1] == 1:
                    preds = torch.sigmoid(outputs)
                else:
                    preds = torch.softmax(outputs, dim=1)

            if preds.shape[1] == 1:
                preds = (preds >= self.threshold).to(dtype=preds.dtype)
                targets = (targets >= self.threshold).to(dtype=preds.dtype)
            else:
                pred_indices = preds.argmax(dim=1)
                preds = self._to_one_hot(
                    targets=pred_indices,
                    num_classes=preds.shape[1],
                    spatial_shape=tuple(preds.shape[2:]),
                ).to(dtype=preds.dtype)

                target_indices = targets.argmax(dim=1)
                targets = self._to_one_hot(
                    targets=target_indices,
                    num_classes=targets.shape[1],
                    spatial_shape=tuple(targets.shape[2:]),
                ).to(dtype=preds.dtype)

            preds = preds.reshape(preds.shape[0], preds.shape[1], -1)
            targets = targets.reshape(targets.shape[0], targets.shape[1], -1)

            intersection = (preds * targets).sum(dim=-1)
            union = preds.sum(dim=-1) + targets.sum(dim=-1)
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return dice.mean()
