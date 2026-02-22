import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for binary and multiclass segmentation.

    Supported target formats:
    - Binary: `B x ...` or `B x 1 x ...`
    - Multiclass indices: `B x ...` with model output `B x C x ...`
    - Multiclass one-hot/probabilities: `B x C x ...`
    """

    def __init__(
        self,
        smooth: float = 1e-6,
        from_logits: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        if smooth < 0.0:
            raise ValueError(f"smooth must be >= 0.0, got {smooth}")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"reduction must be one of ['none', 'mean', 'sum'], got {reduction}"
            )
        self.smooth = smooth
        self.from_logits = from_logits
        self.reduction = reduction

    @staticmethod
    def _to_one_hot(
        targets: torch.Tensor,
        num_classes: int,
        spatial_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Convert class-index targets (`B x ...`) to one-hot (`B x C x ...`)."""
        if targets.shape[1:] != spatial_shape:
            raise ValueError(
                "Class-index targets must match model spatial shape. "
                f"Expected {spatial_shape}, got {tuple(targets.shape[1:])}."
            )

        class_indices = targets.long()

        return F.one_hot(class_indices, num_classes=num_classes).movedim(-1, 1)

    @staticmethod
    def _expected_target_shapes(model_output: torch.Tensor) -> str:
        batch_size = model_output.shape[0]
        num_classes = model_output.shape[1]
        spatial_shape = tuple(model_output.shape[2:])

        if num_classes == 1:
            return (
                "Expected targets for binary DiceLoss (C=1): "
                f"{(batch_size, *spatial_shape)} or {(batch_size, 1, *spatial_shape)}."
            )

        return (
            f"Expected targets for multiclass DiceLoss (C={num_classes}): "
            f"{(batch_size, *spatial_shape)} for class indices in [0, {num_classes - 1}], "
            f"or {(batch_size, num_classes, *spatial_shape)} for one-hot/probabilities."
        )

    def _prepare_targets(
        self,
        model_output: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if model_output.dim() < 2:
            raise ValueError(
                "model_output must have at least 2 dimensions (B x C x ...), got "
                f"{tuple(model_output.shape)}."
            )

        batch_size = model_output.shape[0]
        num_classes = model_output.shape[1]
        spatial_shape = tuple(model_output.shape[2:])

        if targets.dim() < 1:
            raise ValueError(
                "targets must have at least 1 dimension (batch-first), got "
                f"{tuple(targets.shape)}."
            )

        if targets.shape[0] != batch_size:
            raise ValueError(
                "Batch size mismatch for DiceLoss. "
                f"model_output={tuple(model_output.shape)}, targets={tuple(targets.shape)}. "
                f"Expected batch size {batch_size}, got {targets.shape[0]}."
            )

        if num_classes == 1:
            if targets.dim() == model_output.dim() and targets.shape == model_output.shape:
                return model_output, targets

            if (
                targets.dim() == model_output.dim() - 1
                and tuple(targets.shape) == (batch_size, *spatial_shape)
            ):
                return model_output, targets.unsqueeze(1)
        else:
            if targets.dim() == model_output.dim() and targets.shape == model_output.shape:
                return model_output, targets

            if (
                targets.dim() == model_output.dim() - 1
                and tuple(targets.shape) == (batch_size, *spatial_shape)
            ):
                one_hot = self._to_one_hot(
                    targets=targets,
                    num_classes=num_classes,
                    spatial_shape=spatial_shape,
                )
                return model_output, one_hot

        raise ValueError(
            "Incompatible shapes for DiceLoss. "
            f"model_output={tuple(model_output.shape)}, targets={tuple(targets.shape)}. "
            f"{self._expected_target_shapes(model_output)}"
        )

    def forward(self, model_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if model_output.device != targets.device:
            raise ValueError(
                "DiceLoss expects model_output and targets on the same device, got "
                f"model_output={model_output.device}, targets={targets.device}."
            )

        model_output, targets = self._prepare_targets(model_output, targets)

        if self.from_logits:
            if model_output.shape[1] == 1:
                model_output = torch.sigmoid(model_output)
            else:
                model_output = torch.softmax(model_output, dim=1)

        targets = targets.to(dtype=model_output.dtype)
        if model_output.shape != targets.shape:
            raise ValueError(
                "model_output and targets must have the same shape after "
                f"normalization, got {tuple(model_output.shape)} and {tuple(targets.shape)}."
            )

        model_output = model_output.reshape(model_output.shape[0], model_output.shape[1], -1)
        targets = targets.reshape(targets.shape[0], targets.shape[1], -1)

        intersection = (model_output * targets).sum(dim=-1)
        union = model_output.sum(dim=-1) + targets.sum(dim=-1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()
