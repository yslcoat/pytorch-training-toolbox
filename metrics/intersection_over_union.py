import torch
import torch.nn as nn


class BBoxIoUScore(nn.Module):
    def __init__(
        self,
        smooth: float = 1e-6,
        from_logits: bool = False,
        box_format: str = "xyxy", # Some datasets give x, y, width, height, others give x_min, y_min, x_max, y_max. This implementation supports both.
        reduction: str = "mean",
    ):
        super().__init__()
        if smooth < 0.0:
            raise ValueError(f"smooth must be >= 0.0, got {smooth}")
        if box_format not in {"xyxy", "cxcywh"}:
            raise ValueError(
                "box_format must be one of ['xyxy', 'cxcywh'], "
                f"got {box_format}"
            )
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                "reduction must be one of ['none', 'mean', 'sum'], "
                f"got {reduction}"
            )
        self.smooth = smooth
        self.from_logits = from_logits
        self.box_format = box_format
        self.reduction = reduction

    @staticmethod
    def _activate_bbox_preds(
        bbox_preds: torch.Tensor,
        box_format: str,
    ) -> torch.Tensor:
        if box_format == "xyxy":
            # For normalized corner coordinates, sigmoid is the expected mapping.
            return bbox_preds.sigmoid()

        # For cxcywh logits used by common detection heads:
        # - center coordinates are typically bounded with sigmoid
        # - width/height use exp so boxes are not restricted to <= 1
        center = bbox_preds[..., :2].sigmoid()
        size = bbox_preds[..., 2:].clamp(max=20.0).exp()
        return torch.cat((center, size), dim=-1)

    @staticmethod
    def _to_xyxy(boxes: torch.Tensor, box_format: str) -> torch.Tensor:
        if box_format == "xyxy":
            x1 = torch.minimum(boxes[..., 0], boxes[..., 2])
            y1 = torch.minimum(boxes[..., 1], boxes[..., 3])
            x2 = torch.maximum(boxes[..., 0], boxes[..., 2])
            y2 = torch.maximum(boxes[..., 1], boxes[..., 3])
            return torch.stack((x1, y1, x2, y2), dim=-1)

        cx, cy, w, h = boxes.unbind(dim=-1)
        w = w.clamp(min=0.0)
        h = h.clamp(min=0.0)
        half_w = w / 2.0
        half_h = h / 2.0
        x1 = cx - half_w
        y1 = cy - half_h
        x2 = cx + half_w
        y2 = cy + half_h
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def forward(
        self,
        bbox_preds: torch.Tensor,
        bbox_targets: torch.Tensor,
    ) -> torch.Tensor:
        # bbox_preds.shape: [batch_size, num_boxes, 4]
        preds = bbox_preds
        if self.from_logits:
            preds = self._activate_bbox_preds(preds, self.box_format)

        preds = self._to_xyxy(preds, self.box_format)
        targets = self._to_xyxy(
            bbox_targets.to(dtype=preds.dtype),
            self.box_format,
        )

        pred_x1, pred_y1, pred_x2, pred_y2 = preds.unbind(dim=-1)
        tgt_x1, tgt_y1, tgt_x2, tgt_y2 = targets.unbind(dim=-1)

        inter_x1 = torch.maximum(pred_x1, tgt_x1)
        inter_y1 = torch.maximum(pred_y1, tgt_y1)
        inter_x2 = torch.minimum(pred_x2, tgt_x2)
        inter_y2 = torch.minimum(pred_y2, tgt_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
        intersection = inter_w * inter_h

        pred_w = (pred_x2 - pred_x1).clamp(min=0.0)
        pred_h = (pred_y2 - pred_y1).clamp(min=0.0)
        target_w = (tgt_x2 - tgt_x1).clamp(min=0.0)
        target_h = (tgt_y2 - tgt_y1).clamp(min=0.0)
        pred_area = pred_w * pred_h
        target_area = target_w * target_h

        union = pred_area + target_area - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)

        if self.reduction == "none":
            return iou
        if self.reduction == "sum":
            return iou.sum()
        return iou.mean()
