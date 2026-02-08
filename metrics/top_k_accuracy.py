import torch
import torch.nn as nn


class TopKAccuracy(nn.Module):
    def __init__(self, topk: tuple = (1,)):
        super(TopKAccuracy, self).__init__()
        self.topk = topk

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(self.topk)
            batch_size = targets.size(0)

            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            res = []
            for k in self.topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res