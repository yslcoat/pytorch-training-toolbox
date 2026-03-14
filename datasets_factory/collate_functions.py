from torchvision.transforms import v2
from torch.utils.data import default_collate


class MixUpCollator:
    def __init__(self, num_classes):
        self.mixup = v2.MixUp(num_classes=num_classes)

    def __call__(self, batch):
        return self.mixup(*default_collate(batch))