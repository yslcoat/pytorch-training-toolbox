import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, n_samples: int, inputs_tensor_shape: tuple, num_classes: int):
        self.n_samples = n_samples
        self.inputs_tensor_shape = inputs_tensor_shape
        self.num_classes = num_classes

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        inputs = torch.randn(self.inputs_tensor_shape)
        targets = torch.randint(0, self.num_classes, (1,)).squeeze()
        return inputs, targets
