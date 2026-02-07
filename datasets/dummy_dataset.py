import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, n_samples: int, inputs_tensor_shape: tuple, targets_tensor_shape: tuple):
        self.n_samples = n_samples
        self.inputs_tensor_shape = inputs_tensor_shape
        self.targets_tensor_shape = targets_tensor_shape

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        inputs = torch.randn(self.inputs_tensor_shape)
        targets = torch.randn(self.targets_tensor_shape)
        return inputs, targets
