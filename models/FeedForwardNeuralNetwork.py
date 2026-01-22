import torch
import torch.nn as nn


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, output_size: int):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x: torch.Tensor):
        # x.shape: [batch_size, input_size]
        return self.layers(x)
