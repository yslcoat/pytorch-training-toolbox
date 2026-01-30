import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, n_hidden_layers: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        # x.shape: [batch_size, input_size]
        x = F.relu(self.input_layer(x))
        x = self.dropout(x) 

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
            x = self.dropout(x)

        x = self.output_layer(x)

        return x
