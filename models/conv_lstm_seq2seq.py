import torch
import torch.nn as nn

from custom_layers import ConvLSTMLayer


class ConvLSTMSeq2Seq(nn.Module):
    def __init__(self):
        super().__init__()