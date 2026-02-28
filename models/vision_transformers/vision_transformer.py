import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionTransformer(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, emb_dim, n_heads, n_blocks, dropout):
        super(VisionTransformer, self).__init__()

        self.patch_size = patch_size
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.dropout = dropout

        self.to_patch_embedding = None
        self.pos_embedding = None

    def forward(self, x):
        pass