import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class MLP(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

    def forward(self, x):
        pass 


class MSA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class TransformerEncoder(nn.Module):
    def __init__(self, n_heads: int, emb_dim: int, attn_head_dim: int, dropout: float = 0.):
        super().__init__()


    def forward(self, x):
        pass


class VisionTransformer(nn.Module):
    def __init__(self, *, image_size: tuple[int, int] | int, n_channels: int, patch_size: int, num_classes: int, emb_dim: int, n_heads: int, n_blocks: int, attn_head_dim: int, dropout: float = 0., emb_dropout: float = 0.):
        super().__init__()

        self.patch_size = patch_size
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        if image_size[0] % patch_size != 0:
            raise ValueError(f"Image height must be divisible by patch size. Got {image_size[0]} and {patch_size}.")
        if image_size[1] % patch_size != 0:
            raise ValueError(f"Image width must be divisible by patch size. Got {image_size[1]} and {patch_size}.")

        self.n_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        patch_dim = n_channels * patch_size**2

        self.to_patch_embedding = torch.nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), # Flattens input image to patches. Previous shape: (batch_size, n_channels, patch_size, patch_size). Rearranged shape: (batch_size, num_patches, patch_dim)
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )

        self.cls_token = nn.Parameter(torch.randn(1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(self.n_patches + 1, emb_dim)) # +1 for CLS token

        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        # Expected input shape: (batch_size, n_channels, height, width)
        x = self.to_patch_embedding(x)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=x.shape[0]) # Creates copies of CLS token for each image in batch. 'n' is the number of CLS tokens, and b is the batch size inferred from the input. 
        x = torch.cat((cls_tokens, x), dim=1)
        n_tokens = x.shape[1]
        x = x + self.pos_embedding[:n_tokens]
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    model = VisionTransformer(
        image_size=224,
        n_channels=3,
        patch_size=16,
        num_classes=1000,
        emb_dim=768,
        n_heads=12,
        n_blocks=12,
        attn_head_dim=64,
        dropout=0.1
    )

    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)