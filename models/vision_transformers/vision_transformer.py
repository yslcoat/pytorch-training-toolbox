import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionTransformer(nn.Module):
    def __init__(self, *, image_size: tuple[int, int] | int, patch_size: int, num_classes: int, emb_dim: int, n_heads: int, n_blocks: int, dropout: float):
        super(VisionTransformer, self).__init__()

        self.patch_size = patch_size
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.dropout = dropout

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        if image_size[0] % patch_size != 0:
            raise ValueError(f"Image height must be divisible by patch size. Got {image_size[0]} and {patch_size}.")
        if image_size[1] % patch_size != 0:
            raise ValueError(f"Image width must be divisible by patch size. Got {image_size[1]} and {patch_size}.")

        self.n_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        self.to_patch_embedding = None
        self.pos_embedding = None

    def forward(self, x):
        # Expected input shape: (batch_size, n_channels, height, width)
        pass


if __name__ == "__main__":
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        emb_dim=768,
        n_heads=12,
        n_blocks=12,
        dropout=0.1
    )

    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)