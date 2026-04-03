import warnings
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, spatial_dim, input_dim, hidden_dim, kernel_size, stride=1, padding=None, bias=True):
        """
        ConvLSTMCell for N-dimensional inputs (1D, 2D, 3D).

        Args:
            spatial_dim (int): Number of spatial dimensions (1, 2, or 3).
            input_dim (int): Number of input channels.
            hidden_dim (int): Number of hidden channels.
            kernel_size (int or tuple[int, ...]): Convolutional kernel size. Odd values are strongly recommended
                for symmetric padding; even values will produce an output skewed by 1 pixel per dimension.
            stride (int or tuple[int, ...]): Convolution stride. Use >1 for downsampling.
            padding (int or tuple[int, ...] | None): Explicit padding. Defaults to kernel_size // 2 (same-size output for stride=1).
        """
        super().__init__()

        if spatial_dim not in (1, 2, 3):
            raise ValueError("spatial_dim must be 1, 2, or 3.")

        self.spatial_dim = spatial_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * spatial_dim
        elif len(kernel_size) != spatial_dim:
            raise ValueError(f"kernel_size must be an int or a tuple of length {spatial_dim}")

        if any(k % 2 == 0 for k in kernel_size):
            warnings.warn(
                "Even kernel_size detected. Default padding (k // 2) will not produce symmetric same-size output. "
                "Pass explicit padding or use odd kernel sizes to avoid spatial dimension skew.",
                UserWarning,
                stacklevel=2,
            )

        if isinstance(stride, int):
            stride = (stride,) * spatial_dim
        elif len(stride) != spatial_dim:
            raise ValueError(f"stride must be an int or a tuple of length {spatial_dim}")

        if padding is None:
            padding = tuple(k // 2 for k in kernel_size)
        elif isinstance(padding, int):
            padding = (padding,) * spatial_dim
        elif len(padding) != spatial_dim:
            raise ValueError(f"padding must be an int or a tuple of length {spatial_dim}")

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        conv_classes = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
        ConvLayer = conv_classes[spatial_dim]

        self.conv_x = ConvLayer(input_dim, 4 * hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv_h = ConvLayer(hidden_dim, 4 * hidden_dim, kernel_size=kernel_size, stride=1, padding=tuple(k // 2 for k in kernel_size), bias=False)

        self.norm = nn.GroupNorm(num_groups=4, num_channels=4 * hidden_dim)

        if self.conv_x.bias is not None:
            nn.init.constant_(self.conv_x.bias[hidden_dim: 2 * hidden_dim], 1.0)

    def forward(self, x, current_state):
        if current_state is None:
            current_state = self.init_hidden(x.shape[0], x.shape[2:])

        h, c = current_state
        conv_output = self.norm(self.conv_x(x) + self.conv_h(h))

        cc_input, cc_forget, cc_output, cc_candidate = conv_output.chunk(4, dim=1)

        input_gate = torch.sigmoid(cc_input)
        forget_gate = torch.sigmoid(cc_forget)
        output_gate = torch.sigmoid(cc_output)
        cell_candidate = torch.tanh(cc_candidate)

        new_c = (forget_gate * c) + (input_gate * cell_candidate)
        new_h = output_gate * torch.tanh(new_c)

        return new_h, new_c

    def init_hidden(self, batch_size, input_spatial_shape):
        output_spatial_shape = tuple(
            (s + 2 * p - k) // st + 1
            for s, p, k, st in zip(input_spatial_shape, self.padding, self.kernel_size, self.stride)
        )
        device = self.conv_x.weight.device
        dtype = self.conv_x.weight.dtype
        return (torch.zeros(batch_size, self.hidden_dim, *output_spatial_shape, device=device, dtype=dtype),
                torch.zeros(batch_size, self.hidden_dim, *output_spatial_shape, device=device, dtype=dtype))


class ConvLSTMLayer(nn.Module):
    def __init__(self, spatial_dim, input_dim, hidden_dim, kernel_size, stride=1, padding=None, bias=True):
        super().__init__()
        self.cell = ConvLSTMCell(spatial_dim, input_dim, hidden_dim, kernel_size, stride, padding, bias)

    def forward(self, x, hidden_state=None):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Channels, *Spatial_Dims)

        Returns:
            output: Tensor of shape (Batch, Seq_Len, hidden_dim, *output_Spatial_Dims)
            (h, c): Final hidden and cell state.
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        input_spatial_shape = x.shape[3:]

        if hidden_state is None:
            hidden_state = self.cell.init_hidden(batch_size, input_spatial_shape)

        h, c = hidden_state
        output = []

        for t in range(seq_len):
            h, c = self.cell(x[:, t], (h, c))
            output.append(h)

        return torch.stack(output, dim=1), (h, c)


if __name__ == "__main__":
    BATCH_SIZE = 2
    SEQ_LEN = 10
    CHANNELS = 3
    HIDDEN_DIM = 16
    KERNEL_SIZE = 3
    HEIGHT = 64
    WIDTH = 64

    # stride=1: output spatial shape matches input
    model = ConvLSTMLayer(spatial_dim=2, input_dim=CHANNELS, hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, CHANNELS, HEIGHT, WIDTH)
    out, _ = model(x)
    print("stride=1 output:", out.shape)  # (2, 10, 16, 64, 64)

    # stride=2: output spatial shape is halved
    model_s2 = ConvLSTMLayer(spatial_dim=2, input_dim=CHANNELS, hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE, stride=2, padding=1)
    out_s2, _ = model_s2(x)
    print("stride=2 output:", out_s2.shape)  # (2, 10, 16, 32, 32)
