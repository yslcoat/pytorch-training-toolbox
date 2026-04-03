import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, spatial_dim, input_dim, hidden_dim, kernel_size, bias=True):
        """
        ConvLSTMCell initialization for N-dimensional inputs. Supports vector input (1D), 2D spatial inputs (images) and 3D volumetric inputs.

        Args:
            spatial_dim (int): Number of spatial dimensions (1, 2, or 3).
            input_dim (int): Number of input channels.
            hidden_dim (int): Number of hidden channels.
            kernel_size (int or tuple[int, ...]): Size of the convolutional kernel.
        """
        super().__init__()

        if spatial_dim not in (1, 2, 3):
            raise ValueError("Invalid spatial dimensions. Supported dimensions are 1, 2, or 3.")

        self.spatial_dim = spatial_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * spatial_dim
        elif len(kernel_size) != spatial_dim:
            raise ValueError(f"kernel_size must be an int or a tuple of length {spatial_dim}")

        padding = tuple(k // 2 for k in kernel_size)

        conv_classes = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
        ConvLayer = conv_classes[spatial_dim]

        self.conv = ConvLayer(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim, # 4 * hidden_dim because we use the same conv_layer for all gates
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, current_state):
        if current_state is None:
            spatial_shape = x.shape[2:]
            current_state = self.init_hidden(x.shape[0], spatial_shape)

        current_hidden_state, current_cell_state = current_state
        combined = torch.cat([x, current_hidden_state], dim=1)
        conv_output = self.conv(combined)

        cc_input, cc_forget, cc_output, cc_candidate = conv_output.chunk(4, dim=1)

        input_gate = torch.sigmoid(cc_input)
        forget_gate = torch.sigmoid(cc_forget)
        output_gate = torch.sigmoid(cc_output)
        cell_candidate = torch.tanh(cc_candidate)

        new_cell_state = (forget_gate * current_cell_state) + (input_gate * cell_candidate)
        new_hidden_state = output_gate * torch.tanh(new_cell_state)

        return new_hidden_state, new_cell_state

    def init_hidden(self, batch_size, spatial_shape):
        device = self.conv.weight.device
        dtype = self.conv.weight.dtype
        return (torch.zeros(batch_size, self.hidden_dim, *spatial_shape, device=device, dtype=dtype),
                torch.zeros(batch_size, self.hidden_dim, *spatial_shape, device=device, dtype=dtype))


class ConvLSTMLayer(nn.Module):
    def __init__(self, spatial_dim, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.cell = ConvLSTMCell(spatial_dim, input_dim, hidden_dim, kernel_size, bias)

    def forward(self, x, hidden_state=None):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Channels, *Spatial_Dims)

        Returns:
            output: Tensor of shape (Batch, Seq_Len, hidden_dim, *Spatial_Dims)
            (h, c): Final hidden and cell state.
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        spatial_shape = x.shape[3:]

        if hidden_state is None:
            hidden_state = self.cell.init_hidden(batch_size, spatial_shape)

        h, c = hidden_state
        output = []

        for t in range(seq_len):
            h, c = self.cell(x[:, t], (h, c))
            output.append(h)

        return torch.stack(output, dim=1), (h, c)


if __name__ == "__main__":
    BATCH_SIZE = 32
    SEQ_LEN = 10
    CHANNELS = 3
    HIDDEN_DIM = 16
    KERNEL_SIZE = 3
    HEIGHT = 64
    WIDTH = 64

    model = ConvLSTMLayer(
        spatial_dim=2,
        input_dim=CHANNELS,
        hidden_dim=HIDDEN_DIM,
        kernel_size=KERNEL_SIZE,
    )

    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, CHANNELS, HEIGHT, WIDTH)

    output, _ = model(input_tensor)
    print(output.shape)
