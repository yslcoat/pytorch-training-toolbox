import warnings
import torch
import torch.nn as nn


class ConvLSTMTransposeCell(nn.Module):
    def __init__(self, spatial_dim, input_dim, hidden_dim, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        """
        Transposed ConvLSTMCell for N-dimensional inputs (1D, 2D, 3D).
        Mirrors ConvLSTMCell but uses transposed convolutions for upsampling.

        Output spatial size per dimension:
            (input - 1) * stride - 2 * padding + kernel_size + output_padding

        Note: ConvTranspose with stride > 1 can produce checkerboard artifacts in generated outputs.
        If you observe grid-like patterns, consider replacing conv_x with nn.Upsample followed by a
        standard Conv (stride=1, padding='same'), which avoids the uneven overlap pattern.

        Args:
            spatial_dim (int): Number of spatial dimensions (1, 2, or 3).
            input_dim (int): Number of input channels.
            hidden_dim (int): Number of hidden channels.
            kernel_size (int or tuple[int, ...]): Convolutional kernel size. Odd values are strongly
                recommended for symmetric padding.
            stride (int or tuple[int, ...]): Upsampling stride.
            padding (int or tuple[int, ...]): Padding applied to input.
            output_padding (int or tuple[int, ...]): Extra padding added to output size. Use to match exact encoder spatial dims.
        """
        super().__init__()

        if spatial_dim not in (1, 2, 3):
            raise ValueError("spatial_dim must be 1, 2, or 3.")

        self.spatial_dim = spatial_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        def _to_tuple(val, name):
            if isinstance(val, int):
                return (val,) * spatial_dim
            if len(val) != spatial_dim:
                raise ValueError(f"{name} must be an int or a tuple of length {spatial_dim}")
            return tuple(val)

        self.kernel_size = _to_tuple(kernel_size, "kernel_size")
        self.stride = _to_tuple(stride, "stride")
        self.padding = _to_tuple(padding, "padding")
        self.output_padding = _to_tuple(output_padding, "output_padding")

        if any(k % 2 == 0 for k in self.kernel_size):
            warnings.warn(
                "Even kernel_size detected. Default padding (k // 2) for conv_h will not produce symmetric same-size output. "
                "Use odd kernel sizes to avoid spatial dimension skew.",
                UserWarning,
                stacklevel=2,
            )

        deconv_classes = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
        conv_classes = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
        DeconvLayer = deconv_classes[spatial_dim]
        ConvLayer = conv_classes[spatial_dim]

        self.conv_x = DeconvLayer(input_dim, 4 * hidden_dim, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=self.output_padding, bias=bias)
        self.conv_h = ConvLayer(hidden_dim, 4 * hidden_dim, kernel_size=self.kernel_size, stride=1, padding=tuple(k // 2 for k in self.kernel_size), bias=False)

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
            (s - 1) * st - 2 * p + k + op
            for s, st, p, k, op in zip(
                input_spatial_shape, self.stride, self.padding, self.kernel_size, self.output_padding
            )
        )
        device = self.conv_x.weight.device
        dtype = self.conv_x.weight.dtype
        return (torch.zeros(batch_size, self.hidden_dim, *output_spatial_shape, device=device, dtype=dtype),
                torch.zeros(batch_size, self.hidden_dim, *output_spatial_shape, device=device, dtype=dtype))


class ConvLSTMTransposeLayer(nn.Module):
    def __init__(self, spatial_dim, input_dim, hidden_dim, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        super().__init__()
        self.cell = ConvLSTMTransposeCell(spatial_dim, input_dim, hidden_dim, kernel_size, stride, padding, output_padding, bias)

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
    from conv_lstm import ConvLSTMLayer

    BATCH_SIZE = 2
    SEQ_LEN = 10
    CHANNELS = 3
    HIDDEN_DIM = 16
    KERNEL_SIZE = 3
    HEIGHT = 64
    WIDTH = 64

    # Encoder: 64x64 -> 32x32
    encoder = ConvLSTMLayer(spatial_dim=2, input_dim=CHANNELS, hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE, stride=2, padding=1)
    # Decoder: 32x32 -> 64x64
    decoder = ConvLSTMTransposeLayer(spatial_dim=2, input_dim=HIDDEN_DIM, hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE, stride=2, padding=1, output_padding=1)

    x = torch.randn(BATCH_SIZE, SEQ_LEN, CHANNELS, HEIGHT, WIDTH)

    enc_out, _ = encoder(x)
    print("Encoder output:", enc_out.shape)

    dec_out, _ = decoder(enc_out)
    print("Decoder output:", dec_out.shape)
