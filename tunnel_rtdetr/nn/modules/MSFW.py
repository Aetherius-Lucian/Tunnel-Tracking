"""MSFW. RT-DETR detection workflow."""

from __future__ import annotations

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Convolution followed by BatchNorm and SiLU."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size=1,
        stride: int = 1,
        padding=None,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            else:
                padding = kernel_size // 2

        self.conv = nn.Conv2d(c1, c2, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


def create_wavelet_filter(
    wave: str,
    in_channels: int,
    out_channels: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create fixed 2D wavelet decomposition and reconstruction filters."""
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    dec_filters = torch.stack(
        [
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
        ],
        dim=0,
    )
    dec_filters = dec_filters[:, None].repeat(in_channels, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])
    rec_filters = torch.stack(
        [
            rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
        ],
        dim=0,
    )
    rec_filters = rec_filters[:, None].repeat(out_channels, 1, 1, 1)
    return dec_filters, rec_filters


def wavelet_transform(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """Apply grouped 2D DWT and return [B, C, 4, H/2, W/2] coefficients."""
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    return x.reshape(b, c, 4, h // 2, w // 2)


def inverse_wavelet_transform(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """Apply grouped inverse 2D DWT from [B, C, 4, H/2, W/2] coefficients."""
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    return F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)


class ScaleModule(nn.Module):
    """Learnable per-channel scale used by WTConv2d."""

    def __init__(self, dims: list[int], init_scale: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x


class WTConv2d(nn.Module):
    """Wavelet convolution using DWT, sub-band depthwise conv, and IDWT."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        bias: bool = True,
        wt_levels: int = 1,
        wt_type: str = "db1",
    ) -> None:
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("WTConv2d requires in_channels == out_channels.")

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        wt_filter, iwt_filter = create_wavelet_filter(wt_type, in_channels, out_channels)
        self.register_buffer("wt_filter", wt_filter, persistent=False)
        self.register_buffer("iwt_filter", iwt_filter, persistent=False)

        padding = (kernel_size - 1) // 2
        self.base_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            stride=1,
            dilation=1,
            groups=in_channels,
            bias=bias,
        )
        self.base_scale = ScaleModule([1, in_channels, 1, 1])
        self.wavelet_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels * 4,
                    in_channels * 4,
                    kernel_size,
                    padding=padding,
                    stride=1,
                    dilation=1,
                    groups=in_channels * 4,
                    bias=False,
                )
                for _ in range(wt_levels)
            ]
        )
        self.wavelet_scale = nn.ModuleList(
            [ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(wt_levels)]
        )
        self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride) if stride > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if curr_shape[2] % 2 or curr_shape[3] % 2:
                curr_x_ll = F.pad(curr_x_ll, (0, curr_shape[3] % 2, 0, curr_shape[2] % 2))

            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        for _ in range(self.wt_levels):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, : curr_shape[2], : curr_shape[3]]

        x = self.base_scale(self.base_conv(x)) + next_x_ll
        if self.do_stride is not None:
            x = self.do_stride(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 convolution plus pointwise 1x1 projection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=5, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WaveletConvBranch(nn.Module):
    """Wavelet convolution branch with BN and SiLU."""

    def __init__(self, channels: int, kernel_size: int = 7, wt_levels: int = 1, wt_type: str = "db1") -> None:
        super().__init__()
        self.block = nn.Sequential(
            WTConv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=1,
                wt_levels=wt_levels,
                wt_type=wt_type,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StripContextBranch(nn.Module):
    """Directional strip convolution branch for efficient long-range context."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            ConvBNAct(
                channels,
                channels,
                kernel_size=(1, kernel_size),
                padding=(0, padding),
            ),
            ConvBNAct(
                channels,
                channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ChannelGate(nn.Module):
    """Lightweight squeeze-excitation gate."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class MSFWBlock(nn.Module):
    """Multi-scale feature fusion block.

    Args:
        channels: Number of input and output channels. The block is residual and
            preserves this channel count.
        reduction: Reduction ratio for the lightweight channel gate.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.dw_branch = DepthwiseSeparableConv(channels)
        self.wavelet_branch = WaveletConvBranch(channels)
        self.strip_branch = StripContextBranch(channels)
        self.fuse = ConvBNAct(channels * 3, channels, kernel_size=1, padding=0)
        self.channel_gate = ChannelGate(channels, reduction=reduction)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        features = torch.cat(
            (
                self.dw_branch(x),
                self.wavelet_branch(x),
               self.strip_branch(x),
            ),
            dim=1,
        )
        # features =  self.strip_branch(x)
        fused = self.channel_gate(self.fuse(features))

        # return features
        return x + self.gamma * fused


class MSFWStage(nn.Module):
    """MSFWStage. RT-DETR detection workflow."""

    def __init__(
        self,
        c1: int,
        c2: int,
        stride: int = 2,
        num_blocks: int = 1,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        if num_blocks < 0:
            raise ValueError("num_blocks must be greater than or equal to 0.")

        self.c1 = c1
        self.c2 = c2
        self.stride = stride
        self.proj = ConvBNAct(c1, c2, kernel_size=3, stride=stride)
        self.blocks = nn.Sequential(
            *(MSFWBlock(c2, reduction=reduction) for _ in range(num_blocks))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))

if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = device.type

    block = MSFWStage(3, 32, stride=2, num_blocks=1).to(device)
    image = torch.randn(1, 3, 640, 640, device=device, requires_grad=True)
    feature = block(image)
    feature.mean().backward()
    print(f"single_stage output_shape={tuple(feature.shape)}")
    summary(block, input_size=(3, 640, 640), device=device_name)
