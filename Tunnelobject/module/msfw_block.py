"""Multi-scale feature wavelet fusion block for PyTorch.

``MSFWBlock`` keeps input and output tensors in the same ``[B, C, H, W]`` shape
and combines depthwise-separable convolution, deformable convolution, and fixed
Haar wavelet features. ``MSFWStage`` adds configurable channel changes
and downsampling around the block so it can be stacked stage by stage.
"""

from __future__ import annotations

from typing import Optional, Sequence

try:
    from torchvision.ops import DeformConv2d
except (ImportError, RuntimeError, OSError):
    DeformConv2d = None

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    """Convolution followed by BatchNorm and SiLU."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(c1, c2, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 convolution plus pointwise 1x1 projection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableProjection(nn.Module):
    """Depthwise-separable projection with optional spatial downsampling."""

    def __init__(self, c1: int, c2: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(c1, c1, kernel_size=3, stride=stride, groups=c1),
            ConvBNAct(c1, c2, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeformableConvBranch(nn.Module):
    """Torchvision DCNv2 branch with learned offsets and modulation mask."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if DeformConv2d is None:
            raise ImportError("torchvision.ops.DeformConv2d is required for DeformableConvBranch.")

        self.active_dcn = True
        self.kernel_size = 3

        offset_channels = 2 * self.kernel_size * self.kernel_size
        mask_channels = self.kernel_size * self.kernel_size

        self.offset_mask = nn.Conv2d(
            channels,
            offset_channels + mask_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        self.deform = DeformConv2d(
            channels,
            channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            bias=False,
        )
        self.norm_act = nn.Sequential(nn.BatchNorm2d(channels), nn.SiLU(inplace=True))
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_mask = self.offset_mask(x)
        offset_channels = 2 * self.kernel_size * self.kernel_size
        offset = offset_mask[:, :offset_channels]
        mask = torch.sigmoid(offset_mask[:, offset_channels:])
        return self.norm_act(self.deform(x, offset, mask))


class DeformableProjectionBranch(nn.Module):
    """Deformable projection branch with learned offsets and modulation mask."""

    def __init__(self, c1: int, c2: int, stride: int = 1) -> None:
        super().__init__()
        if DeformConv2d is None:
            raise ImportError("torchvision.ops.DeformConv2d is required for DeformableProjectionBranch.")

        self.active_dcn = True
        self.kernel_size = 3

        offset_channels = 2 * self.kernel_size * self.kernel_size
        mask_channels = self.kernel_size * self.kernel_size
        self.offset_mask = nn.Conv2d(
            c1,
            offset_channels + mask_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=self.kernel_size // 2,
        )
        self.deform = DeformConv2d(
            c1,
            c2,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=self.kernel_size // 2,
            bias=False,
        )
        self.norm_act = nn.Sequential(nn.BatchNorm2d(c2), nn.SiLU(inplace=True))
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_mask = self.offset_mask(x)
        offset_channels = 2 * self.kernel_size * self.kernel_size
        offset = offset_mask[:, :offset_channels]
        mask = torch.sigmoid(offset_mask[:, offset_channels:])
        return self.norm_act(self.deform(x, offset, mask))


class HaarWaveletBranch(nn.Module):
    """Fixed Haar DWT branch for edge and high-frequency texture enhancement."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        haar = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],  # LL
                [[0.5, 0.5], [-0.5, -0.5]],  # LH
                [[0.5, -0.5], [0.5, -0.5]],  # HL
                [[0.5, -0.5], [-0.5, 0.5]],  # HH
            ],
            dtype=torch.float32,
        )
        self.register_buffer("haar_filters", haar[:, None, :, :], persistent=False)
        self.band_fuse = nn.Sequential(
            ConvBNAct(channels * 4, channels, kernel_size=1, padding=0),
            ConvBNAct(channels, channels, kernel_size=3, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x_wave = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            x_wave = x

        filters = self.haar_filters.repeat(c, 1, 1, 1)
        coeffs = F.conv2d(x_wave, filters, stride=2, groups=c)
        coeffs = coeffs.view(b, c, 4, coeffs.shape[-2], coeffs.shape[-1])
        coeffs = coeffs.permute(0, 2, 1, 3, 4).reshape(b, c * 4, coeffs.shape[-2], coeffs.shape[-1])

        fused = self.band_fuse(coeffs)
        fused = F.interpolate(fused, size=(h, w), mode="bilinear", align_corners=False)
        return fused


class HaarDownsampleBranch(nn.Module):
    """Fixed Haar DWT branch that keeps the downsampled feature resolution."""

    def __init__(self, c1: int, c2: int, stride: int = 2) -> None:
        super().__init__()
        if stride < 2:
            raise ValueError("HaarDownsampleBranch expects stride >= 2.")

        haar = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],  # LL
                [[0.5, 0.5], [-0.5, -0.5]],  # LH
                [[0.5, -0.5], [0.5, -0.5]],  # HL
                [[0.5, -0.5], [-0.5, 0.5]],  # HH
            ],
            dtype=torch.float32,
        )
        self.stride = stride
        self.register_buffer("haar_filters", haar[:, None, :, :], persistent=False)
        self.extra_down = nn.AvgPool2d(stride // 2, stride // 2) if stride > 2 and stride % 2 == 0 else None
        self.band_fuse = nn.Sequential(
            ConvBNAct(c1 * 4, c2, kernel_size=1, padding=0),
            ConvBNAct(c2, c2, kernel_size=3, groups=c2),
            ConvBNAct(c2, c2, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, output_size: Optional[tuple[int, int]] = None) -> torch.Tensor:
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        filters = self.haar_filters.repeat(c, 1, 1, 1)
        coeffs = F.conv2d(x, filters, stride=2, groups=c)
        coeffs = coeffs.view(b, c, 4, coeffs.shape[-2], coeffs.shape[-1])
        coeffs = coeffs.permute(0, 2, 1, 3, 4).reshape(b, c * 4, coeffs.shape[-2], coeffs.shape[-1])

        fused = self.band_fuse(coeffs)
        if self.extra_down is not None:
            fused = self.extra_down(fused)
        if output_size is not None and fused.shape[-2:] != output_size:
            fused = F.interpolate(fused, size=output_size, mode="bilinear", align_corners=False)
        return fused


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
    """Multi-scale feature wavelet fusion block.

    Args:
        channels: Number of input and output channels. The block is residual and
            preserves this channel count.
        reduction: Reduction ratio for the lightweight channel gate.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.dw_branch = DepthwiseSeparableConv(channels)
        self.dcn_branch = DeformableConvBranch(channels)
        self.wavelet_branch = HaarWaveletBranch(channels)
        self.fuse = ConvBNAct(channels * 3, channels, kernel_size=1, padding=0)
        self.channel_gate = ChannelGate(channels, reduction=reduction)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        features = torch.cat(
            (
                self.dw_branch(x),
                self.dcn_branch(x),
                self.wavelet_branch(x),
            ),
            dim=1,
        )
        fused = self.channel_gate(self.fuse(features))
        return x + self.gamma * fused


class MSFWStage(nn.Module):
    """Multi-scale feature-wavelet stage with optional spatial downsampling.

    Args:
        c1: Input channels.
        c2: Output channels.
        stride: Projection stride. Use 2 for downsampling, 1 for only changing
            channels or refining features.
        num_blocks: Number of ``MSFWBlock`` modules after the projection.
        reduction: Reduction ratio for ``MSFWBlock`` channel gates.
    """

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
        self.downsample_fusion = stride > 1

        if self.downsample_fusion:
            self.dw_branch = DepthwiseSeparableProjection(c1, c2, stride=stride)
            self.dcn_branch = DeformableProjectionBranch(c1, c2, stride=stride)
            self.wavelet_branch = HaarDownsampleBranch(c1, c2, stride=stride)
            self.fuse = ConvBNAct(c2 * 3, c2, kernel_size=1, padding=0)
            self.channel_gate = ChannelGate(c2, reduction=reduction)
            refinement_blocks = max(num_blocks - 1, 0)
            self.blocks = nn.Sequential(
                *(MSFWBlock(c2, reduction=reduction) for _ in range(refinement_blocks))
            )
        else:
            self.proj = ConvBNAct(c1, c2, kernel_size=3, stride=stride)
            self.blocks = nn.Sequential(
                *(MSFWBlock(c2, reduction=reduction) for _ in range(num_blocks))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.downsample_fusion:
            return self.blocks(self.proj(x))

        dw = self.dw_branch(x)
        dcn = self.dcn_branch(x)
        wavelet = self.wavelet_branch(x, output_size=dw.shape[-2:])
        fused = self.channel_gate(self.fuse(torch.cat((dw, dcn, wavelet), dim=1)))
        return self.blocks(fused)


class MSFWDownsample(nn.Module):
    """Stack a configurable number of ``MSFWStage`` modules."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        num_blocks: int = 1,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels must contain at least one output channel value.")

        stages = []
        c1 = in_channels
        for c2 in channels:
            stages.append(
                MSFWStage(
                    c1,
                    c2,
                    stride=2,
                    num_blocks=num_blocks,
                    reduction=reduction,
                )
            )
            c1 = c2

        self.channels = tuple(channels)
        self.out_channels = channels[-1]
        self.downsample_ratio = 2 ** len(channels)
        self.stages = nn.Sequential(*stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(x)


class MSFWDownsample32(MSFWDownsample):
    """Backward-compatible five-stage 32x downsampler."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        num_blocks: int = 1,
        reduction: int = 4,
    ) -> None:
        if len(channels) != 5:
            raise ValueError("MSFWDownsample32 expects exactly five output channel values.")
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_blocks=num_blocks,
            reduction=reduction,
        )


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
