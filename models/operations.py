"""DARTS candidate operations and mixed operation implementation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

import torch
import torch.nn as nn

PRIMITIVES: list[str] = [
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "zero",
]


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Drop complete sample paths as used by DARTS evaluation networks."""
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    mask = x.new_empty((x.shape[0], 1, 1, 1)).bernoulli_(keep_prob)
    return x.div(keep_prob) * mask


class ReLUConvBN(nn.Sequential):
    """ReLU, convolution, and batch normalization block."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        affine: bool = True,
    ) -> None:
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )


class DilConv(nn.Sequential):
    """Depthwise separable dilated convolution used in the DARTS search space."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        affine: bool = True,
    ) -> None:
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in,
                c_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=c_in,
                bias=False,
            ),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )


class SepConv(nn.Sequential):
    """Two depthwise separable convolution blocks as used by DARTS."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        affine: bool = True,
    ) -> None:
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in,
                c_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=c_in,
                bias=False,
            ),
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in,
                c_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=c_in,
                bias=False,
            ),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )


class Identity(nn.Module):
    """Identity operation for skip connections."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input tensor unchanged."""
        return x


class Zero(nn.Module):
    """Zero operation that preserves expected tensor dimensions."""

    def __init__(self, stride: int = 1) -> None:
        super().__init__()
        self.stride = int(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return zeros with optional spatial downsampling."""
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
    """Reduce spatial resolution by two while increasing channels."""

    def __init__(self, c_in: int, c_out: int, affine: bool = True) -> None:
        super().__init__()
        if c_out % 2 != 0:
            raise ValueError("FactorizedReduce requires an even number of output channels.")
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c_out, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply factorized channel projection and spatial reduction."""
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(out)


class PoolBN(nn.Sequential):
    """Pooling followed by batch normalization."""

    def __init__(self, pool: nn.Module, channels: int, affine: bool = True) -> None:
        super().__init__(pool, nn.BatchNorm2d(channels, affine=affine))


def _skip_or_reduce(channels: int, stride: int, affine: bool) -> nn.Module:
    """Return identity for stride one or factorized reduction for stride two."""
    if stride == 1:
        return Identity()
    return FactorizedReduce(channels, channels, affine=affine)


OPS: dict[str, Callable[[int, int, bool], nn.Module]] = {
    "sep_conv_3x3": lambda c, stride, affine: SepConv(c, c, 3, stride, 1, affine=affine),
    "sep_conv_5x5": lambda c, stride, affine: SepConv(c, c, 5, stride, 2, affine=affine),
    "dil_conv_3x3": lambda c, stride, affine: DilConv(c, c, 3, stride, 2, 2, affine=affine),
    "dil_conv_5x5": lambda c, stride, affine: DilConv(c, c, 5, stride, 4, 2, affine=affine),
    "max_pool_3x3": lambda c, stride, affine: PoolBN(nn.MaxPool2d(3, stride=stride, padding=1), c, affine=affine),
    "avg_pool_3x3": lambda c, stride, affine: PoolBN(
        nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False), c, affine=affine
    ),
    "skip_connect": lambda c, stride, affine: _skip_or_reduce(c, stride, affine),
    "zero": lambda c, stride, affine: Zero(stride),
}


class MixedOp(nn.Module):
    """Weighted mixture of all primitive operations on one cell edge."""

    def __init__(self, channels: int, stride: int, affine: bool = False) -> None:
        super().__init__()
        self.ops = nn.ModuleList([OPS[name](channels, stride, affine) for name in PRIMITIVES])

    def forward(self, x: torch.Tensor, weights: Iterable[torch.Tensor]) -> torch.Tensor:
        """Compute the weighted sum over all candidate operations."""
        weighted_outputs = [w * op(x) for w, op in zip(weights, self.ops)]
        return sum(weighted_outputs)
