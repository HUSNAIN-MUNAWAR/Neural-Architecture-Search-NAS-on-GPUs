"""Unit tests for candidate operations."""

from __future__ import annotations

import torch

torch.set_num_threads(1)

from models.operations import OPS, PRIMITIVES, FactorizedReduce, Identity, MixedOp, Zero


def test_all_operations_stride_one_shape() -> None:
    """Every primitive should preserve shape at stride one."""
    x = torch.randn(1, 4, 8, 8)
    for name in PRIMITIVES:
        op = OPS[name](4, 1, True)
        y = op(x)
        assert y.shape == x.shape


def test_all_operations_stride_two_shape() -> None:
    """Every primitive should downsample spatial size at stride two."""
    x = torch.randn(1, 4, 8, 8)
    for name in PRIMITIVES:
        op = OPS[name](4, 2, True)
        y = op(x)
        assert y.shape == (1, 4, 4, 4)


def test_identity_returns_same_values() -> None:
    """Identity should return exactly the same tensor values."""
    x = torch.randn(1, 4, 8, 8)
    assert torch.equal(Identity()(x), x)


def test_zero_outputs_zeros() -> None:
    """Zero operation should produce an all-zero tensor."""
    x = torch.randn(1, 4, 8, 8)
    y = Zero(2)(x)
    assert y.shape == (1, 4, 4, 4)
    assert torch.count_nonzero(y).item() == 0


def test_factorized_reduce_shape() -> None:
    """FactorizedReduce should halve spatial dimensions and change channels."""
    x = torch.randn(1, 4, 8, 8)
    y = FactorizedReduce(4, 8)(x)
    assert y.shape == (1, 8, 4, 4)


def test_mixed_op_shape() -> None:
    """MixedOp should return a valid weighted mixture tensor."""
    x = torch.randn(1, 4, 8, 8)
    weights = torch.softmax(torch.randn(len(PRIMITIVES)), dim=0)
    y = MixedOp(4, 1)(x, weights)
    assert y.shape == x.shape
