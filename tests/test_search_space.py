"""Tests for search and discrete cells."""

from __future__ import annotations

import torch

torch.set_num_threads(1)

from models.genotypes import DARTS_V1
from models.operations import PRIMITIVES
from models.search_space import DiscreteCell, SearchCell


def _weights() -> list[torch.Tensor]:
    return [torch.softmax(torch.randn(len(PRIMITIVES)), dim=0) for _ in range(14)]


def test_search_cell_forward_normal() -> None:
    """Normal search cell should preserve spatial dimensions and multiply channels."""
    cell = SearchCell(4, 4, 4, 4, 2, reduction=False, reduction_prev=False)
    y = cell(torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4), _weights())
    assert y.shape == (1, 8, 4, 4)


def test_search_cell_forward_reduction() -> None:
    """Reduction search cell should halve spatial dimensions."""
    cell = SearchCell(4, 4, 4, 4, 2, reduction=True, reduction_prev=False)
    y = cell(torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4), _weights())
    assert y.shape == (1, 8, 2, 2)


def test_discrete_cell_forward() -> None:
    """Discrete cell compiled from genotype should run forward."""
    cell = DiscreteCell(DARTS_V1, 4, 4, 2, reduction=False, reduction_prev=False)
    y = cell(torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4))
    assert y.shape == (1, 8, 4, 4)
