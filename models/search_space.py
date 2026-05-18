"""Search and discrete cell definitions for DARTS."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn

from models.operations import FactorizedReduce, MixedOp, OPS, ReLUConvBN, drop_path

Gene = Tuple[str, int]
GenotypeDict = Dict[str, List]


class SearchCell(nn.Module):
    """DARTS search cell with mixed operations on each directed edge."""

    def __init__(
        self,
        steps: int,
        multiplier: int,
        c_prev_prev: int,
        c_prev: int,
        channels: int,
        reduction: bool,
        reduction_prev: bool,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.steps = int(steps)
        self.multiplier = int(multiplier)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, channels, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, channels, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(c_prev, channels, 1, 1, 0, affine=False)
        self.ops = nn.ModuleList()
        for i in range(self.steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                self.ops.append(MixedOp(channels, stride, affine=False))

    def forward(self, s0: torch.Tensor, s1: torch.Tensor, weights: Sequence[torch.Tensor]) -> torch.Tensor:
        """Forward cell computation using architecture edge weights."""
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self.steps):
            new_state = sum(self.ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(new_state)
        return torch.cat(states[-self.multiplier :], dim=1)


class NormalCell(SearchCell):
    """Search cell that preserves spatial dimensions."""

    def __init__(
        self,
        steps: int,
        multiplier: int,
        c_prev_prev: int,
        c_prev: int,
        channels: int,
        reduction_prev: bool,
    ) -> None:
        super().__init__(steps, multiplier, c_prev_prev, c_prev, channels, False, reduction_prev)


class ReductionCell(SearchCell):
    """Search cell that reduces spatial resolution by a factor of two."""

    def __init__(
        self,
        steps: int,
        multiplier: int,
        c_prev_prev: int,
        c_prev: int,
        channels: int,
        reduction_prev: bool,
    ) -> None:
        super().__init__(steps, multiplier, c_prev_prev, c_prev, channels, True, reduction_prev)


class DiscreteCell(nn.Module):
    """Cell compiled from a discrete genotype."""

    def __init__(
        self,
        genotype: GenotypeDict,
        c_prev_prev: int,
        c_prev: int,
        channels: int,
        reduction: bool,
        reduction_prev: bool,
        steps: int = 4,
        multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.steps = int(steps)
        self.multiplier = int(multiplier)
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, channels, affine=True)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, channels, 1, 1, 0, affine=True)
        self.preprocess1 = ReLUConvBN(c_prev, channels, 1, 1, 0, affine=True)
        gene_key = "reduce" if reduction else "normal"
        concat_key = "reduce_concat" if reduction else "normal_concat"
        gene = [tuple(item) for item in genotype[gene_key]]
        self.concat = list(genotype.get(concat_key, range(2, 2 + multiplier)))
        self.ops = nn.ModuleList()
        self.indices: List[int] = []
        for name, index in gene:
            stride = 2 if reduction and int(index) < 2 else 1
            self.ops.append(OPS[str(name)](channels, stride, True))
            self.indices.append(int(index))

    def forward(self, s0: torch.Tensor, s1: torch.Tensor, drop_prob: float = 0.0) -> torch.Tensor:
        """Forward discrete cell with optional path dropout."""
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        op_offset = 0
        for _ in range(self.steps):
            h1 = states[self.indices[op_offset]]
            h2 = states[self.indices[op_offset + 1]]
            op1 = self.ops[op_offset]
            op2 = self.ops[op_offset + 1]
            out1 = op1(h1)
            out2 = op2(h2)
            if self.training and drop_prob > 0.0:
                if op1.__class__.__name__ != "Identity":
                    out1 = drop_path(out1, drop_prob, self.training)
                if op2.__class__.__name__ != "Identity":
                    out2 = drop_path(out2, drop_prob, self.training)
            states.append(out1 + out2)
            op_offset += 2
        return torch.cat([states[i] for i in self.concat], dim=1)
