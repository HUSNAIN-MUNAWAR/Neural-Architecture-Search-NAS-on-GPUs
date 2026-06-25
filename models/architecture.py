"""Search and evaluation networks for CIFAR-10 DARTS."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.operations import PRIMITIVES
from models.search_space import DiscreteCell, SearchCell
from search.utils import derive_genotype

Genotype = dict[str, list]


class AuxiliaryHeadCIFAR(nn.Module):
    """Auxiliary classifier head used during full architecture evaluation."""

    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward auxiliary logits."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class DartsNetworkCIFAR(nn.Module):
    """Differentiable DARTS search network for CIFAR-10."""

    def __init__(
        self,
        init_channels: int = 16,
        num_classes: int = 10,
        layers: int = 8,
        criterion: nn.Module | None = None,
        steps: int = 4,
        multiplier: int = 4,
        stem_channels: int = 16,
    ) -> None:
        super().__init__()
        self.init_channels = int(init_channels)
        self.num_classes = int(num_classes)
        self.layers = int(layers)
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.steps = int(steps)
        self.multiplier = int(multiplier)

        c_curr = int(stem_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(3, c_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_curr),
        )
        c_prev_prev = c_curr
        c_prev = c_curr
        c_curr = int(init_channels)
        self.cells = nn.ModuleList()
        reduction_prev = False
        for layer_idx in range(self.layers):
            if layer_idx in (2, 5):
                c_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = SearchCell(
                steps=self.steps,
                multiplier=self.multiplier,
                c_prev_prev=c_prev_prev,
                c_prev=c_prev,
                channels=c_curr,
                reduction=reduction,
                reduction_prev=reduction_prev,
            )
            reduction_prev = reduction
            self.cells.append(cell)
            c_prev_prev, c_prev = c_prev, self.multiplier * c_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, self.num_classes)
        self._initialize_alphas()

    def _initialize_alphas(self) -> None:
        """Create learnable architecture parameters for normal and reduction cells."""
        num_edges = sum(2 + i for i in range(self.steps))
        num_ops = len(PRIMITIVES)
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(num_edges, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(num_edges, num_ops))
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

    def arch_parameters(self) -> list[nn.Parameter]:
        """Return architecture parameters optimized by Adam."""
        return self._arch_parameters

    def weight_parameters(self) -> list[nn.Parameter]:
        """Return network parameters excluding architecture weights."""
        arch_ids = {id(param) for param in self.arch_parameters()}
        return [param for param in self.parameters() if id(param) not in arch_ids]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation through the differentiable architecture."""
        s0 = self.stem(x)
        s1 = s0
        weights_normal = F.softmax(self.alphas_normal, dim=-1)
        weights_reduce = F.softmax(self.alphas_reduce, dim=-1)
        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute classification loss."""
        return self.criterion(self(x), target)

    def genotype(self) -> Genotype:
        """Derive a discrete genotype from learned architecture parameters."""
        return derive_genotype(self.alphas_normal.detach(), self.alphas_reduce.detach())


class NetworkCIFAR(nn.Module):
    """Discrete CIFAR-10 network compiled from a derived DARTS genotype."""

    def __init__(
        self,
        genotype: Genotype,
        init_channels: int = 36,
        num_classes: int = 10,
        layers: int = 20,
        auxiliary: bool = True,
        steps: int = 4,
        multiplier: int = 4,
        stem_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.drop_path_prob = 0.0
        self.layers = int(layers)
        self.auxiliary = bool(auxiliary)
        self.auxiliary_head_index = 2 * self.layers // 3
        stem_out = int(stem_channels or init_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_out),
        )
        c_prev_prev = stem_out
        c_prev = stem_out
        c_curr = int(init_channels)
        self.cells = nn.ModuleList()
        reduction_prev = False
        self.auxiliary_head: AuxiliaryHeadCIFAR | None = None
        for layer_idx in range(self.layers):
            if layer_idx in (self.layers // 3, 2 * self.layers // 3):
                c_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = DiscreteCell(
                genotype=genotype,
                c_prev_prev=c_prev_prev,
                c_prev=c_prev,
                channels=c_curr,
                reduction=reduction,
                reduction_prev=reduction_prev,
                steps=steps,
                multiplier=multiplier,
            )
            reduction_prev = reduction
            self.cells.append(cell)
            c_prev_prev, c_prev = c_prev, multiplier * c_curr
            if self.auxiliary and layer_idx == self.auxiliary_head_index:
                self.auxiliary_head = AuxiliaryHeadCIFAR(c_prev, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward computation returning logits and optional auxiliary logits during training."""
        s0 = self.stem(x)
        s1 = s0
        aux_logits: torch.Tensor | None = None
        for layer_idx, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if self.training and self.auxiliary_head is not None and layer_idx == self.auxiliary_head_index:
                aux_logits = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if aux_logits is not None:
            return logits, aux_logits
        return logits
