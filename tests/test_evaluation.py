"""Tests for evaluation helpers."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)

from evaluation.evaluate import count_parameters, evaluate_accuracy, measure_latency
from models.architecture import NetworkCIFAR
from models.genotypes import DARTS_V1
from search.utils import derive_genotype


def test_genotype_derivation_structure() -> None:
    """Derived genotype should include normal and reduction genes with eight ops each."""
    alphas = torch.randn(14, 8)
    genotype = derive_genotype(alphas, alphas)
    assert len(genotype["normal"]) == 8
    assert len(genotype["reduce"]) == 8
    assert genotype["normal_concat"] == [2, 3, 4, 5]


def test_evaluation_metrics_smoke() -> None:
    """Evaluation helpers should work on a tiny random model and dataset."""
    model = NetworkCIFAR(DARTS_V1, init_channels=2, layers=1, auxiliary=False, stem_channels=2)
    loader = DataLoader(TensorDataset(torch.randn(2, 3, 16, 16), torch.randint(0, 10, (2,))), batch_size=1)
    device = torch.device("cpu")
    acc = evaluate_accuracy(model, loader, device)
    params = count_parameters(model)
    latency = measure_latency(model, device, repeats=1, warmup=1)
    assert 0.0 <= acc <= 100.0
    assert params > 0
    assert latency > 0.0
