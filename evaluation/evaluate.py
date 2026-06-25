"""Evaluation utilities for accuracy, parameter count, and latency."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate top-1 accuracy on a dataloader."""
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        predictions = outputs.argmax(dim=1)
        correct += int((predictions == targets).sum().item())
        total += int(targets.numel())
    return 100.0 * correct / max(1, total)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 1,
    warmup: int = 20,
    repeats: int = 100,
) -> float:
    """Measure median-like average inference latency in milliseconds per batch."""
    model.eval()
    inputs = torch.randn(batch_size, 3, 32, 32, device=device)
    for _ in range(warmup):
        _ = model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        _ = model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / max(1, repeats)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Return accuracy, parameter count, and latency metrics."""
    return {
        "test_accuracy": evaluate_accuracy(model, loader, device),
        "parameters": float(count_parameters(model)),
        "latency_ms_batch1": measure_latency(model, device, batch_size=1),
    }
