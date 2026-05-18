"""Export trained models to TorchScript and ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


def export_torchscript(model: nn.Module, output_path: str, device: torch.device) -> str:
    """Export a model as a TorchScript traced artifact."""
    model.eval()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, 3, 32, 32, device=device)
    traced = torch.jit.trace(model, example)
    traced.save(str(output))
    return str(output)


def export_onnx(model: nn.Module, output_path: str, device: torch.device) -> str:
    """Export a model as ONNX with dynamic batch axes."""
    model.eval()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, 3, 32, 32, device=device)
    torch.onnx.export(
        model,
        example,
        str(output),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    return str(output)


def export_all(model: nn.Module, output_dir: str, device: torch.device) -> Dict[str, str]:
    """Export both TorchScript and ONNX artifacts."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "torchscript": export_torchscript(model, str(directory / "model.ts"), device),
        "onnx": export_onnx(model, str(directory / "model.onnx"), device),
    }
