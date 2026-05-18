"""Safe checkpoint and JSON serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch


def ensure_dir(path: str) -> Path:
    """Create and return a directory path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """Atomically save a PyTorch checkpoint."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(output.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(output)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """Load a PyTorch checkpoint from disk."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=map_location)


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save JSON data with stable formatting."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(output.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    tmp_path.replace(output)


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON data from disk."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"JSON file not found: {input_path}")
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {input_path}")
    return data
