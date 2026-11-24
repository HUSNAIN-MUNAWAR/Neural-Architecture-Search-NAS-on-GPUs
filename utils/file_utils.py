"""Safe checkpoint and JSON serialization helpers."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import torch


def ensure_dir(path: str) -> Path:
    """Create and return a directory path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_checkpoint(state: dict[str, Any], path: str) -> None:
    """Atomically save a PyTorch checkpoint."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, output)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a PyTorch checkpoint from disk."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Checkpoint could not be loaded safely: {checkpoint_path}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint object in {checkpoint_path}")
    return checkpoint


def save_json(data: dict[str, Any], path: str) -> None:
    """Save JSON data with stable formatting."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, output)


def load_json(path: str) -> dict[str, Any]:
    """Load JSON data from disk."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"JSON file not found: {input_path}")
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON file cannot be read safely: {input_path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {input_path}")
    return data
