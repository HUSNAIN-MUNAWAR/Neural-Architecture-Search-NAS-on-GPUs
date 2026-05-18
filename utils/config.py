"""YAML configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml


T = TypeVar("T")


@dataclass
class SearchConfig:
    """Configuration for DARTS architecture search."""

    data_root: str = "./data"
    batch_size: int = 64
    val_batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 0.025
    arch_learning_rate: float = 0.0003
    arch_weight_decay: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0003
    unrolled: bool = False
    grad_clip: float = 5.0
    num_workers: int = 4
    save_dir: str = "search_checkpoints"
    seed: int = 42
    init_channels: int = 16
    layers: int = 8
    num_classes: int = 10
    amp: bool = True
    multi_gpu: bool = True
    distributed: bool = False
    cutout: bool = False
    cutout_length: int = 16
    validation_fraction: float = 0.5
    log_interval: int = 50
    use_synthetic_data: bool = False
    synthetic_train_size: int = 64
    synthetic_val_size: int = 64
    synthetic_test_size: int = 64
    download_data: bool = True


@dataclass
class EvalConfig:
    """Configuration for full training and model export."""

    data_root: str = "./data"
    batch_size: int = 96
    val_batch_size: int = 256
    epochs: int = 600
    learning_rate: float = 0.025
    momentum: float = 0.9
    weight_decay: float = 0.0003
    grad_clip: float = 5.0
    num_workers: int = 4
    save_dir: str = "eval_checkpoints"
    seed: int = 42
    init_channels: int = 36
    layers: int = 20
    num_classes: int = 10
    auxiliary: bool = True
    auxiliary_weight: float = 0.4
    drop_path_prob: float = 0.2
    amp: bool = True
    cutout: bool = True
    cutout_length: int = 16
    validation_fraction: float = 0.1
    use_synthetic_data: bool = False
    synthetic_train_size: int = 128
    synthetic_val_size: int = 128
    synthetic_test_size: int = 128
    download_data: bool = True


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file into a Python dictionary."""
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {yaml_path}")
    return data


def _coerce_dataclass(config_cls: Type[T], data: Dict[str, Any]) -> T:
    """Create a dataclass instance while rejecting unknown keys."""
    allowed = {field.name for field in fields(config_cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys for {config_cls.__name__}: {sorted(unknown)}")
    return config_cls(**data)


def load_search_config(path: str) -> SearchConfig:
    """Load and validate a search configuration."""
    return _coerce_dataclass(SearchConfig, load_yaml(path))


def load_eval_config(path: str) -> EvalConfig:
    """Load and validate an evaluation configuration."""
    return _coerce_dataclass(EvalConfig, load_yaml(path))


def save_yaml(data: Dict[str, Any], path: str) -> None:
    """Save a dictionary as YAML."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
