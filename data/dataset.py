"""CIFAR-10 and synthetic data pipelines with GPU-efficient dataloaders."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

try:
    from torchvision import datasets, transforms
except Exception as exc:  # pragma: no cover - depends on local torchvision/CUDA build
    datasets = None
    transforms = None
    _TORCHVISION_IMPORT_ERROR = exc
else:
    _TORCHVISION_IMPORT_ERROR = None


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class Cutout:
    """Randomly mask out one square region of an image tensor."""

    def __init__(self, length: int = 16, probability: float = 1.0) -> None:
        self.length = int(length)
        self.probability = float(probability)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply cutout to a normalized CHW image tensor."""
        if self.length <= 0 or random.random() > self.probability:
            return img
        height = img.size(1)
        width = img.size(2)
        y = np.random.randint(height)
        x = np.random.randint(width)
        y1 = int(np.clip(y - self.length // 2, 0, height))
        y2 = int(np.clip(y + self.length // 2, 0, height))
        x1 = int(np.clip(x - self.length // 2, 0, width))
        x2 = int(np.clip(x + self.length // 2, 0, width))
        img[:, y1:y2, x1:x2] = 0.0
        return img


@dataclass(frozen=True)
class DataLoaderBundle:
    """Container for train, validation, and test dataloaders."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


def _split_train_val(
    dataset: Dataset,
    validation_fraction: float,
    seed: int,
) -> Tuple[Subset, Subset]:
    """Create a deterministic train/validation split from a dataset."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError("Dataset must contain at least two samples for train/validation split.")
    indices = list(range(dataset_size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = max(1, int(np.floor(validation_fraction * dataset_size)))
    split = min(split, dataset_size - 1)
    val_indices = indices[:split]
    train_indices = indices[split:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_transforms(
    cutout: bool = False,
    cutout_length: int = 16,
) -> Tuple[Any, Any]:
    """Create training and evaluation transforms."""
    if transforms is None:
        raise ImportError(
            "torchvision could not be imported. Install a PyTorch/torchvision build that matches your CUDA/CPU runtime."
        ) from _TORCHVISION_IMPORT_ERROR
    train_ops = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    if cutout:
        train_ops.append(Cutout(cutout_length))
    train_transform = transforms.Compose(train_ops)
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_transform, eval_transform


def _make_synthetic_dataset(size: int, num_classes: int, seed: int) -> TensorDataset:
    """Create deterministic CIFAR-shaped tensors for offline smoke tests.

    The synthetic dataset is intentionally small and is not a benchmark. It exists
    so CI, Docker, recruiters, and clients can verify the full training/search
    plumbing without downloading CIFAR-10 or requiring internet access.
    """
    if size < 1:
        raise ValueError("Synthetic dataset size must be positive.")
    generator = torch.Generator().manual_seed(int(seed))
    images = torch.randn(size, 3, 32, 32, generator=generator)
    labels = torch.randint(low=0, high=int(num_classes), size=(size,), generator=generator)
    return TensorDataset(images, labels)


def _make_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, drop_last: bool) -> DataLoader:
    """Create a dataloader with safe worker/pinned-memory settings."""
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )


def get_synthetic_dataloaders(
    batch_size: int,
    val_batch_size: Optional[int] = None,
    num_workers: int = 0,
    num_classes: int = 10,
    train_size: int = 64,
    val_size: int = 64,
    test_size: int = 64,
    seed: int = 42,
) -> DataLoaderBundle:
    """Build deterministic offline dataloaders for smoke tests and CI.

    These loaders should not be used to claim model accuracy. They only verify
    that data, model, optimizer, metrics, logging, and export paths are wired.
    """
    val_batch_size = int(val_batch_size or batch_size)
    train_dataset = _make_synthetic_dataset(train_size, num_classes, seed)
    val_dataset = _make_synthetic_dataset(val_size, num_classes, seed + 1)
    test_dataset = _make_synthetic_dataset(test_size, num_classes, seed + 2)
    return DataLoaderBundle(
        train=_make_loader(train_dataset, batch_size, num_workers, shuffle=True, drop_last=True),
        val=_make_loader(val_dataset, val_batch_size, num_workers, shuffle=False, drop_last=True),
        test=_make_loader(test_dataset, val_batch_size, num_workers, shuffle=False, drop_last=False),
    )


def get_dataloaders(
    batch_size: int,
    num_workers: int,
    data_root: str,
    val_batch_size: Optional[int] = None,
    validation_fraction: float = 0.5,
    seed: int = 42,
    cutout: bool = False,
    cutout_length: int = 16,
    download: bool = True,
) -> DataLoaderBundle:
    """Build CIFAR-10 dataloaders with pinned memory for CUDA training."""
    if datasets is None:
        raise ImportError(
            "torchvision datasets are unavailable. Install a compatible torchvision package before downloading CIFAR-10."
        ) from _TORCHVISION_IMPORT_ERROR
    train_transform, eval_transform = build_transforms(cutout=cutout, cutout_length=cutout_length)
    train_dataset_aug = datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=train_transform,
        download=download,
    )
    train_dataset_eval = datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=eval_transform,
        download=download,
    )
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        transform=eval_transform,
        download=download,
    )

    train_subset, _ = _split_train_val(train_dataset_aug, validation_fraction, seed)
    _, val_subset = _split_train_val(train_dataset_eval, validation_fraction, seed)
    val_batch_size = int(val_batch_size or batch_size)

    return DataLoaderBundle(
        train=_make_loader(train_subset, batch_size, num_workers, shuffle=True, drop_last=True),
        val=_make_loader(val_subset, val_batch_size, num_workers, shuffle=True, drop_last=True),
        test=_make_loader(test_dataset, val_batch_size, num_workers, shuffle=False, drop_last=False),
    )
