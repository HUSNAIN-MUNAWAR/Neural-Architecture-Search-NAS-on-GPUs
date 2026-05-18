"""Tests for offline dataset utilities."""

from __future__ import annotations

from data.dataset import get_synthetic_dataloaders


def test_synthetic_dataloaders_are_cifar_shaped() -> None:
    """Synthetic smoke loaders should not require downloads and should match CIFAR shapes."""
    loaders = get_synthetic_dataloaders(
        batch_size=2,
        val_batch_size=2,
        num_workers=0,
        train_size=4,
        val_size=4,
        test_size=4,
        seed=123,
    )
    images, labels = next(iter(loaders.train))
    assert images.shape == (2, 3, 32, 32)
    assert labels.shape == (2,)
    assert int(labels.min()) >= 0
    assert int(labels.max()) < 10
