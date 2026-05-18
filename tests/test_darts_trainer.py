"""Smoke tests for DARTS trainer components."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)

from search.darts_trainer import DartsTrainer
from utils.config import SearchConfig


def test_darts_trainer_one_step(tmp_path) -> None:
    """A one-epoch tiny search should update architecture parameters and save genotype."""
    x = torch.randn(4, 3, 8, 8)
    y = torch.randint(0, 10, (4,))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    config = SearchConfig(
        epochs=1,
        batch_size=2,
        val_batch_size=2,
        num_workers=0,
        save_dir=str(tmp_path),
        init_channels=2,
        layers=1,
        amp=False,
        multi_gpu=False,
        log_interval=1,
    )
    trainer = DartsTrainer(config, loader, loader)
    before = [p.detach().clone() for p in trainer.raw_model.arch_parameters()]
    genotype = trainer.fit()
    after = trainer.raw_model.arch_parameters()
    assert isinstance(genotype, dict)
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    assert (tmp_path / "genotype.json").exists()
