"""CLI entry point for DARTS architecture search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import get_dataloaders, get_synthetic_dataloaders
from search.darts_trainer import DartsTrainer
from utils.config import load_search_config
from utils.file_utils import save_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run DARTS search on CIFAR-10.")
    parser.add_argument("--config", default="configs/default_search.yaml", help="Path to search YAML config.")
    return parser.parse_args()


def main() -> None:
    """Run DARTS search from a YAML configuration."""
    args = parse_args()
    config = load_search_config(args.config)
    DartsTrainer.set_seed(config.seed)
    if config.use_synthetic_data:
        loaders = get_synthetic_dataloaders(
            batch_size=config.batch_size,
            val_batch_size=config.val_batch_size,
            num_workers=config.num_workers,
            num_classes=config.num_classes,
            train_size=config.synthetic_train_size,
            val_size=config.synthetic_val_size,
            test_size=config.synthetic_test_size,
            seed=config.seed,
        )
    else:
        loaders = get_dataloaders(
            batch_size=config.batch_size,
            val_batch_size=config.val_batch_size,
            num_workers=config.num_workers,
            data_root=config.data_root,
            validation_fraction=config.validation_fraction,
            seed=config.seed,
            cutout=config.cutout,
            cutout_length=config.cutout_length,
            download=config.download_data,
        )
    trainer = DartsTrainer(config, loaders.train, loaders.val)
    genotype = trainer.fit()
    save_json(genotype, str(Path(config.save_dir) / "genotype.json"))


if __name__ == "__main__":
    main()
