"""CLI entry point for training a derived architecture from scratch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.train import train_from_scratch
from utils.config import load_eval_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate a derived DARTS architecture.")
    parser.add_argument("--config", default="configs/default_eval.yaml", help="Path to evaluation YAML config.")
    parser.add_argument("--genotype", default="search_checkpoints/genotype.json", help="Path to genotype JSON.")
    return parser.parse_args()


def main() -> None:
    """Run full evaluation training."""
    args = parse_args()
    config = load_eval_config(args.config)
    metrics = train_from_scratch(config, args.genotype)
    print(metrics)


if __name__ == "__main__":
    main()
