"""Logging and TensorBoard setup utilities."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


class NullSummaryWriter:
    """Small no-op SummaryWriter replacement when TensorBoard is unavailable."""

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """Ignore scalar logging while keeping the public API stable."""
        _ = (tag, scalar_value, global_step)

    def add_text(self, tag: str, text_string: str, global_step: int = 0) -> None:
        """Ignore text logging while keeping the public API stable."""
        _ = (tag, text_string, global_step)

    def flush(self) -> None:
        """Flush no-op writer."""
        return None

    def close(self) -> None:
        """Close no-op writer."""
        return None


def get_summary_writer(log_dir: str) -> Any:
    """Create a TensorBoard SummaryWriter with a no-op fallback."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=log_dir)
    except Exception as exc:
        logging.getLogger(__name__).warning("TensorBoard writer disabled: %s", exc)
        return NullSummaryWriter()


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Create a console and file logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    return logger
