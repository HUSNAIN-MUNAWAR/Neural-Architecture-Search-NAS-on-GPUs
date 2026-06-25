"""GPU detection, memory inspection, and distributed helpers."""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


def configure_cpu_threads(max_threads: int = 2) -> None:
    """Limit CPU threading for tiny smoke runs and constrained CI machines.

    NAS cells execute many small convolutions. On low-resource CI/container
    environments, PyTorch's default thread count can cause severe overhead.
    Limiting CPU threads makes smoke runs predictable without affecting CUDA runs.
    """
    if torch.cuda.is_available():
        return
    threads = max(1, min(int(max_threads), os.cpu_count() or 1))
    try:
        torch.set_num_threads(threads)
    except RuntimeError:
        LOGGER.debug("Could not change torch num_threads after parallel work started.")
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        LOGGER.debug("Could not change torch interop threads after parallel work started.")


def enable_cudnn_benchmark() -> None:
    """Enable CuDNN benchmark mode for stable CIFAR-size input tensors."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


def gpu_memory_stats() -> list[dict[str, float]]:
    """Return memory statistics for every visible CUDA device."""
    if not torch.cuda.is_available():
        return []
    stats: list[dict[str, float]] = []
    for index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        used_bytes = total_bytes - free_bytes
        stats.append(
            {
                "index": float(index),
                "total_gb": total_bytes / 1024**3,
                "free_gb": free_bytes / 1024**3,
                "used_gb": used_bytes / 1024**3,
            }
        )
    return stats


def select_device(prefer_gpu: bool = True) -> torch.device:
    """Select the CUDA device with the most free memory, otherwise CPU."""
    enable_cudnn_benchmark()
    if prefer_gpu and torch.cuda.is_available():
        stats = gpu_memory_stats()
        best = max(stats, key=lambda item: item["free_gb"])
        device_index = int(best["index"])
        torch.cuda.set_device(device_index)
        LOGGER.info("Selected CUDA device %s with %.2f GiB free", device_index, best["free_gb"])
        return torch.device(f"cuda:{device_index}")
    configure_cpu_threads()
    LOGGER.info("CUDA unavailable; selected CPU with limited CPU thread count for predictable smoke/demo runs")
    return torch.device("cpu")


def init_distributed_from_env() -> int | None:
    """Initialize torch.distributed from standard environment variables."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return None
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    LOGGER.info("Initialized DDP rank=%s world_size=%s local_rank=%s", rank, world_size, local_rank)
    return local_rank


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model for DataParallel or DDP wrappers."""
    if hasattr(model, "module"):
        return model.module
    return model


def prepare_model_for_gpu(
    model: nn.Module,
    device: torch.device,
    multi_gpu: bool = True,
    distributed: bool = False,
    local_rank: int | None = None,
) -> nn.Module:
    """Move a model to device and wrap for DataParallel or DDP when requested."""
    model = model.to(device)
    if device.type != "cuda":
        return model
    if distributed:
        rank = 0 if local_rank is None else int(local_rank)
        return nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    if multi_gpu and torch.cuda.device_count() > 1:
        LOGGER.info("Using DataParallel across %s GPUs", torch.cuda.device_count())
        return nn.DataParallel(model)
    return model
