"""Full training pipeline for derived DARTS architectures."""

from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch import amp

from data.dataset import get_dataloaders, get_synthetic_dataloaders
from evaluation.evaluate import evaluate_model
from evaluation.export import export_all
from models.architecture import NetworkCIFAR
from utils.config import EvalConfig
from utils.file_utils import ensure_dir, load_json, save_checkpoint, save_json
from utils.gpu_utils import prepare_model_for_gpu, select_device, unwrap_model
from utils.logging_utils import get_summary_writer, setup_logger


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    config: EvalConfig,
) -> tuple[float, float]:
    """Train a discrete architecture for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    raw_model = unwrap_model(model)
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type="cuda", enabled=config.amp and device.type == "cuda"):
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                logits, aux_logits = outputs
                loss = criterion(logits, targets) + config.auxiliary_weight * criterion(aux_logits, targets)
            else:
                logits = outputs
                loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(raw_model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * targets.size(0)
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += int(targets.numel())
    return total_loss / max(1, total), 100.0 * total_correct / max(1, total)


def train_from_scratch(config: EvalConfig, genotype_path: str) -> dict[str, float]:
    """Train a genotype from scratch and export the best model."""
    set_seed(config.seed)
    save_dir = ensure_dir(config.save_dir)
    logger = setup_logger("darts_eval", str(save_dir / "eval.log"))
    writer = get_summary_writer(str(save_dir / "tensorboard"))
    genotype = load_json(genotype_path)
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
    device = select_device(True)
    model = NetworkCIFAR(
        genotype=genotype,
        init_channels=config.init_channels,
        layers=config.layers,
        num_classes=config.num_classes,
        auxiliary=config.auxiliary,
    )
    model = prepare_model_for_gpu(model, device, multi_gpu=True, distributed=False)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        unwrap_model(model).parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    scaler = amp.GradScaler(enabled=config.amp and device.type == "cuda")
    best_acc = 0.0
    best_metrics: dict[str, float] = {}
    for epoch in range(config.epochs):
        start = time.time()
        unwrap_model(model).drop_path_prob = config.drop_path_prob * epoch / max(1, config.epochs - 1)
        train_loss, train_acc = train_one_epoch(model, loaders.train, criterion, optimizer, scaler, device, config)
        val_metrics = evaluate_model(model, loaders.val, device)
        scheduler.step()
        metrics = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_metrics["test_accuracy"],
            "seconds": time.time() - start,
        }
        for key, value in metrics.items():
            writer.add_scalar(f"eval/{key}", value, epoch)
        logger.info("epoch=%03d metrics=%s", epoch, metrics)
        if metrics["val_acc"] >= best_acc:
            best_acc = metrics["val_acc"]
            best_metrics = metrics
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": unwrap_model(model).state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "genotype": genotype,
                    "metrics": metrics,
                },
                str(save_dir / "best_model.pt"),
            )
    final_metrics = evaluate_model(model, loaders.test, device)
    best_metrics.update({f"test_{key}": value for key, value in final_metrics.items()})
    save_json(best_metrics, str(save_dir / "metrics.json"))
    export_all(unwrap_model(model), str(save_dir / "exports"), device)
    writer.close()
    return best_metrics
