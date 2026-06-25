"""Training loop for first-order DARTS architecture search."""

from __future__ import annotations

import itertools
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader

from models.architecture import DartsNetworkCIFAR
from search.architect import Architect
from search.utils import genotype_to_string
from utils.config import SearchConfig
from utils.file_utils import ensure_dir, save_checkpoint, save_json
from utils.gpu_utils import prepare_model_for_gpu, select_device, unwrap_model
from utils.logging_utils import get_summary_writer, setup_logger


class AverageMeter:
    """Track streaming average values."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all meter statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Add one value to the meter."""
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)


def accuracy(logits: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)) -> tuple[torch.Tensor, ...]:
    """Compute top-k classification accuracy."""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / batch_size))
    return tuple(results)


class DartsTrainer:
    """Orchestrates data, model, architecture optimizer, AMP, and checkpointing."""

    def __init__(self, config: SearchConfig, train_loader: DataLoader, val_loader: DataLoader) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = ensure_dir(config.save_dir)
        self.logger = setup_logger("darts_search", str(self.save_dir / "search.log"))
        self.writer = get_summary_writer(str(self.save_dir / "tensorboard"))
        self.metrics_path = self.save_dir / "metrics.jsonl"
        self.device = select_device(True)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.model = DartsNetworkCIFAR(
            init_channels=config.init_channels,
            num_classes=config.num_classes,
            layers=config.layers,
            criterion=self.criterion,
        )
        self.model = prepare_model_for_gpu(
            self.model,
            self.device,
            multi_gpu=config.multi_gpu,
            distributed=config.distributed,
        )
        self.raw_model = unwrap_model(self.model)
        self.optimizer = torch.optim.SGD(
            self.raw_model.weight_parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, config.epochs)
        self.architect = Architect(self.model, config.arch_learning_rate, config.arch_weight_decay)
        self.scaler = amp.GradScaler(enabled=config.amp and self.device.type == "cuda")

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set deterministic random seeds where practical."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _append_metrics(self, data: dict[str, float]) -> None:
        """Append one metrics record to JSONL for Streamlit monitoring."""
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data) + "\n")

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one DARTS search epoch."""
        self.model.train()
        objs = AverageMeter()
        top1 = AverageMeter()
        arch_losses = AverageMeter()
        val_iter = itertools.cycle(self.val_loader)
        lr = float(self.scheduler.get_last_lr()[0])
        start = time.time()
        for step, (input_train, target_train) in enumerate(self.train_loader):
            input_valid, target_valid = next(val_iter)
            input_train = input_train.to(self.device, non_blocking=True)
            target_train = target_train.to(self.device, non_blocking=True)
            input_valid = input_valid.to(self.device, non_blocking=True)
            target_valid = target_valid.to(self.device, non_blocking=True)

            arch_loss = self.architect.step(
                input_train,
                target_train,
                input_valid,
                target_valid,
                lr,
                self.optimizer,
                self.config.unrolled,
            )
            arch_losses.update(float(arch_loss.item()), input_valid.size(0))

            self.optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", enabled=self.config.amp and self.device.type == "cuda"):
                logits = self.model(input_train)
                loss = self.criterion(logits, target_train)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.raw_model.weight_parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            prec1 = accuracy(logits.detach(), target_train, topk=(1,))[0]
            batch_size = input_train.size(0)
            objs.update(float(loss.item()), batch_size)
            top1.update(float(prec1.item()), batch_size)

            if step % self.config.log_interval == 0:
                self.logger.info(
                    "epoch=%03d step=%04d lr=%.5f train_loss=%.4f train_acc=%.2f arch_loss=%.4f",
                    epoch,
                    step,
                    lr,
                    objs.avg,
                    top1.avg,
                    arch_losses.avg,
                )
        elapsed = time.time() - start
        return {
            "epoch": float(epoch),
            "learning_rate": lr,
            "train_loss": objs.avg,
            "train_acc": top1.avg,
            "arch_loss": arch_losses.avg,
            "seconds": elapsed,
        }

    def fit(self) -> dict[str, list]:
        """Run the architecture search and save checkpoints/genotype."""
        self.set_seed(self.config.seed)
        best_acc = 0.0
        final_genotype: dict[str, list] = {}
        for epoch in range(self.config.epochs):
            metrics = self._train_epoch(epoch)
            self.scheduler.step()
            genotype = self.raw_model.genotype()
            final_genotype = genotype
            for key, value in metrics.items():
                self.writer.add_scalar(f"search/{key}", value, epoch)
            self.writer.add_text("search/genotype", genotype_to_string(genotype), epoch)
            self._append_metrics(metrics)
            checkpoint = {
                "epoch": epoch,
                "model_state": self.raw_model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "arch_optimizer_state": self.architect.optimizer.state_dict(),
                "genotype": genotype,
                "metrics": metrics,
            }
            save_checkpoint(checkpoint, str(self.save_dir / "last_search.pt"))
            if metrics["train_acc"] >= best_acc:
                best_acc = metrics["train_acc"]
                save_checkpoint(checkpoint, str(self.save_dir / "best_search.pt"))
            save_json(genotype, str(self.save_dir / "genotype.json"))
            self.logger.info("epoch=%03d summary=%s", epoch, metrics)
        self.writer.close()
        self.logger.info("Final genotype:\n%s", genotype_to_string(final_genotype))
        return final_genotype
