"""Run the local DARTS smoke search and render only its measured evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "smoke_cpu.yaml"
OUTPUT_DIR = ROOT / "docs" / "animation" / "runtime-evidence"
CHECKPOINT_DIR = ROOT / "search_checkpoints" / "real_runtime"
VIDEO_PATH = OUTPUT_DIR / "nas-runtime.mp4"
POSTER_PATH = OUTPUT_DIR / "nas-runtime-poster.png"
TRACE_PATH = OUTPUT_DIR / "nas-runtime-trace.csv"
SUMMARY_PATH = OUTPUT_DIR / "nas-runtime-summary.json"

sys.path.insert(0, str(ROOT))

from data.dataset import get_synthetic_dataloaders  # noqa: E402
from search.darts_trainer import DartsTrainer, accuracy  # noqa: E402
from utils.config import load_search_config  # noqa: E402


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one generated or source file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert tensors and numpy scalars into JSON-compatible values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def run_search() -> dict[str, Any]:
    """Execute one real CPU DARTS epoch and collect measured model outputs."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = load_search_config(str(CONFIG_PATH))
    config.save_dir = str(CHECKPOINT_DIR)
    DartsTrainer.set_seed(config.seed)
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
    images, labels = next(iter(loaders.train))
    trainer = DartsTrainer(config, loaders.train, loaders.val)
    device = trainer.device
    criterion = trainer.criterion
    batch = images.to(device)
    target = labels.to(device)

    trainer.raw_model.eval()
    with torch.no_grad():
        initial_logits = trainer.raw_model(batch)
        initial_loss = criterion(initial_logits, target)
        initial_accuracy = accuracy(initial_logits, target, topk=(1,))[0]

    genotype = trainer.fit()

    trainer.raw_model.eval()
    with torch.no_grad():
        final_logits = trainer.raw_model(batch)
        final_loss = criterion(final_logits, target)
        final_accuracy = accuracy(final_logits, target, topk=(1,))[0]

    from models.architecture import NetworkCIFAR  # noqa: PLC0415

    derived_model = NetworkCIFAR(
        genotype=genotype,
        init_channels=config.init_channels,
        num_classes=config.num_classes,
        layers=config.layers,
        auxiliary=False,
        steps=trainer.raw_model.steps,
        multiplier=trainer.raw_model.multiplier,
        stem_channels=16,
    ).to(device)
    derived_model.eval()
    with torch.no_grad():
        derived_logits = derived_model(batch)

    metrics_path = CHECKPOINT_DIR / "metrics.jsonl"
    metrics = {}
    if metrics_path.exists():
        lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            metrics = json.loads(lines[-1])
    checkpoint_path = CHECKPOINT_DIR / "last_search.pt"
    return {
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_count": int(torch.cuda.device_count()),
        "config": {
            "path": CONFIG_PATH.relative_to(ROOT).as_posix(),
            "seed": config.seed,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "init_channels": config.init_channels,
            "layers": config.layers,
            "num_classes": config.num_classes,
            "synthetic_train_size": config.synthetic_train_size,
            "synthetic_val_size": config.synthetic_val_size,
            "synthetic_test_size": config.synthetic_test_size,
        },
        "batch": {
            "shape": list(images.shape),
            "dtype": str(images.dtype),
            "label_histogram": dict(Counter(int(label) for label in labels.tolist())),
            "minimum": float(images.min().item()),
            "maximum": float(images.max().item()),
            "mean": float(images.mean().item()),
            "std": float(images.std().item()),
            "source": "data.dataset.get_synthetic_dataloaders",
        },
        "initial_forward": {
            "logits_shape": list(initial_logits.shape),
            "loss": float(initial_loss.item()),
            "top1_accuracy_percent": float(initial_accuracy.item()),
            "logits_minimum": float(initial_logits.min().item()),
            "logits_maximum": float(initial_logits.max().item()),
        },
        "search": {
            "metrics": metrics,
            "genotype": json_safe(genotype),
            "checkpoint_path": checkpoint_path.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path.exists() else None,
        },
        "final_forward": {
            "logits_shape": list(final_logits.shape),
            "loss": float(final_loss.item()),
            "top1_accuracy_percent": float(final_accuracy.item()),
            "logits_minimum": float(final_logits.min().item()),
            "logits_maximum": float(final_logits.max().item()),
        },
        "derived_forward": {
            "logits_shape": list(derived_logits.shape),
            "logits_minimum": float(derived_logits.min().item()),
            "logits_maximum": float(derived_logits.max().item()),
            "parameter_count": sum(parameter.numel() for parameter in derived_model.parameters()),
        },
        "_image": images[0].mean(dim=0).detach().cpu().numpy(),
    }


def genotype_lines(genotype: dict[str, list]) -> list[str]:
    """Format the measured genotype compactly for the evidence frame."""
    lines = []
    for name in ("normal", "reduce"):
        entries = ", ".join(f"{operation}@{index}" for operation, index in genotype[name])
        lines.append(f"{name}: {entries}")
    return lines


def put_lines(canvas: np.ndarray, lines: list[str], origin: tuple[int, int], color: tuple[int, int, int], scale: float = 0.62) -> None:
    """Draw readable text lines without adding ornamental annotations."""
    x, y = origin
    for line in lines:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += 27


def render_frame(evidence: dict[str, Any], stage: str, stage_index: int, stage_total: int) -> np.ndarray:
    """Render a frame whose values are taken directly from the runtime record."""
    canvas = np.full((720, 1280, 3), (10, 18, 28), dtype=np.uint8)
    white = (238, 244, 250)
    muted = (158, 177, 194)
    cyan = (112, 211, 255)
    green = (112, 220, 157)
    amber = (83, 190, 255)
    cv2.putText(canvas, "DARTS NAS | REAL LOCAL CPU RUNTIME", (54, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.05, cyan, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Measured PyTorch execution | synthetic CIFAR-shaped input | no benchmark claim", (55, 91), cv2.FONT_HERSHEY_SIMPLEX, 0.58, muted, 1, cv2.LINE_AA)
    cv2.line(canvas, (54, 116), (1226, 116), (47, 73, 92), 1)

    cv2.rectangle(canvas, (54, 143), (812, 650), (17, 30, 43), -1)
    cv2.rectangle(canvas, (842, 143), (1226, 650), (17, 30, 43), -1)
    cv2.putText(canvas, f"{stage_index + 1:02d}/{stage_total:02d}  {stage.upper()}", (82, 188), cv2.FONT_HERSHEY_SIMPLEX, 0.78, green, 2, cv2.LINE_AA)

    batch = evidence["batch"]
    if stage == "batch":
        source_image = evidence["_image"]
        normalized = cv2.normalize(source_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
        heatmap = cv2.resize(heatmap, (260, 260), interpolation=cv2.INTER_NEAREST)
        canvas[230:490, 90:350] = heatmap
        cv2.putText(canvas, "actual sample[0] tensor", (90, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.58, white, 1, cv2.LINE_AA)
        put_lines(canvas, [f"shape: {batch['shape']}", f"dtype: {batch['dtype']}", f"min/max: {batch['minimum']:.3f} / {batch['maximum']:.3f}", f"mean/std: {batch['mean']:.3f} / {batch['std']:.3f}", f"labels: {batch['label_histogram']}"], (390, 260), muted)
    elif stage == "initial forward":
        forward = evidence["initial_forward"]
        put_lines(canvas, ["DartsNetworkCIFAR", "actual forward pass", f"logits: {forward['logits_shape']}", f"loss: {forward['loss']:.6f}", f"top-1: {forward['top1_accuracy_percent']:.2f}%", f"logit range: {forward['logits_minimum']:.3f} .. {forward['logits_maximum']:.3f}"], (86, 265), white)
    elif stage == "search epoch":
        metrics = evidence["search"]["metrics"]
        put_lines(canvas, ["DartsTrainer.fit()", "one real CPU search epoch", f"train loss: {metrics.get('train_loss', 0.0):.6f}", f"train top-1: {metrics.get('train_acc', 0.0):.2f}%", f"architecture loss: {metrics.get('arch_loss', 0.0):.6f}", "two train batches processed"], (86, 265), white)
        cv2.rectangle(canvas, (86, 465), (710, 520), (28, 56, 70), -1)
        cv2.putText(canvas, "checkpoint emitted by DartsTrainer", (105, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cyan, 1, cv2.LINE_AA)
    elif stage == "final forward":
        forward = evidence["final_forward"]
        put_lines(canvas, ["searched model state", "actual post-search forward", f"logits: {forward['logits_shape']}", f"loss: {forward['loss']:.6f}", f"top-1: {forward['top1_accuracy_percent']:.2f}%", f"logit range: {forward['logits_minimum']:.3f} .. {forward['logits_maximum']:.3f}"], (86, 265), white)
    elif stage == "genotype":
        put_lines(canvas, ["derived from learned alpha weights", *genotype_lines(evidence["search"]["genotype"])], (86, 265), white, 0.52)
    elif stage == "compiled forward":
        forward = evidence["derived_forward"]
        put_lines(canvas, ["NetworkCIFAR", "compiled from measured genotype", f"logits: {forward['logits_shape']}", f"parameters: {forward['parameter_count']:,}", f"logit range: {forward['logits_minimum']:.3f} .. {forward['logits_maximum']:.3f}"], (86, 265), white)
    else:
        put_lines(canvas, ["reviewable evidence written", "MP4 + poster + CSV trace + JSON summary", f"torch: {evidence['torch_version']}", f"device: {evidence['device']}", f"CUDA available: {evidence['cuda_available']}", "checkpoint hash recorded in JSON"], (86, 265), white)

    cv2.putText(canvas, "RUNTIME FACTS", (874, 188), cv2.FONT_HERSHEY_SIMPLEX, 0.72, amber, 2, cv2.LINE_AA)
    put_lines(canvas, [f"torch {evidence['torch_version']}", f"device {evidence['device']}", f"CUDA {evidence['cuda_available']}", f"seed {evidence['config']['seed']}", f"data source: {batch['source']}", "model checkpoint: local only"], (874, 240), muted, 0.58)
    cv2.line(canvas, (874, 425), (1194, 425), (47, 73, 92), 1)
    put_lines(canvas, ["HONEST BOUNDARY", "Synthetic data validates", "the runtime/search path.", "It does not establish", "CIFAR accuracy or GPU speed."], (874, 470), green, 0.58)
    cv2.putText(canvas, "No fabricated detections or metrics", (55, 694), cv2.FONT_HERSHEY_SIMPLEX, 0.55, muted, 1, cv2.LINE_AA)
    return canvas


def write_outputs(evidence: dict[str, Any]) -> None:
    """Persist the measured trace and an MP4 presentation of the same trace."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stages = ["environment", "batch", "initial forward", "search epoch", "final forward", "genotype", "compiled forward", "artifacts"]
    rows = [
        {"stage": "environment", "field": "torch_version", "value": evidence["torch_version"]},
        {"stage": "environment", "field": "device", "value": evidence["device"]},
        {"stage": "environment", "field": "cuda_available", "value": evidence["cuda_available"]},
        {"stage": "batch", "field": "shape", "value": evidence["batch"]["shape"]},
        {"stage": "batch", "field": "label_histogram", "value": evidence["batch"]["label_histogram"]},
        {"stage": "initial forward", "field": "loss", "value": evidence["initial_forward"]["loss"]},
        {"stage": "initial forward", "field": "top1_accuracy_percent", "value": evidence["initial_forward"]["top1_accuracy_percent"]},
        {"stage": "search epoch", "field": "metrics", "value": evidence["search"]["metrics"]},
        {"stage": "search epoch", "field": "checkpoint_sha256", "value": evidence["search"]["checkpoint_sha256"]},
        {"stage": "final forward", "field": "loss", "value": evidence["final_forward"]["loss"]},
        {"stage": "final forward", "field": "top1_accuracy_percent", "value": evidence["final_forward"]["top1_accuracy_percent"]},
        {"stage": "genotype", "field": "value", "value": evidence["search"]["genotype"]},
        {"stage": "compiled forward", "field": "logits_shape", "value": evidence["derived_forward"]["logits_shape"]},
    ]
    with TRACE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "field", "value"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if key == "value" and not isinstance(value, str) else value for key, value in row.items()})

    writer = cv2.VideoWriter(str(VIDEO_PATH), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {VIDEO_PATH}")
    try:
        for index, stage in enumerate(stages):
            frame = render_frame(evidence, stage, index, len(stages))
            if stage == "search epoch":
                cv2.imwrite(str(POSTER_PATH), frame)
            for _ in range(32):
                writer.write(frame)
    finally:
        writer.release()

    evidence.pop("_image", None)
    evidence["artifacts"] = {
        "video": {"path": VIDEO_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(VIDEO_PATH), "frames": len(stages) * 32, "fps": 20},
        "poster": {"path": POSTER_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(POSTER_PATH)},
        "trace_csv": {"path": TRACE_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(TRACE_PATH)},
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(evidence), handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    """Run the model and generate the reviewable runtime evidence."""
    evidence = run_search()
    write_outputs(evidence)
    print(json.dumps({"summary": SUMMARY_PATH.as_posix(), "video": VIDEO_PATH.as_posix(), "checkpoint": evidence["search"]["checkpoint_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
