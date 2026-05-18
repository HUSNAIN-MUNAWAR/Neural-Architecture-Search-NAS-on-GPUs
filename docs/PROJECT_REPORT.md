# Project Report: Neural Architecture Search on GPUs

## 1. Introduction

Neural Architecture Search automates the process of designing neural networks. This repository implements a DARTS-style search system for CIFAR-10 with a focus on credible ML engineering: modular code, GPU-aware training, reproducible configuration, monitoring, UI control, and deployment assets.

## 2. Design Goals

The repository is designed to demonstrate:

- AutoML research knowledge.
- Production-oriented PyTorch engineering.
- Dashboarding and experiment monitoring.
- Deployment readiness using Docker and VPS patterns.
- Honest benchmark reporting without fake numbers.

## 3. System Architecture

The pipeline is:

1. CIFAR-10 dataloader with train/validation/test transforms.
2. Search network built from DARTS MixedOps.
3. DARTS trainer updates network weights and architecture weights separately.
4. Genotype decoder converts alpha tensors into a discrete architecture.
5. Evaluation pipeline trains a derived network from scratch.
6. Export pipeline writes ONNX and TorchScript artifacts.
7. Streamlit dashboard launches and monitors local jobs.

See `docs/architecture.png` and `docs/ARCHITECTURE.md`.

## 4. Search Space

Each search cell uses candidate operations:

- SepConv 3x3
- SepConv 5x5
- DilConv 3x3
- DilConv 5x5
- MaxPool 3x3
- AvgPool 3x3
- Skip connection
- Zero

Architecture parameters are learned per edge. The final genotype selects the strongest non-zero operations.

## 5. GPU Acceleration

The project uses:

- CUDA device selection.
- CuDNN benchmark mode.
- Automatic mixed precision.
- Pinned-memory dataloaders.
- Non-blocking tensor transfers.
- Gradient clipping.
- DataParallel support and DDP helper utilities.

## 6. Streamlit Control Plane

The Streamlit app provides:

- Search launch presets.
- Subprocess-based training execution.
- PID/status display.
- Live log viewer.
- JSONL and TensorBoard metric plots.
- Genotype JSON viewer.
- Cell diagram rendering.
- Artifact downloads.
- Deployment guidance.

## 7. Deployment

Deployment paths include:

- Local Streamlit.
- Docker Compose.
- GPU Docker Compose.
- VPS with Docker restart policy.
- VPS with systemd and Nginx reverse proxy.

## 8. Limitations

This is a research-grade DARTS baseline and ML engineering portfolio system. It does not claim state-of-the-art performance. Real benchmarks require a GPU run and should be added to `docs/BENCHMARKS.md` with evidence.
