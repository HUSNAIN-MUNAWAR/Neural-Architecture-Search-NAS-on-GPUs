# Reproducibility Guide

NAS experiments are sensitive to hardware, seeds, libraries, and training budget. This project provides reproducibility controls without pretending that every run will be bit-identical.

## Seeds

The trainer sets:

- Python `random`
- NumPy
- PyTorch CPU seed
- PyTorch CUDA seeds when CUDA is available

Configured in YAML:

```yaml
seed: 42
```

## Config Files

Key configs:

- `configs/smoke_cpu.yaml` for quick verification.
- `configs/demo_gpu.yaml` for moderate GPU demos.
- `configs/default_search.yaml` for longer architecture search.
- `configs/default_eval.yaml` for full architecture evaluation.

## Logged Artifacts

Each run can produce:

- `search.log`
- `streamlit_subprocess.log`
- `metrics.jsonl`
- `tensorboard/` event files
- `genotype.json`
- `last_search.pt`
- `best_search.pt`

## Hardware Notes

Record the following before publishing benchmark results:

```bash
python scripts/smoke_test.py
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

## Deterministic Limitations

Full determinism can reduce performance and may not be guaranteed across CUDA kernels, GPU models, and library versions. This repo favors practical reproducibility: fixed configs, logged outputs, and transparent hardware notes.

## Benchmark Template

Use `docs/BENCHMARKS.md` to record measured runs. Accuracy, latency, and parameter counts should be reported only from reproducible evaluation commands and saved artifacts.
