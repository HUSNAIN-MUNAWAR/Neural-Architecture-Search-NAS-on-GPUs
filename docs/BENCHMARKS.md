# Benchmarks

This file intentionally avoids fake benchmark claims. Fill it only after a real measured run.

## Search Runs

| Date | GPU | Config | Search epochs | Init channels | Layers | Best train acc | Final genotype path | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| To be filled after GPU run | To be filled | `configs/demo_gpu.yaml` | 5 | 16 | 8 | To be measured | `search_checkpoints/demo_gpu/genotype.json` | Placeholder row |

## Evaluation Runs

| Date | GPU | Genotype | Eval epochs | Test accuracy | Params (M) | Latency (ms) | Export path | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| To be filled after GPU run | To be filled | To be filled | To be filled | To be measured | To be measured | To be measured | To be filled | Placeholder row |

## Smoke Artifact Examples

The repository includes synthetic smoke artifacts under `examples/` so reviewers can see the expected output formats without assuming benchmark quality.

| Artifact | Meaning | Benchmark? |
|---|---|---:|
| `examples/smoke_genotype.synthetic.json` | Genotype JSON format from a tiny offline synthetic run | No |
| `examples/smoke_metrics.synthetic.jsonl` | JSONL metric format from a tiny offline synthetic run | No |

Synthetic smoke data is random CIFAR-shaped tensor data. It verifies code plumbing only and must never be presented as CIFAR-10 accuracy.

## Required Evidence for Published Numbers

When adding results, include:

- Commit hash.
- YAML config used.
- GPU model and CUDA version.
- TensorBoard or JSONL metrics.
- Genotype JSON.
- Evaluation checkpoint path.
- Exact command used.
