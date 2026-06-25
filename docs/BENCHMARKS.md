# Benchmarks

This file is the benchmark registry for measured NAS search and evaluation runs. It currently has no published GPU benchmark rows. Add results only when the run can be reproduced from the commit, config, hardware notes, metrics file, and genotype artifact.

## Search Runs

| Date | Commit | GPU | Config | Search epochs | Init channels | Layers | Best train acc | Final genotype path | Metrics path | Notes |
|---|---|---|---|---:|---:|---:|---:|---|---|---|

## Evaluation Runs

| Date | Commit | GPU | Genotype | Eval config | Eval epochs | Test accuracy | Params (M) | Latency (ms) | Export path | Metrics path | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---|---|---|

## Smoke Artifact Examples

The repository includes synthetic smoke artifacts under `examples/` so reviewers can see the expected output formats without assuming benchmark quality.

| Artifact | Meaning | Benchmark? |
|---|---|---:|
| `examples/smoke_genotype.synthetic.json` | Genotype JSON format from a tiny offline synthetic run | No |
| `examples/smoke_metrics.synthetic.jsonl` | JSONL metric format from a tiny offline synthetic run | No |

Synthetic smoke data is random CIFAR-shaped tensor data. It verifies code paths and artifact formats only.

## Required Evidence for Published Numbers

When adding results, include:

- Commit hash.
- YAML config used.
- GPU model and CUDA version.
- TensorBoard or JSONL metrics.
- Genotype JSON.
- Evaluation checkpoint path.
- Exact command used.
