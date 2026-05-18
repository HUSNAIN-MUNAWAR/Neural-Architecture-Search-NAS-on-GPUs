# Example Artifacts

These files are generated from the offline synthetic smoke configuration and are included only to show artifact formats.

They are **not benchmark results** and must not be used to claim model quality.

| File | Purpose |
|---|---|
| `smoke_genotype.synthetic.json` | Example genotype JSON produced by `configs/smoke_cpu.yaml` |
| `smoke_metrics.synthetic.jsonl` | Example JSONL metric format from a tiny synthetic smoke run |

To generate real artifacts, run CIFAR-10 search/evaluation with `configs/demo_gpu.yaml` or `configs/default_search.yaml` and document the hardware/config in `docs/BENCHMARKS.md`.
