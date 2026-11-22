# NAS search animation enhancement

## Purpose

Give the repository a visual, reproducible explanation of its DARTS-style workflow: synthetic CIFAR-shaped batches feed a differentiable search network, continuous architecture weights become a genotype, and a discrete network is compiled for evaluation/export.

## Model boundary

The generator references the repository’s actual implementations:

- `data/dataset.py::get_synthetic_dataloaders` creates deterministic offline tensors.
- `models/architecture.py::DartsNetworkCIFAR` executes the mixed-operation search network.
- `search/utils.py::derive_genotype` derives normal/reduction genes.
- `models/architecture.py::NetworkCIFAR` rebuilds the discrete genotype.

When PyTorch is installed, generation runs one real search-network and derived-network forward pass. If the environment lacks PyTorch, the generator falls back to the committed synthetic genotype reference and records that mode explicitly; neither path is presented as accuracy or GPU-performance evidence.

## Reproduce

```bash
python scripts/generate_nas_animation.py
```

Outputs:

- `docs/animation/nas_search_sequence.json` — fixture checksum, model references, execution mode, genotype summary, tensor evidence, and scene events.
- `frontend/public/animation/nas-search-reel.html` — dependency-free animated reference reel with pause, replay, and stage selection.
