"""Fast project smoke test for imports, config loading, model construction, and one forward pass."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

if not torch.cuda.is_available():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.architecture import DartsNetworkCIFAR, NetworkCIFAR
from search.utils import derive_genotype
from utils.config import load_search_config
from utils.gpu_utils import gpu_memory_stats


def _status() -> dict[str, object]:
    """Collect runtime status for reproducibility notes."""
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_memory": gpu_memory_stats(),
    }


def main() -> None:
    """Run smoke checks without downloading CIFAR-10."""
    config = load_search_config(str(PROJECT_ROOT / "configs" / "smoke_cpu.yaml"))
    criterion = torch.nn.CrossEntropyLoss()
    model = DartsNetworkCIFAR(
        init_channels=config.init_channels,
        num_classes=config.num_classes,
        layers=config.layers,
        criterion=criterion,
    )
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, config.num_classes), logits.shape

    genotype = derive_genotype(model.alphas_normal, model.alphas_reduce)
    derived = NetworkCIFAR(
        genotype=genotype,
        init_channels=4,
        num_classes=config.num_classes,
        layers=2,
        auxiliary=False,
    )
    derived_logits = derived(x)
    assert derived_logits.shape == (2, config.num_classes), derived_logits.shape

    print("Smoke test passed")
    print(json.dumps(_status(), indent=2, default=str))


if __name__ == "__main__":
    main()
