from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "animation" / "nas_search_sequence.json"
REEL = ROOT / "frontend" / "public" / "animation" / "nas-search-reel.html"


def test_nas_animation_manifest_is_reproducible() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/generate_nas_animation.py"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "normal edges:" in result.stdout

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    reel = REEL.read_text(encoding="utf-8")
    assert manifest["source"]["synthetic"] is True
    assert manifest["model"]["search_implementation"].endswith("DartsNetworkCIFAR")
    assert manifest["architecture"]["normal_edges"] == 8
    assert manifest["architecture"]["reduction_edges"] == 8
    assert len(manifest["events"]) == 6
    assert "synthetic data only" in reel.lower()
