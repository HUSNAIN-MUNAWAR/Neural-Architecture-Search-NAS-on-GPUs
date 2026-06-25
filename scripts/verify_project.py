"""Repository verification script for required files and lightweight import checks."""

from __future__ import annotations

import importlib
import importlib.util
import py_compile
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "Dockerfile",
    "Dockerfile.cpu",
    "docker-compose.yml",
    "docker-compose.gpu.yml",
    ".dockerignore",
    ".gitignore",
    "pyproject.toml",
    "configs/default_search.yaml",
    "configs/default_eval.yaml",
    "configs/smoke_cpu.yaml",
    "configs/demo_gpu.yaml",
    "streamlit_app/app.py",
    "docs/DEPLOYMENT.md",
    "docs/REPRODUCIBILITY.md",
    "docs/BENCHMARKS.md",
    "docs/PROJECT_REPORT.md",
    "docs/PORTFOLIO_PITCH.md",
    "docs/UPWORK_PROPOSAL_SNIPPETS.md",
    "docs/ARCHITECTURE.md",
    "docs/DEMO_GUIDE.md",
    "docs/PUBLISHING_CHECKLIST.md",
    "examples/README.md",
    "examples/smoke_genotype.synthetic.json",
    "examples/smoke_metrics.synthetic.jsonl",
    "deployment/vps/README.md",
    "deployment/vps/systemd/nas-streamlit.service",
    "deployment/vps/nginx/nas-streamlit.conf",
    ".github/workflows/ci.yml",
]

IMPORTS = [
    "data.dataset",
    "models.operations",
    "models.search_space",
    "models.architecture",
    "search.darts_trainer",
    "evaluation.evaluate",
    "evaluation.export",
    "utils.config",
    "utils.gpu_utils",
    "streamlit_app.app",
]


def _check_required_files() -> list[str]:
    """Return missing required files."""
    return [rel for rel in REQUIRED_FILES if not (PROJECT_ROOT / rel).exists()]


def _check_yaml(paths: Iterable[Path]) -> list[str]:
    """Validate YAML config files."""
    errors: list[str] = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            if not isinstance(data, dict):
                errors.append(f"{path}: expected mapping")
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return errors


def _check_readme_links() -> list[str]:
    """Check local markdown links in README."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme)
    missing: list[str] = []
    for link in links:
        if link.startswith(("http://", "https://", "#")):
            continue
        target = link.split("#", 1)[0]
        if target and not (PROJECT_ROOT / target).exists():
            missing.append(link)
    return missing


def _check_imports() -> list[str]:
    """Import core modules and return failures.

    Optional UI/data dependencies can be unavailable in minimal CI or local CPU
    environments. In those cases, the script falls back to syntax compilation for
    the affected file while still importing core model/search/evaluation modules.
    """
    failures: list[str] = []
    optional_modules = {"streamlit_app.app", "data.dataset"}
    for name in IMPORTS:
        try:
            importlib.import_module(name)
        except Exception as exc:
            if name in optional_modules:
                rel = Path(*name.split(".")).with_suffix(".py")
                try:
                    py_compile.compile(str(PROJECT_ROOT / rel), doraise=True)
                    print(f"Optional import warning for {name}: {exc}. Syntax compile passed.")
                    continue
                except Exception as compile_exc:
                    failures.append(f"{name}: import failed ({exc}); syntax compile failed ({compile_exc})")
                    continue
            failures.append(f"{name}: {exc}")
    return failures


def main() -> None:
    """Run repository verification."""
    errors: list[str] = []
    missing = _check_required_files()
    if missing:
        errors.extend([f"Missing required file: {item}" for item in missing])
    errors.extend(_check_yaml((PROJECT_ROOT / "configs").glob("*.yaml")))
    link_errors = _check_readme_links()
    errors.extend([f"README local link missing: {link}" for link in link_errors])
    errors.extend([f"Import failed: {failure}" for failure in _check_imports()])

    if errors:
        print("Project verification failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Project verification passed")


if __name__ == "__main__":
    main()
