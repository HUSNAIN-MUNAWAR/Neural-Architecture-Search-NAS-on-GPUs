# Contributing

Thanks for contributing to Neural Architecture Search on GPUs.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Branch Naming

Use clear branch names:

- `feature/streamlit-monitoring`
- `fix/config-validation`
- `docs/vps-deployment`
- `test/search-trainer-smoke`

## Code Style

- Python 3.9+.
- Prefer type hints and docstrings for public functions/classes.
- Keep modules small and focused.
- Use Ruff and Black settings from `pyproject.toml`.

```bash
ruff check .
black .
pytest
```

## Pull Request Rules

Before opening a PR:

1. Run `pytest`.
2. Run `python scripts/smoke_test.py`.
3. Run `python scripts/verify_project.py`.
4. Update docs if behavior changed.
5. Do not commit datasets, checkpoints, logs, TensorBoard events, credentials, or generated cache files.

## GPU Tests

GPU-heavy tests should be marked with `@pytest.mark.gpu`. CI skips GPU-only tests by default because GitHub-hosted runners usually do not provide CUDA GPUs.
