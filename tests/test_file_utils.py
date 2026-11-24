import pytest

from utils.file_utils import load_json, save_json


def test_json_write_is_recoverable_and_atomic(tmp_path):
    path = tmp_path / "metrics.json"
    save_json({"epoch": 1}, str(path))
    assert load_json(str(path)) == {"epoch": 1}
    assert list(tmp_path.glob("*.tmp")) == []


def test_malformed_json_is_not_silently_replaced(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cannot be read safely"):
        load_json(str(path))
    assert path.read_text(encoding="utf-8") == "{broken"
