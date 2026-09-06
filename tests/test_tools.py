import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import fetch_file, search_keyword


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_search_keyword_finds_exact_matches_case_insensitively(tmp_path):
    _write(tmp_path / "consumer" / "worker.py", "MAX_RETRIES = 3\ndef foo():\n    pass\n")
    hits = search_keyword(tmp_path, "max_retries")
    assert len(hits) == 1
    assert hits[0]["source"] == "consumer/worker.py"
    assert hits[0]["line_number"] == 1


def test_search_keyword_respects_max_hits(tmp_path):
    _write(tmp_path / "a.py", "TARGET\n" * 20)
    hits = search_keyword(tmp_path, "TARGET", max_hits=3)
    assert len(hits) == 3


def test_fetch_file_returns_full_contents(tmp_path):
    _write(tmp_path / "shared" / "models.py", "line1\nline2\nline3\n")
    content = fetch_file(tmp_path, "shared/models.py")
    assert content == "line1\nline2\nline3"


def test_fetch_file_respects_line_range(tmp_path):
    _write(tmp_path / "big.py", "\n".join(f"line{i}" for i in range(1, 11)))
    content = fetch_file(tmp_path, "big.py", start_line=3, end_line=5)
    assert content == "line3\nline4\nline5"


def test_fetch_file_raises_clearly_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        fetch_file(tmp_path, "does/not/exist.py")


def test_fetch_file_refuses_path_traversal_outside_repo_root(tmp_path):
    (tmp_path / "repo").mkdir()
    _write(tmp_path / "secret.py", "top secret")
    with pytest.raises(ValueError):
        fetch_file(tmp_path / "repo", "../secret.py")
