import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest import build_index, chunk_file, find_source_files


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_find_source_files_filters_by_extension_and_skips_pycache(tmp_path):
    _write(tmp_path / "a.py", "x = 1\n")
    _write(tmp_path / "b.md", "# hi\n")
    _write(tmp_path / "c.txt", "not indexed\n")
    _write(tmp_path / "__pycache__" / "a.cpython-311.pyc", "binary junk")

    found = {p.name for p in find_source_files(tmp_path)}
    assert found == {"a.py", "b.md"}


def test_chunk_file_produces_overlapping_windows_covering_the_whole_file(tmp_path):
    path = tmp_path / "big.py"
    lines = [f"line {i}" for i in range(1, 101)]  # 100 lines
    _write(path, "\n".join(lines) + "\n")

    chunks = list(chunk_file(path, tmp_path))
    # Every line should be covered by at least one chunk.
    covered = set()
    for c in chunks:
        covered.update(range(c["start_line"] - 1, c["end_line"]))
    assert covered == set(range(100))
    # The last chunk should end exactly at the last line, not run past it.
    assert chunks[-1]["end_line"] == 100


def test_chunk_file_skips_blank_files(tmp_path):
    path = tmp_path / "empty.py"
    _write(path, "")
    assert list(chunk_file(path, tmp_path)) == []


def test_build_index_indexes_expected_file_count(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "app.py", "\n".join(f"line {i}" for i in range(50)))
    _write(repo / "README.md", "# Project\n\nSome docs.\n")

    store = tmp_path / "store"
    collection = build_index(repo, store, "test_collection")
    assert collection.count() > 0

    sources = {m["source"] for m in collection.get()["metadatas"]}
    assert sources == {"app.py", "README.md"}
