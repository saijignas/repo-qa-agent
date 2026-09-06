import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import _distinctive_token, answer
from ingest import build_index

requires_gemini_key = pytest.mark.skipif(
    "GEMINI_API_KEY" not in os.environ,
    reason="GEMINI_API_KEY not set -- generation-dependent test skipped, not faked",
)


def test_distinctive_token_picks_the_longest_non_stopword():
    assert _distinctive_token("what is MAX_RETRIES set to") == "MAX_RETRIES"


def test_distinctive_token_returns_none_for_all_stopword_input():
    assert _distinctive_token("what is the") is None


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@requires_gemini_key
def test_answer_cites_a_source_that_actually_exists_in_the_repo(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "worker.py", "\n".join([
        "MAX_RETRIES = 3",
        "def process_order(order_id, attempt):",
        "    if attempt >= MAX_RETRIES:",
        "        return 'failed_dlq'",
    ] * 4))
    build_index(repo, tmp_path / "store", "test_agent")

    result = answer("How many times does the consumer retry before giving up?", repo, tmp_path / "store", "test_agent")

    assert result["sources"], "expected at least one cited source"
    for source in result["sources"]:
        assert (repo / source).is_file(), f"cited source {source} doesn't exist in the repo"
    assert "MAX_RETRIES" in result["answer"] or "3" in result["answer"]


@requires_gemini_key
def test_answer_declines_when_context_is_irrelevant(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "unrelated.py", "def greet():\n    return 'hello world'\n")
    build_index(repo, tmp_path / "store", "test_agent_irrelevant")

    result = answer("What color is the CEO's car?", repo, tmp_path / "store", "test_agent_irrelevant")
    assert "don't have enough information" in result["answer"].lower() or "cannot" in result["answer"].lower()
