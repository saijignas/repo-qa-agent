"""The automated checks in eval/run_engine_eval.py decide whether an engine's
answers pass, so they get their own tests (they once missed citations written
as several files inside one bracket pair)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_engine_eval import CITATION_RE, check, is_decline

IN_SCOPE = {"in_scope": True, "expected_sources": ["consumer/worker.py"]}
OUT_OF_SCOPE = {"in_scope": False, "expected_sources": []}


def files(text):
    return sorted({m.group(1) for m in CITATION_RE.finditer(text)})


def test_citation_regex_handles_single_and_multi_citation_brackets():
    assert files("see [consumer/worker.py:38-51]") == ["consumer/worker.py"]
    assert files("[README.md:35-43, consumer/worker.py:3-9]") == ["README.md", "consumer/worker.py"]
    assert files("no citations here, just 3-9 and a.b") == []


def test_in_scope_answer_with_real_citation_passes_every_check(tmp_path):
    (tmp_path / "consumer").mkdir()
    (tmp_path / "consumer" / "worker.py").write_text("x = 1\n")
    result = {"answer": "It locks the row [consumer/worker.py:38-51].", "sources": ["consumer/worker.py"]}
    assert check(IN_SCOPE, result, tmp_path) == {
        "not_declined": True, "has_citation": True, "citations_exist": True, "hit_expected_source": True,
    }


def test_citation_of_a_file_that_does_not_exist_fails(tmp_path):
    result = {"answer": "See [nope/ghost.py:1-2].", "sources": ["consumer/worker.py"]}
    assert check(IN_SCOPE, result, tmp_path)["citations_exist"] is False


def test_decline_on_an_in_scope_question_fails_not_declined(tmp_path):
    result = {"answer": "I don't have enough information in the indexed context to answer that", "sources": []}
    checks = check(IN_SCOPE, result, tmp_path)
    assert checks["not_declined"] is False and checks["has_citation"] is False


def test_out_of_scope_question_passes_only_when_declined(tmp_path):
    declined = {"answer": "I don't have enough information in the indexed context to answer that.", "sources": []}
    invented = {"answer": "The CEO drives a red Tesla.", "sources": ["README.md"]}
    assert check(OUT_OF_SCOPE, declined, tmp_path) == {"declined": True}
    assert check(OUT_OF_SCOPE, invented, tmp_path) == {"declined": False}


def test_is_decline_is_case_insensitive():
    assert is_decline("I DON'T HAVE ENOUGH INFORMATION to say")
    assert not is_decline("The answer is 3 [shared/rabbitmq.py:28]")
