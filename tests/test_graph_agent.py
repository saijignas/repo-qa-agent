"""Tests for the LangGraph version of the agent. None of these call Chroma or
Gemini: every collaborator is stubbed, so each route through the graph is
driven deterministically and the graph can be compared with the hand-rolled
agent on identical inputs."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("langgraph")

import agent as hand_rolled
import graph_agent

THRESHOLD = hand_rolled.LOW_CONFIDENCE_THRESHOLD


def hit(source, dist, start=1, end=40, text="chunk text"):
    return {"source": source, "start_line": start, "end_line": end, "text": text, "distance": dist}


def kw(source, line, text="MAX_RETRIES = 3"):
    return {"source": source, "line_number": line, "line_text": text}


class Stubs:
    """Records calls so tests can assert what each engine did."""

    def __init__(self, vector_hits, keyword_hits=None, llm_text="stub answer [a.py:1-2]"):
        self.vector_hits = vector_hits
        self.keyword_hits = keyword_hits or []
        self.llm_text = llm_text
        self.keyword_terms = []
        self.prompts = []

    def retriever(self, collection, query, k=5):
        return [dict(h) for h in self.vector_hits]

    def keyword(self, repo_root, term):
        self.keyword_terms.append(term)
        return [dict(h) for h in self.keyword_hits]

    def llm(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        return self.llm_text


def run_graph(stubs, question):
    g = graph_agent.build_graph(
        retriever=stubs.retriever,
        keyword_search=stubs.keyword,
        llm=stubs.llm,
        collection_getter=lambda *a, **k: None,
    )
    return graph_agent.answer(question, "repo", "store", graph=g)


def run_hand_rolled(stubs, question, monkeypatch):
    monkeypatch.setattr(hand_rolled, "get_collection", lambda *a, **k: None)
    monkeypatch.setattr(hand_rolled, "retrieve", stubs.retriever)
    monkeypatch.setattr(hand_rolled, "search_keyword", stubs.keyword)
    monkeypatch.setattr(hand_rolled, "call_gemini", stubs.llm)
    return hand_rolled.answer(question, "repo", "store")


def test_confident_retrieval_skips_the_keyword_tool():
    stubs = Stubs([hit("consumer/worker.py", 0.4), hit("README.md", 0.9)])
    result = run_graph(stubs, "How does the consumer avoid double processing?")
    assert result["path"] == ["retrieve", "check_context", "generate"]
    assert result["tools_used"] == []
    assert stubs.keyword_terms == []
    assert result["sources"] == ["README.md", "consumer/worker.py"]


def test_weak_retrieval_runs_keyword_fallback_and_folds_hits_into_context():
    stubs = Stubs([hit("README.md", THRESHOLD + 0.2)], keyword_hits=[kw("shared/rabbitmq.py", 28)])
    result = run_graph(stubs, "What value is MAX_RETRIES set to?")
    assert result["path"] == ["retrieve", "keyword_search", "check_context", "generate"]
    assert result["tools_used"] == [{"tool": "search_keyword", "term": "MAX_RETRIES", "hit_count": 1}]
    assert stubs.keyword_terms == ["MAX_RETRIES"]
    assert "shared/rabbitmq.py" in result["sources"]
    assert "[shared/rabbitmq.py:28-28]" in stubs.prompts[0]


def test_distance_exactly_at_threshold_is_still_confident():
    stubs = Stubs([hit("a.py", THRESHOLD)])
    result = run_graph(stubs, "How does it work?")
    assert "keyword_search" not in result["path"]


def test_no_hits_and_no_keyword_term_declines_without_calling_the_model():
    stubs = Stubs([])
    result = run_graph(stubs, "what is the")  # all stopwords, so no distinctive token
    assert result["path"] == ["retrieve", "keyword_search", "check_context", "decline"]
    assert result["answer"] == graph_agent.DECLINE_TEXT
    assert result["sources"] == []
    assert stubs.prompts == []


def test_no_hits_and_keyword_finds_nothing_declines():
    stubs = Stubs([], keyword_hits=[])
    result = run_graph(stubs, "What color is the CEO's car?")
    assert result["path"][-1] == "decline"
    assert result["tools_used"] == [{"tool": "search_keyword", "term": "color", "hit_count": 0}]
    assert stubs.prompts == []


def test_weak_retrieval_with_keyword_miss_still_answers_from_the_weak_hits():
    stubs = Stubs([hit("README.md", THRESHOLD + 0.5)], keyword_hits=[])
    result = run_graph(stubs, "What color is the CEO's car?")
    assert result["path"][-1] == "generate"
    assert result["tools_used"][0]["hit_count"] == 0


SCENARIOS = {
    "confident": (Stubs([hit("a.py", 0.3), hit("b.py", 0.8)]), "How does the consumer retry?"),
    "weak_with_keyword_hits": (
        Stubs([hit("README.md", 1.6)], keyword_hits=[kw("shared/rabbitmq.py", 28), kw("consumer/worker.py", 7)]),
        "What value is MAX_RETRIES set to?",
    ),
    "weak_keyword_miss": (Stubs([hit("README.md", 1.4)], keyword_hits=[]), "What color is the CEO's car?"),
    "empty_all_stopwords": (Stubs([]), "what is the"),
    "empty_keyword_hit": (Stubs([], keyword_hits=[kw("x.py", 3)]), "Where is idempotency_key defined?"),
    "at_threshold": (Stubs([hit("a.py", THRESHOLD)]), "How many replicas run?"),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_graph_and_hand_rolled_agent_make_identical_decisions(name, monkeypatch):
    template, question = SCENARIOS[name]
    # fresh stub instances per engine so recorded calls do not leak between them
    def fresh():
        return Stubs(template.vector_hits, template.keyword_hits, template.llm_text)

    graph_stubs, hand_stubs = fresh(), fresh()
    graph_result = run_graph(graph_stubs, question)
    hand_result = run_hand_rolled(hand_stubs, question, monkeypatch)

    assert graph_result["answer"] == hand_result["answer"]
    assert graph_result["sources"] == hand_result["sources"]
    assert graph_result["tools_used"] == hand_result["tools_used"]
    assert graph_stubs.keyword_terms == hand_stubs.keyword_terms
    assert graph_stubs.prompts == hand_stubs.prompts  # same prompt text reaches the model


def test_mermaid_diagram_names_every_node():
    diagram = graph_agent.mermaid()
    for node in ["retrieve", "keyword_search", "check_context", "generate", "decline"]:
        assert node in diagram
