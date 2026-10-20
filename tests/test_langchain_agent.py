"""Tests for the LangChain-wrapped version. No network and no Gemini: the model is
a recording fake. Chroma itself is real, because the question that matters most
here is whether the LangChain wrapper reads the existing index in the same
embedding space as the hand-written retriever."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("langchain_chroma")
pytest.importorskip("langchain_google_genai")

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import agent as hand_rolled
import langchain_agent
from ingest import build_index
from retrieve import get_collection
from retrieve import retrieve as hand_retrieve
from tools import search_keyword

QUERIES = [
    "How are failed messages retried before giving up?",
    "What database stores the orders?",
    "How many replicas does the consumer run?",
    "What is the idempotency key used for?",
]


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    root = tmp_path_factory.mktemp("lc") / "repo"
    (root / "consumer").mkdir(parents=True)
    (root / "k8s").mkdir()
    (root / "consumer" / "worker.py").write_text(
        "\n".join(
            [
                "MAX_RETRIES = 3",
                "def process_order(order_id, attempt):",
                "    # retry a failed message until MAX_RETRIES, then send it to the dead letter queue",
                "    if attempt >= MAX_RETRIES:",
                "        return 'failed_dlq'",
                "    return 'completed'",
            ]
            * 8
        )
    )
    (root / "README.md").write_text(
        "# Orders\nOrders are stored in PostgreSQL. The producer takes an Idempotency-Key header so a "
        "client retry never creates a duplicate order.\n" * 12
    )
    (root / "k8s" / "consumer.yaml").write_text("replicas: 1\n# one replica because the VM has 1GB of RAM\n" * 10)
    store = root.parent / "store"
    build_index(root, store, "lc_test")
    return root, store


def test_langchain_retrieval_matches_the_hand_written_retriever(indexed):
    root, store = indexed
    collection = get_collection(store, "lc_test")
    vectorstore = langchain_agent.open_vectorstore(store, "lc_test")
    for q in QUERIES:
        mine = hand_retrieve(collection, q, k=5)
        theirs = langchain_agent.retriever(vectorstore, q, k=5)
        assert [(h["source"], h["start_line"], h["end_line"]) for h in mine] == \
               [(h["source"], h["start_line"], h["end_line"]) for h in theirs], q
        assert [h["distance"] for h in theirs] == pytest.approx([h["distance"] for h in mine], abs=1e-4), q
        assert [h["text"] for h in mine] == [h["text"] for h in theirs]


def test_opening_a_missing_collection_fails_instead_of_creating_one(tmp_path):
    import chromadb

    chromadb.PersistentClient(path=str(tmp_path / "store"))  # an empty store
    with pytest.raises(Exception):
        langchain_agent.open_vectorstore(tmp_path / "store", "does_not_exist")


def test_the_keyword_tool_wrapper_returns_exactly_what_the_function_returns(indexed):
    root, _ = indexed
    assert langchain_agent.keyword_search(root, "MAX_RETRIES") == search_keyword(root, "MAX_RETRIES")
    assert langchain_agent.search_keyword_tool.name == "search_keyword_tool"
    assert langchain_agent.search_keyword_tool.description


class Recorder:
    """A fake chat model that records the prompt text it was sent."""

    def __init__(self, reply):
        self.prompts = []
        self.reply = reply

    def __call__(self, prompt_value):
        self.prompts.append(prompt_value.to_string())
        return AIMessage(content=self.reply)


def test_the_chain_sends_the_same_prompt_text_as_the_hand_written_agent():
    rec = Recorder("answer [a.py:1-2]")
    chain = langchain_agent.build_chain(RunnableLambda(rec))
    out = chain.invoke({"context": "[a.py:1-2]\nx = 1", "question": "What is x?"})
    expected = hand_rolled.ANSWER_PROMPT_TEMPLATE.format(context="[a.py:1-2]\nx = 1", question="What is x?")
    assert out == "answer [a.py:1-2]"
    assert len(rec.prompts) == 1
    assert rec.prompts[0].endswith(expected) or expected in rec.prompts[0]


def test_the_parser_extracts_text_from_gemini_style_content_blocks():
    chain = langchain_agent.build_chain(RunnableLambda(lambda pv: AIMessage(content=[{"type": "text", "text": "hello"}])))
    assert chain.invoke({"context": "c", "question": "q"}) == "hello"


@pytest.mark.parametrize("question", [
    "How are failed messages retried before giving up?",   # confident retrieval
    "What value is MAX_RETRIES set to?",                    # exercises the keyword path if retrieval is weak
    "What color is the CEO's car?",                         # out of scope
])
def test_the_langchain_graph_makes_the_same_decisions_as_the_hand_written_agent(indexed, monkeypatch, question):
    root, store = indexed
    rec = Recorder("stub answer [consumer/worker.py:1-2]")
    graph = langchain_agent.build_graph(RunnableLambda(rec))
    lc = langchain_agent.answer(question, root, store, "lc_test", graph=graph)

    monkeypatch.setattr(hand_rolled, "call_gemini", lambda prompt, *a, **k: "stub answer [consumer/worker.py:1-2]")
    mine = hand_rolled.answer(question, root, store, "lc_test")

    assert lc["answer"] == mine["answer"]
    assert lc["sources"] == mine["sources"]
    assert lc["tools_used"] == mine["tools_used"]


def test_make_model_leaves_sampling_at_the_api_default_and_hides_the_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-test-key-123")
    model = langchain_agent.make_model()
    assert model.temperature is None
    assert model.max_output_tokens == 1024
    assert "super-secret-test-key-123" not in repr(model)
    assert "super-secret-test-key-123" not in str(model)
