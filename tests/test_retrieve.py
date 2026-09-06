import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest import build_index
from retrieve import format_hits_for_prompt, get_collection, retrieve


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_small_repo(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "payments.py", "\n".join([
        "def charge_card(amount):",
        "    # Idempotency key prevents double-charging on retry.",
        "    if seen_idempotency_key(amount):",
        "        return existing_charge()",
        "    return create_charge(amount)",
    ] * 3))  # repeated to give the chunker something to actually chunk
    _write(repo / "shipping.py", "\n".join([
        "def schedule_delivery(order_id):",
        "    truck = find_available_truck()",
        "    return dispatch(truck, order_id)",
    ] * 3))
    build_index(repo, tmp_path / "store", "test_retrieve")
    return get_collection(tmp_path / "store", "test_retrieve")


def test_retrieve_ranks_the_semantically_relevant_chunk_first(tmp_path):
    collection = _build_small_repo(tmp_path)
    hits = retrieve(collection, "how does the code avoid charging a customer twice", k=2)
    assert hits[0]["source"] == "payments.py"


def test_retrieve_returns_requested_k(tmp_path):
    collection = _build_small_repo(tmp_path)
    hits = retrieve(collection, "delivery truck dispatch", k=1)
    assert len(hits) == 1


def test_format_hits_for_prompt_labels_each_block_with_source_and_lines():
    hits = [{"source": "a.py", "start_line": 1, "end_line": 5, "text": "code here", "distance": 0.5}]
    rendered = format_hits_for_prompt(hits)
    assert "[a.py:1-5]" in rendered
    assert "code here" in rendered
