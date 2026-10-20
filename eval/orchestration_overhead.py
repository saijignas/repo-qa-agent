"""Measures the orchestration cost of the LangGraph engine against the hand-rolled
one with every collaborator stubbed (no Chroma, no model), so the numbers are
the framework's own overhead and nothing else. Real answers take seconds because
of the model call; this shows how small the orchestration part is next to that.

Machine-specific: run it yourself rather than trusting a number from a README.
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent as hand_rolled
import graph_agent

N = 300
VECTOR_HITS = [{"source": "consumer/worker.py", "start_line": 1, "end_line": 40, "text": "x", "distance": 0.5}]


def retriever(collection, query, k=5):
    return [dict(h) for h in VECTOR_HITS]


def llm(prompt, *a, **k):
    return "stub"


def keyword(repo_root, term):
    return []


def timed(fn):
    samples = []
    for _ in range(N):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1000)
    return statistics.median(samples), statistics.quantiles(samples, n=20)[18]


if __name__ == "__main__":
    hand_rolled.get_collection = lambda *a, **k: None
    hand_rolled.retrieve = retriever
    hand_rolled.search_keyword = keyword
    hand_rolled.call_gemini = llm
    graph = graph_agent.build_graph(retriever=retriever, keyword_search=keyword, llm=llm, collection_getter=lambda *a, **k: None)

    q = "How does the consumer avoid double-processing an order?"
    # warm-up
    hand_rolled.answer(q, "r", "s")
    graph_agent.answer(q, "r", "s", graph=graph)

    h_med, h_p95 = timed(lambda: hand_rolled.answer(q, "r", "s"))
    g_med, g_p95 = timed(lambda: graph_agent.answer(q, "r", "s", graph=graph))
    print(f"{N} calls each, all collaborators stubbed")
    print(f"hand-rolled: median {h_med:.3f} ms, p95 {h_p95:.3f} ms")
    print(f"langgraph:   median {g_med:.3f} ms, p95 {g_p95:.3f} ms")
    print(f"overhead:    about {g_med - h_med:.2f} ms per question (median)")
