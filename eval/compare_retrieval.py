"""Retrieval-only comparison (no model calls): the hand-written retriever against the
LangChain Chroma wrapper on every in-scope evaluation question. Because the wrapper
reuses the same embedding model and the same index, the two should return identical
chunks in identical order with identical distances; this checks it on the real
questions and the real index, and reports precision@k for both.

Usage: python eval/compare_retrieval.py [--store ./chroma_store] [--collection repo_qa] [-k 5]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import langchain_agent
from retrieve import get_collection, retrieve

QUESTIONS = Path(__file__).parent / "questions.json"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=str(Path(__file__).parent.parent / "chroma_store"))
    ap.add_argument("--collection", default="repo_qa")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()

    qs = [q for q in json.loads(QUESTIONS.read_text()) if q["in_scope"]]
    collection = get_collection(Path(a.store), a.collection)
    vs = langchain_agent.open_vectorstore(a.store, a.collection)

    hits_mine = hits_lc = same_chunks = 0
    max_gap = 0.0
    for q in qs:
        mine = retrieve(collection, q["question"], k=a.k)
        lc = langchain_agent.retriever(vs, q["question"], k=a.k)
        key = lambda hs: [(h["source"], h["start_line"], h["end_line"]) for h in hs]
        same = key(mine) == key(lc)
        same_chunks += same
        gap = max(abs(x["distance"] - y["distance"]) for x, y in zip(mine, lc)) if same else float("nan")
        max_gap = max(max_gap, gap) if same else max_gap
        expected = set(q["expected_sources"])
        hm = bool({h["source"] for h in mine} & expected)
        hl = bool({h["source"] for h in lc} & expected)
        hits_mine += hm
        hits_lc += hl
        print(f"{q['id']}: same chunks and order: {same} | hit (hand-written / LangChain): {hm} / {hl}")
    n = len(qs)
    print(f"\nprecision@{a.k}: hand-written {hits_mine}/{n} = {hits_mine / n:.2f}, LangChain {hits_lc}/{n} = {hits_lc / n:.2f}")
    print(f"identical chunks in identical order on {same_chunks}/{n} questions; largest distance difference {max_gap:.2e}")
