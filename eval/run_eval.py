"""Evaluates the repo-QA agent two ways:

1. Retrieval precision@k: for each in-scope question, does at least one
   of its expected_sources actually appear in the top-k vector-search
   results? Measured on retrieval alone, no LLM call needed.

2. RAG vs. no-retrieval baseline: for every question (including the
   deliberately out-of-scope ones), calls the full agent (retrieval +
   generation) and a plain Gemini call with NO repo context, on the
   identical question. This is the same "compare two approaches on the
   same data, disclose the difference" pattern as every other project
   in this portfolio -- not a claim that RAG "wins," a measurement of
   where and how much it helps.

Uses very few LLM calls on purpose (2 per question: one RAG, one
baseline) since the free Gemini tier's daily quota is small.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import answer
from llm_client import call_gemini
from retrieve import get_collection, retrieve

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
NO_CONTEXT_PROMPT_TEMPLATE = """Answer this question about a codebase you have not been shown. If you don't actually know the specific codebase, say so rather than guessing generically.

Question: {question}

Answer:"""


def eval_retrieval_precision(questions, store_dir, collection_name, k=5):
    collection = get_collection(store_dir, collection_name)
    results = []
    for q in questions:
        if not q["in_scope"]:
            continue
        hits = retrieve(collection, q["question"], k=k)
        retrieved_sources = {h["source"] for h in hits}
        hit = bool(retrieved_sources & set(q["expected_sources"]))
        results.append({"id": q["id"], "hit": hit, "retrieved_sources": sorted(retrieved_sources)})
    precision_at_k = sum(r["hit"] for r in results) / len(results) if results else None
    return precision_at_k, results


def eval_rag_vs_baseline(questions, repo_root, store_dir, collection_name):
    comparisons = []
    for i, q in enumerate(questions):
        if i > 0:
            time.sleep(3)  # proactive pacing -- free-tier RPM cap, not just a daily quota
        rag_result = answer(q["question"], repo_root, store_dir, collection_name)
        time.sleep(3)
        baseline_answer = call_gemini(NO_CONTEXT_PROMPT_TEMPLATE.format(question=q["question"]))
        comparisons.append({
            "id": q["id"],
            "question": q["question"],
            "in_scope": q["in_scope"],
            "rag_answer": rag_result["answer"],
            "rag_sources": rag_result["sources"],
            "rag_tools_used": rag_result["tools_used"],
            "baseline_answer": baseline_answer.strip(),
        })
    return comparisons


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root")
    parser.add_argument("--store", default="../chroma_store")
    parser.add_argument("--collection", default="repo_qa")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--skip-generation", action="store_true", help="Only run retrieval precision@k, skip LLM calls entirely")
    parser.add_argument("--out", default="eval_results.json")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text())
    store_dir = Path(args.store).resolve()
    repo_root = Path(args.repo_root).resolve()

    precision, retrieval_results = eval_retrieval_precision(questions, store_dir, args.collection, k=args.k)
    print(f"Retrieval precision@{args.k}: {precision:.2f} ({sum(r['hit'] for r in retrieval_results)}/{len(retrieval_results)} in-scope questions)")
    for r in retrieval_results:
        status = "HIT " if r["hit"] else "MISS"
        print(f"  [{status}] {r['id']}: retrieved {r['retrieved_sources']}")

    output = {"retrieval_precision_at_k": precision, "retrieval_results": retrieval_results}

    if not args.skip_generation:
        comparisons = eval_rag_vs_baseline(questions, repo_root, store_dir, args.collection)
        output["rag_vs_baseline"] = comparisons
        print(f"\nRan {len(comparisons)} RAG-vs-baseline comparisons ({len(comparisons) * 2} total LLM calls).")

    Path(args.out).write_text(json.dumps(output, indent=2))
    print(f"\nFull results written to {args.out}")
