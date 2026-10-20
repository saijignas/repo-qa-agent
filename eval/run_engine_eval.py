"""Runs the full agent (retrieval + generation) on every evaluation question with
each orchestration engine, and records automated checks plus the raw answers.

Engines:
  handrolled  agent.answer       (explicit if/else policy)
  langgraph   graph_agent.answer (the same policy as a LangGraph state graph)

Each engine is run --runs times so the comparison has a noise baseline: the
model is sampled at its default temperature, so two runs of the SAME engine
already differ in wording. A difference between engines only means something if
it is larger than that.

Automated checks (they do not judge whether an answer is *correct*; read the
saved answers for that):
  in-scope questions
    - not_declined         the answer is not the "I don't have enough information" decline
    - has_citation         the answer contains at least one [file:line-range] citation
    - citations_exist      every cited file exists in the target repo
    - hit_expected_source  at least one of the question's expected_sources is among the sources used
  out-of-scope questions
    - declined             the answer declines instead of inventing something

Pacing: a few seconds between model calls because of the free-tier per-minute cap.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent as hand_rolled
import graph_agent

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
# matches file:line and file:start-end anywhere, including several inside one bracket pair,
# e.g. "[README.md:35-43, consumer/worker.py:3-9]"
CITATION_RE = re.compile(r"([A-Za-z0-9_./\-]+\.[A-Za-z0-9]+):\d+(?:-\d+)?")
DECLINE_MARKERS = ("don't have enough information", "do not have enough information", "cannot answer", "can't answer")

ENGINES = {
    "handrolled": hand_rolled.answer,
    "langgraph": graph_agent.answer,
}


def is_decline(text):
    t = text.lower()
    return any(m in t for m in DECLINE_MARKERS)


def check(question, result, repo_root):
    text = result["answer"]
    cited_files = sorted({m.group(1) for m in CITATION_RE.finditer(text)})
    if question["in_scope"]:
        return {
            "not_declined": not is_decline(text),
            "has_citation": bool(cited_files),
            "citations_exist": bool(cited_files) and all((Path(repo_root) / f).is_file() for f in cited_files),
            "hit_expected_source": bool(set(result["sources"]) & set(question["expected_sources"])),
        }
    return {"declined": is_decline(text)}


def run(engine_name, questions, repo_root, store_dir, collection, pace):
    fn = ENGINES[engine_name]
    rows = []
    for i, q in enumerate(questions):
        if i:
            time.sleep(pace)
        start = time.perf_counter()
        try:
            result = fn(q["question"], repo_root, store_dir, collection)
        except Exception as exc:  # a failed call is a result, not a reason to lose the whole run
            elapsed = time.perf_counter() - start
            rows.append({"id": q["id"], "in_scope": q["in_scope"], "error": type(exc).__name__, "seconds": round(elapsed, 2), "checks": {}})
            print(f"  [{engine_name}] {q['id']} FAILED ({type(exc).__name__})", flush=True)
            continue
        elapsed = time.perf_counter() - start
        rows.append({
            "id": q["id"],
            "in_scope": q["in_scope"],
            "answer": result["answer"],
            "sources": result["sources"],
            "tools_used": result["tools_used"],
            "path": result.get("path"),
            "seconds": round(elapsed, 2),
            "checks": check(q, result, repo_root),
        })
        print(f"  [{engine_name}] {q['id']} {elapsed:5.1f}s checks={rows[-1]['checks']}", flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_root")
    parser.add_argument("--store", default="../chroma_store")
    parser.add_argument("--collection", default="repo_qa")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--pace", type=float, default=3.0, help="seconds between model calls")
    parser.add_argument("--out", default=str(Path(__file__).parent / "engine_results.json"))
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text())
    repo_root = Path(args.repo_root).resolve()
    store_dir = Path(args.store).resolve()

    output = {"runs": args.runs, "engines": {}}
    for engine in ENGINES:
        output["engines"][engine] = []
        for r in range(args.runs):
            print(f"{engine}, run {r + 1}/{args.runs}", flush=True)
            output["engines"][engine].append(run(engine, questions, repo_root, store_dir, args.collection, args.pace))
            Path(args.out).write_text(json.dumps(output, indent=2))  # save after every run
            time.sleep(args.pace)
    Path(args.out).write_text(json.dumps(output, indent=2))
    print(f"\nwritten to {args.out}")
