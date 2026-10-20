"""Summarises eval/engine_results.json (written by run_engine_eval.py) as markdown.

Three views:
1. Automated checks passed per engine and run.
2. Decisions: does the graph take the same actions as the hand-rolled agent
   (tools used, sources used, decline or answer) on every question?
3. Wording noise: how similar are two answers to the same question? Compared
   within one engine (run 1 vs run 2, which is pure sampling noise) and between
   engines (hand-rolled vs graph). If the between-engine similarity is no lower
   than the within-engine similarity, the engines are indistinguishable at this
   resolution.
"""
import argparse
import difflib
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_engine_eval import check  # single source of truth for the automated checks


def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def passed(rows):
    ok = total = 0
    for r in rows:
        if "error" in r:
            total += 1  # a failed call counts against the engine
            continue
        for v in r["checks"].values():
            total += 1
            ok += bool(v)
    return ok, total


def main(path, repo_root):
    data = json.loads(Path(path).read_text())
    engines = data["engines"]
    names = list(engines)
    # Recompute the checks from the stored answers so a fix to the check logic
    # applies to results that were already collected.
    questions = {q["id"]: q for q in json.loads((Path(__file__).parent / "questions.json").read_text())}
    for rows_by_run in engines.values():
        for rows in rows_by_run:
            for r in rows:
                if "error" not in r:
                    r["checks"] = check(questions[r["id"]], {"answer": r["answer"], "sources": r["sources"]}, repo_root)

    print("### Automated checks passed\n")
    print("| engine | run | checks passed | failed calls |")
    print("|---|---|---|---|")
    for n in names:
        for i, rows in enumerate(engines[n], 1):
            ok, total = passed(rows)
            print(f"| {n} | {i} | {ok}/{total} | {sum('error' in r for r in rows)} |")

    a, b = names[0], names[1]
    print(f"\n### Decisions: {a} vs {b} (run-by-run, same question)\n")
    print("| run | questions compared | same tools used | same sources | same decline/answer decision |")
    print("|---|---|---|---|---|")
    for i, (ra, rb) in enumerate(zip(engines[a], engines[b]), 1):
        pairs = [(x, y) for x, y in zip(ra, rb) if "error" not in x and "error" not in y]
        same_tools = sum(x["tools_used"] == y["tools_used"] for x, y in pairs)
        same_src = sum(x["sources"] == y["sources"] for x, y in pairs)
        same_dec = sum(x["checks"].get("declined", not x["checks"].get("not_declined"))
                       == y["checks"].get("declined", not y["checks"].get("not_declined")) for x, y in pairs)
        print(f"| {i} | {len(pairs)} | {same_tools}/{len(pairs)} | {same_src}/{len(pairs)} | {same_dec}/{len(pairs)} |")

    def mean_sim(rows_x, rows_y):
        vals = [similarity(x["answer"], y["answer"]) for x, y in zip(rows_x, rows_y) if "error" not in x and "error" not in y]
        return sum(vals) / len(vals) if vals else float("nan"), len(vals)

    print("\n### Wording similarity (0 to 1; higher means closer text)\n")
    print("| comparison | mean similarity | answer pairs |")
    print("|---|---|---|")
    for n in names:
        if len(engines[n]) >= 2:
            m, c = mean_sim(engines[n][0], engines[n][1])
            print(f"| {n} run 1 vs run 2 (sampling noise) | {m:.2f} | {c} |")
    cross = []
    for ra, rb in product(engines[a], engines[b]):
        m, c = mean_sim(ra, rb)
        cross.append(m)
    print(f"| {a} vs {b} (all run pairs) | {sum(cross)/len(cross):.2f} | {len(cross)} run pairs |")

    secs = {n: [r["seconds"] for rows in engines[n] for r in rows if "error" not in r] for n in names}
    print("\n### Mean seconds per question (dominated by the model call)\n")
    for n in names:
        print(f"- {n}: {sum(secs[n]) / len(secs[n]):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_root", help="the repo that was indexed (to confirm cited files exist)")
    parser.add_argument("--results", default=str(Path(__file__).parent / "engine_results.json"))
    args = parser.parse_args()
    main(args.results, Path(args.repo_root).resolve())
