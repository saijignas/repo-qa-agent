"""Summarises eval/engine_results.json (written by run_engine_eval.py) as markdown,
for any number of engines (the first one listed in the file is the baseline).

Four views:
1. Automated checks passed per engine and run.
2. Decisions: does each engine take the same actions as the baseline (tools used,
   sources used, decline or answer) on every question?
3. Wording noise: how similar are two answers to the same question? Compared
   within one engine (run 1 vs run 2, which is pure sampling noise) and between
   engines. If the between-engine similarity is no lower than the within-engine
   similarity, the engines are indistinguishable at this resolution.
4. Mean seconds per question (dominated by the model call, not the engine).

The checks are recomputed from the stored answers, so a fix to the check logic
applies to results that were already collected.
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


def decision(row):
    c = row["checks"]
    return c["declined"] if "declined" in c else not c["not_declined"]


def mean_sim(rows_x, rows_y):
    vals = [similarity(x["answer"], y["answer"]) for x, y in zip(rows_x, rows_y) if "error" not in x and "error" not in y]
    return (sum(vals) / len(vals) if vals else float("nan")), len(vals)


def main(path, repo_root):
    data = json.loads(Path(path).read_text())
    engines = data["engines"]
    names = list(engines)
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

    base = names[0]
    for other in names[1:]:
        print(f"\n### Decisions: {other} vs {base} (run by run, same question)\n")
        print("| run | questions compared | same tools used | same sources | same decline/answer decision |")
        print("|---|---|---|---|---|")
        for i, (rb, ro) in enumerate(zip(engines[base], engines[other]), 1):
            pairs = [(x, y) for x, y in zip(rb, ro) if "error" not in x and "error" not in y]
            same_tools = sum(x["tools_used"] == y["tools_used"] for x, y in pairs)
            same_src = sum(x["sources"] == y["sources"] for x, y in pairs)
            same_dec = sum(decision(x) == decision(y) for x, y in pairs)
            print(f"| {i} | {len(pairs)} | {same_tools}/{len(pairs)} | {same_src}/{len(pairs)} | {same_dec}/{len(pairs)} |")

    print("\n### Wording similarity (0 to 1; higher means closer text)\n")
    print("| comparison | mean similarity | answer pairs |")
    print("|---|---|---|")
    for n in names:
        if len(engines[n]) >= 2:
            m, c = mean_sim(engines[n][0], engines[n][1])
            print(f"| {n} run 1 vs run 2 (sampling noise) | {m:.2f} | {c} |")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sims = [mean_sim(ra, rb)[0] for ra, rb in product(engines[a], engines[b])]
            print(f"| {a} vs {b} (all run pairs) | {sum(sims) / len(sims):.2f} | {len(sims)} run pairs |")

    print("\n### Mean seconds per question (dominated by the model call)\n")
    for n in names:
        secs = [r["seconds"] for rows in engines[n] for r in rows if "error" not in r]
        print(f"- {n}: {sum(secs) / len(secs):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_root", help="the repo that was indexed (to confirm cited files exist)")
    parser.add_argument("--results", default=str(Path(__file__).parent / "engine_results.json"))
    args = parser.parse_args()
    main(args.results, Path(args.repo_root).resolve())
