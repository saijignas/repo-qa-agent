# Repo-QA Agent

A retrieval-augmented Q&A agent over a real codebase (indexed here: [Order-Processing-Pipeline](https://github.com/saijignas/Order-Processing-Pipeline)), with retrieval quality actually measured and a disclosed RAG-vs-no-retrieval comparison — not a "built a chatbot" demo.

## Why this exists

Several roles I'm applying to explicitly want "repository indexing, memory, and retrieval systems" and "agent-driven" work — this project is real, tested evidence of that, not a claim.

## Architecture

- **Ingestion** (`ingest.py`): walks the target repo, chunks each file into overlapping ~40-line windows (not syntax-aware — a real production indexer would chunk on function/class boundaries; disclosed as a limitation, not hidden), embeds with Chroma's local ONNX MiniLM model, and stores in a persistent Chroma collection. Zero external API calls or quota cost for indexing.
- **Retrieval** (`retrieve.py`): pure vector search, kept separate from generation so retrieval quality can be measured and tested independently.
- **Tools** (`tools.py`): two, deliberately minimal — `search_keyword` (exact substring match, for identifiers/config values embedding similarity often under-ranks) and `fetch_file` (raw file read, for when a chunk isn't enough context).
- **Agent** (`agent.py`): an explicit, inspectable decision policy rather than a hidden free-form tool-call loop — retrieve, fall back to keyword search if the best match's distance is above a threshold, then generate an answer that must cite `[file:line]` for every claim and must say so explicitly if the context doesn't contain the answer, rather than filling gaps from general knowledge.
- **Generation**: Gemini API (`gemini-3.1-flash-lite`, chosen for free-tier quota headroom — same constraint discovered in [Consumer-Complaint-Triage](https://github.com/saijignas/Consumer-Complaint-Triage)). Key read from `GEMINI_API_KEY`, sent via request header, never logged or hardcoded.

## Evaluation

`eval/questions.json` — 8 hand-written questions about the indexed repo: 6 in-scope (with the source file(s) that should answer them, for measuring retrieval precision) and 2 deliberately out-of-scope (to check the agent declines instead of hallucinating).

`eval/run_eval.py` measures two things:
1. **Retrieval precision@k** — does the correct source file actually appear in the top-k results?
2. **RAG vs. no-retrieval baseline** — the same question answered with and without repo context, so the difference is measured, not assumed.

### Real result (retrieval-only, no LLM calls needed)

```
Retrieval precision@5: 0.83 (5/6 in-scope questions)
```

The one miss is disclosed, not hidden: a question about what the CI pipeline tests against failed to retrieve `.github/workflows/ci.yml` in the top 5, even though that file directly answers it. Root cause: fixed-size chunking gives YAML config files weak semantic overlap with natural-language questions compared to files with more prose (like the README or test docstrings) — a known weakness of pure embedding-similarity search on structured config, and the reason production RAG systems often use hybrid (lexical + semantic) retrieval rather than semantic search alone.

### RAG vs. baseline comparison (real run, 16 LLM calls total)

Same 8 questions, answered twice — once through the full agent (retrieval + generation), once as a plain Gemini call with zero repo context. Full transcript in `eval/eval_results.json`.

On every in-scope question (q1-q6), the baseline responded with a variant of *"I do not have access to your specific codebase, so I cannot tell you..."* followed by generic textbook advice (e.g. for the idempotency question, it suggested four possible patterns — Redis locking, unique constraints, optimistic locking — without knowing which one this repo actually uses). The RAG agent answered all 6 correctly, with `[file:line]` citations, e.g.:

> "MAX_RETRIES is set to 3. It is defined in `shared/rabbitmq.py` [shared/rabbitmq.py:28]."

Notably, **q6 (the retrieval miss above) still got answered correctly** — the low-confidence fallback triggered a keyword search for "pipeline," which surfaced `tests/test_producer.py` (whose docstring also describes the real-service test setup) even though the primary expected source (`ci.yml`) never made the top-5. The tool fallback compensated for the retrieval gap in this case; it won't always.

On the two deliberately out-of-scope questions (q7: "what color is the CEO's car", q8: a personal fact about the author not in the repo), the agent correctly declined both times — `search_keyword` legitimately found 0 hits for both, and it said so rather than guessing.

Reproduce with:
```bash
python eval/run_eval.py ../Order-Processing-Pipeline --store ./chroma_store
```

## Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

17/17 pass with `GEMINI_API_KEY` set; 15 of those need no external API at all, and the other 2 (citation-correctness, decline-on-irrelevant-context) skip cleanly rather than fake-passing when the key isn't set.

## Setup

```bash
pip install -r requirements.txt
python ingest.py /path/to/target/repo          # builds ./chroma_store, no API key needed
export GEMINI_API_KEY=...                       # only needed for generation
python agent.py "your question here" /path/to/target/repo
```

## Engineering notes

The first real eval run hit a 429 (rate limited) partway through — the free Gemini tier caps requests-per-minute, not just per-day. Fixed with exponential backoff honoring the API's own `Retry-After` header, plus proactive pacing between eval questions (`llm_client.py`, `eval/run_eval.py`) — the same "don't just retry blindly, respect what the failure actually tells you" instinct as the retry/DLQ design in Order-Processing-Pipeline itself.

## Known limitations (disclosed, not fixed yet)

- Chunking is fixed-size, not syntax-aware — a function split across a chunk boundary loses context.
- Retrieval is pure semantic search; no lexical/BM25 hybrid layer yet, which is exactly what caused the one measured retrieval miss above.
- The low-confidence fallback threshold (`LOW_CONFIDENCE_THRESHOLD` in `agent.py`) was set from a small amount of empirical testing on one repo, not tuned across multiple repos or corpus sizes.
