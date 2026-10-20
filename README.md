# Repo-QA Agent

![CI](https://github.com/saijignas/repo-qa-agent/actions/workflows/ci.yml/badge.svg)

A retrieval-augmented Q&A agent over a real codebase (indexed here: [Order-Processing-Pipeline](https://github.com/saijignas/Order-Processing-Pipeline)), with retrieval quality actually measured and a disclosed RAG-vs-no-retrieval comparison — not a "built a chatbot" demo.

**Why this exists:** several roles I'm applying to explicitly want "repository indexing, memory, and retrieval systems" and "agent-driven" work — this project is real, tested evidence of that, not a claim.

## Results at a glance

- **Retrieval precision@5: 0.83** (5/6 in-scope questions) — the one miss is root-caused below, not hidden
- **6/6 in-scope questions answered correctly with real `[file:line]` citations**, including one saved by the tool fallback after retrieval alone missed
- **2/2 out-of-scope questions correctly declined**, not hallucinated
- **17/17 tests pass** against real infrastructure (Chroma + live Gemini calls), 2 of which skip cleanly (not fake-pass) without an API key
- **LangGraph version of the same policy** (`graph_agent.py`): on all 8 questions, in 2 live runs each, it made the same decisions as the hand-rolled agent (same tools, same sources, same answer-or-decline), at about 3 ms of orchestration overhead; see [LangGraph version](#langgraph-version-graph_agentpy)
- **MCP server** (`mcp_server.py`): the repo tools and the agent exposed to Claude Desktop, Claude Code or any MCP client over stdio; file access is restricted to indexed files, every input is bounded, and it is tested both in-process and as a real stdio subprocess; see [MCP server](#mcp-server-mcp_serverpy)

## Quickstart

```
$ python agent.py "How does the consumer avoid double-processing an order if the same message gets redelivered?" ../Order-Processing-Pipeline --store ./chroma_store

ANSWER:
The consumer avoids double-processing redelivered messages through the following mechanisms:

*   **Idempotency Check:** Before performing any work, the consumer checks the order's status in Postgres. If the order is already marked as `completed`, the handler returns immediately with no side effects [README.md:35-43, consumer/worker.py:3-9].
*   **Atomic Database Transactions:** The `process_order` function uses a `SELECT ... FOR UPDATE` query at the start of the process. This creates an atomic critical section per `order_id` by locking the database row. If a redelivered message attempts to process the same order concurrently, it will block on the row lock until the first transaction completes, at which point it will re-read the state and see the `completed` status [consumer/worker.py:38-51].

SOURCES: ['README.md', 'consumer/worker.py', 'producer/app.py', 'shared/rabbitmq.py']
TOOLS USED: []
```
(Real, unedited output from a live run — reproduce it yourself with the Setup steps below.)

## Architecture

```mermaid
flowchart TD
    Q["Question"] --> R["Vector search<br/>(retrieve.py, Chroma)"]
    R -- "confident top match" --> CTX["Build cited context"]
    R -- "low-confidence match<br/>(distance > threshold)" --> KW["Keyword-search fallback<br/>(tools.py: exact substring match)"]
    KW --> CTX
    CTX --> GEN["Gemini generate<br/>(must cite [file:line],<br/>must decline if context is insufficient)"]
    GEN --> ANS["Answer + sources + tools_used"]
```

- **Ingestion** (`ingest.py`): walks the target repo, chunks each file into overlapping ~40-line windows (not syntax-aware — a real production indexer would chunk on function/class boundaries; disclosed as a limitation, not hidden), embeds with Chroma's local ONNX MiniLM model. Zero external API calls or quota cost for indexing.
- **Retrieval** (`retrieve.py`): pure vector search, kept separate from generation so retrieval quality can be measured and tested independently.
- **Tools** (`tools.py`): two, deliberately minimal — `search_keyword` (exact substring match, for identifiers/config values embedding similarity often under-ranks) and `fetch_file` (raw file read, for when a chunk isn't enough context).
- **Agent** (`agent.py`): an explicit, inspectable decision policy rather than a hidden free-form tool-call loop — retrieve, fall back to keyword search if the best match's distance is above a threshold, then generate an answer that must cite `[file:line]` for every claim and must decline explicitly if the context doesn't contain the answer, rather than filling gaps from general knowledge.
- **Generation**: Gemini API (`gemini-3.1-flash-lite`, chosen for free-tier quota headroom — same constraint discovered in [Consumer-Complaint-Triage](https://github.com/saijignas/Consumer-Complaint-Triage)). Key read from `GEMINI_API_KEY`, sent via request header, never logged or hardcoded.

## Evaluation

`eval/questions.json` — 8 hand-written questions: 6 in-scope (each tagged with the source file(s) that should answer it, for measuring retrieval precision) and 2 deliberately out-of-scope (to check the agent declines instead of hallucinating). `eval/run_eval.py` runs both retrieval-only precision@k and the full RAG-vs-no-context-baseline comparison; full transcript in `eval/eval_results.json`.

| ID | Question | Retrieval found the right file (top-5)? | Fallback tool fired? | Final answer | Verdict |
|----|----------|:---:|:---:|---|:---:|
| q1 | How does the consumer avoid double-processing a redelivered order? | ✅ | — | Cites the `SELECT ... FOR UPDATE` fix in `consumer/worker.py` | ✅ Correct |
| q2 | What happens after an order exhausts its retry budget? | ✅ | — | DLQ + `status=failed`, cited | ✅ Correct |
| q3 | What is `MAX_RETRIES` set to, and where? | ✅ (low confidence) | ✅ `search_keyword("MAX_RETRIES")` | "3, defined in `shared/rabbitmq.py:28`" | ✅ Correct |
| q4 | How does the producer prevent duplicate orders on retry? | ✅ | — | `Idempotency-Key` header, cited | ✅ Correct |
| q5 | How many consumer replicas run, and why? | ✅ | — | "1 replica — 1GB RAM limit", cited | ✅ Correct |
| q6 | What does CI actually test against — mocks or real services? | ❌ **Missed** (`ci.yml` not in top-5) | ✅ `search_keyword("pipeline")` → found `test_producer.py` | "Real services", correct — **recovered via fallback** | ✅ Correct |
| q7 | What color is the CEO's car? *(out-of-scope)* | n/a (irrelevant hits) | ✅ `search_keyword("color")` → 0 hits | Declined | ✅ Correct decline |
| q8 | What was the author's college GPA? *(out-of-scope)* | n/a (irrelevant hits) | ✅ `search_keyword("college")` → 0 hits | Declined | ✅ Correct decline |

### Spotlight: how q6 actually got answered correctly despite the retrieval miss

1. **Retrieve:** vector search for *"What does the CI pipeline actually test against — mocks or real services?"* returns `README.md`, `k8s/03-producer.yaml`, `tests/test_consumer.py`, `tests/test_producer.py` — **`.github/workflows/ci.yml` is not in the top-5**, even though it's the file that most directly answers the question. Root cause: fixed-size chunking gives YAML config weak semantic overlap with natural-language questions, compared to files with more prose — a known weakness of pure embedding search, and why production RAG systems often add a lexical (BM25-style) layer.
2. **Fallback trigger:** the top retrieved hit's distance is still above `LOW_CONFIDENCE_THRESHOLD`, so the agent automatically runs `search_keyword("pipeline")` as a second opinion — 8 exact-substring hits, including lines inside `tests/test_producer.py`.
3. **Final answer:** with the keyword hits folded into context, Gemini answers *"The CI pipeline tests against real services, specifically a real Postgres database and a real RabbitMQ instance, rather than mocks [tests/test_producer.py:1-2]."* — **correct**, but citing a different, indirect source than the one the question was designed around.
4. **What this proves, honestly:** the tool fallback compensated for a real retrieval gap this time. It won't always — if `test_producer.py`'s docstring hadn't also described the real-service setup, this question would have failed outright. That's a genuine limitation, not a solved problem.

Reproduce with:
```bash
python eval/run_eval.py ../Order-Processing-Pipeline --store ./chroma_store
```

## LangGraph version (`graph_agent.py`)

The same decision policy as `agent.py`, re-expressed as a [LangGraph](https://github.com/langchain-ai/langgraph) state graph, so the two orchestration styles can be compared on identical questions instead of argued about.

```mermaid
graph TD;
    __start__([start]) --> retrieve(retrieve)
    retrieve -.->|weak or empty| keyword_search(keyword_search)
    retrieve -.->|confident| check_context(check_context)
    keyword_search --> check_context
    check_context -.->|hits| generate(generate)
    check_context -.->|no hits| decline(decline)
    generate --> __end__([end])
    decline --> __end__
```

(Hand-simplified from the Mermaid that `python graph_agent.py --mermaid` prints from the compiled graph, which is the version that cannot drift from the code.)

**What is framework code and what is not.** LangGraph owns only the orchestration: the state, the five nodes, the conditional edges and the invocation. Retrieval, the keyword tool, the prompt and the Gemini call are the exact functions `agent.py` uses, and no LangChain LLM or vector-store wrappers are involved. That is deliberate: it means any difference in the evaluation comes from orchestration and not from a different retriever or prompt. Every collaborator is injectable, so each route can be tested without Chroma or a live model.

```bash
python graph_agent.py "How does the consumer avoid double-processing an order?" ../Order-Processing-Pipeline --store ./chroma_store
```

### Hand-rolled vs LangGraph: measured

`eval/run_engine_eval.py` runs every evaluation question through both engines, twice each (32 live Gemini calls), and `eval/compare_engines.py` summarises the result (raw answers are in `eval/engine_results.json`).

| | hand-rolled | LangGraph |
|---|---|---|
| Automated checks passed (per run, 26 checks) | 25/26 and 25/26 | 25/26 and 25/26 |
| Failed model calls | 0 | 0 |
| Same tools used as the other engine | 8/8 and 8/8 | 8/8 and 8/8 |
| Same sources used as the other engine | 8/8 and 8/8 | 8/8 and 8/8 |
| Same answer-or-decline decision | 8/8 and 8/8 | 8/8 and 8/8 |
| Answer wording similarity, run 1 vs run 2 (same engine) | 0.76 | 0.75 |
| Answer wording similarity, hand-rolled vs LangGraph | 0.77 | |
| Orchestration overhead (model and retrieval stubbed, median of 300 calls) | 0.008 ms | 2.9 ms |

How to read it:
- **No measurable difference in behaviour.** The one check that fails in every run, for both engines, is the same retrieval miss disclosed above (q6: `ci.yml` is not retrieved, so the expected source is never used; the answer is correct only because the keyword fallback finds a different file). Between-engine wording similarity (0.77) is no lower than the similarity between two runs of the same engine (0.76 and 0.75), so the engines cannot be told apart at this resolution.
- **The cost is small but real.** About 3 ms per question against several seconds to tens of seconds for a Gemini call (`eval/orchestration_overhead.py`; machine-specific, run it yourself). The mean seconds per question in the raw results (about 26 against 20) reflect model and service latency varying between runs, not an engine property.
- **What the framework does buy here:** an explicit, inspectable graph, a diagram generated from the code, a per-question `path` of visited nodes, and a structure that extends naturally (checkpointing, human-in-the-loop interrupts, more tools). **What it costs:** the policy is 36 lines of plain Python in `agent.py` and about 190 lines (83 inside `build_graph`) here. For a five-step policy the hand-rolled version is the simpler choice; the graph earns its keep when the flow grows branches, state or approvals.
- **Limits.** 8 questions and 2 runs per engine is a small sample. The automated checks verify behaviour (declined or not, citations present and pointing at real files, expected source used); they do not judge whether an answer is *correct*, so read the saved answers for that.

## MCP server (`mcp_server.py`)

An [MCP](https://modelcontextprotocol.io) (Model Context Protocol) server, so an AI client such as Claude Desktop or Claude Code can use this project's tools directly. Built on the official Python SDK (`mcp` 2.x, `MCPServer`).

| Tool | What it does |
|---|---|
| `list_files` | the files the other tools can see (indexed file types only) |
| `search_keyword` | exact, case-insensitive substring search, returns `{source, line_number, line_text}` |
| `fetch_file` | read an indexed file, or lines `start_line` to `end_line` |
| `ask_repo` | the full agent: retrieval, keyword fallback, a cited answer or a decline; returns `{answer, sources, tools_used}` |

**Run and register it.** Configuration is read from environment variables, not tool arguments, so a client cannot point the server at another directory:

```bash
python ingest.py /path/to/repo --store ./chroma_store      # build the index once
```

```json
{
  "mcpServers": {
    "repo-qa": {
      "command": "/path/to/repo-qa-agent/.venv/bin/python",
      "args": ["/path/to/repo-qa-agent/mcp_server.py"],
      "env": {
        "REPO_QA_REPO_ROOT": "/path/to/repo",
        "REPO_QA_STORE": "/path/to/repo-qa-agent/chroma_store"
      }
    }
  }
}
```

That block is the standard `mcpServers` entry for Claude Desktop's config file; Claude Code can register the same command with `claude mcp add` (check `claude mcp add --help` for the exact flags). `GEMINI_API_KEY` is only needed by `ask_repo` and is read from the server's environment: leave it out of the config file and set it in the environment instead. Without it, `ask_repo` returns a clear error and the other three tools keep working.

**Safety decisions.** An MCP tool is called by a model, so its inputs are treated as untrusted:
- **The file tools see only what the indexer indexes** (`.py`, `.md`, `.yml`, `.yaml`, minus `.git`, virtualenvs, `node_modules` and similar). A `.env`, a key file, a path containing `..`, an absolute path and a file inside `.venv` are all refused, and the tests assert that the secret's contents never appear in the error text.
- **Everything is bounded:** search term length, hit count, question length, `k`, and file output size (truncation is announced in the text).
- **Anticipated failures are plain tool errors** (bad input, refused path, no index, no API key, a failed model request); anything unexpected is reported as a generic error with no paths or stack trace sent to the client.
- **stdout is the protocol on the stdio transport.** The Gemini client prints retry notices, so `ask_repo` redirects the agent's output to stderr. (mcp 2.x also diverts stray stdout by itself, so this is belt and braces: a mutation test that removed the redirect still passed against 2.x, which is why the code comment says so rather than claiming the redirect alone is what protects the stream.)

**Verified.** 33 tests (`tests/test_mcp_server.py`): every tool and refusal through an in-process MCP client, plus a real server subprocess over stdio (listing tools, serving calls, refusing `.env`, and an `ask_repo` call whose stub agent prints to stdout). `eval/mcp_smoke.py` runs the real server with the real agent and live Gemini; its output on the Order-Processing-Pipeline index:

```
tools: ['list_files', 'search_keyword', 'fetch_file', 'ask_repo']
search_keyword MAX_RETRIES -> [('consumer/worker.py', 16), ('consumer/worker.py', 25), ('consumer/worker.py', 59)]
fetch_file ../.env -> is_error: True | Error executing tool fetch_file: not an indexed file: '../.env'. ...
answer: MAX_RETRIES is set to 3 [shared/rabbitmq.py:28]. It is defined in the file `shared/rabbitmq.py` ...
```

What is not tested or claimed: I did not drive it from the Claude Desktop or Claude Code UI, only from the SDK's own client over stdio, and there is no authentication because stdio servers run as the user who launched them (an HTTP deployment would need it).

## Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

82 tests: all pass with `GEMINI_API_KEY` set, and 80 pass plus 2 skip cleanly (rather than fake-passing) when the key is absent. The 2 skipped call live Gemini: citation correctness and decline-on-irrelevant-context. The LangGraph tests (`tests/test_graph_agent.py`) use stubs, drive every route through the graph, and check that it makes the same decisions as the hand-rolled agent, including the case where the best distance equals the threshold exactly. The MCP tests are described above.

## Setup

```bash
pip install -r requirements.txt
python ingest.py /path/to/target/repo          # builds ./chroma_store, no API key needed
export GEMINI_API_KEY=...                       # only needed for generation
python agent.py "your question here" /path/to/target/repo
```

## Engineering notes

The first real eval run hit a 429 (rate limited) partway through — the free Gemini tier caps requests-per-minute, not just per-day. Fixed with exponential backoff honoring the API's own `Retry-After` header, plus proactive pacing between eval questions (`llm_client.py`, `eval/run_eval.py`) — the same "don't just retry blindly, respect what the failure actually tells you" instinct as the retry/DLQ design in Order-Processing-Pipeline itself.

Two bugs found while building the LangGraph version, both fixed with tests:

- **The indexer and the keyword tool had no ignore list for virtualenvs.** Pointed at a working copy that contained a `.venv`, indexing ran for 14 minutes without finishing (it was embedding every installed package), and the keyword tool returned hits from `pip` and `pydantic` files. The first full evaluation run was contaminated by exactly that (the out-of-scope "colour" question showed 8 keyword hits instead of 0), so it was discarded and re-run after the fix. Both now share one file list (`ingest.find_source_files`, which skips `.venv`, `node_modules` and similar), so the keyword tool can never read a file that was not indexed.
- **The Gemini client only retried HTTP 429.** A live test failed on a 503 and an evaluation run died on a 30-second read timeout. It now also retries 500, 502, 503, 504 and read timeouts with exponential backoff, and still does not retry client errors such as a 400.

## Known limitations (disclosed, not fixed yet)

- Chunking is fixed-size, not syntax-aware — a function split across a chunk boundary loses context.
- Retrieval is pure semantic search; no lexical/BM25 hybrid layer yet, which is exactly what caused the q6 retrieval miss above.
- The low-confidence fallback threshold (`LOW_CONFIDENCE_THRESHOLD` in `agent.py`) was set from a small amount of empirical testing on one repo, not tuned across multiple repos or corpus sizes.

## Possible extensions

- **More framework adapters:** the LangGraph version above shows the policy survives being moved into a framework unchanged. A LlamaIndex or LangChain-retriever version would be the next comparison; it would change retrieval and prompting as well as orchestration, so it would need its own, more careful evaluation.
- **Hybrid retrieval** (lexical + semantic) to fix the exact gap the q6 spotlight surfaced, rather than relying on the keyword-search fallback to catch it after the fact.
- **Syntax-aware chunking** (function/class boundaries via an AST parse) instead of fixed-size windows.
