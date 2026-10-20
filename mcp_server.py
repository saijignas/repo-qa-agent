"""An MCP (Model Context Protocol) server that exposes this project's repo tools
to any MCP client, such as Claude Desktop or Claude Code.

Tools:
  search_keyword  exact, case-insensitive substring search over the indexed source files
  fetch_file      read an indexed file, or a line range of it
  list_files      the files that are indexed (what the other two tools can see)
  ask_repo        the full agent: retrieval, keyword fallback, cited answer or decline

Configuration comes from environment variables, never from tool arguments, so an
MCP client cannot point the server at a different directory:
  REPO_QA_REPO_ROOT    the repository the tools read (required)
  REPO_QA_STORE        the Chroma index built by ingest.py (default ./chroma_store)
  REPO_QA_COLLECTION   Chroma collection name (default repo_qa)
  GEMINI_API_KEY       only needed by ask_repo, and read by llm_client, never echoed here

Safety decisions (an MCP tool is called by a model, so its inputs are untrusted):
- The file tools can only see the files the indexer indexes (.py, .md, .yml, .yaml,
  minus .git, virtualenvs, node_modules and similar). A `.env`, a key file, a
  path with `..`, an absolute path or a file inside `.venv` is refused, not read.
- All limits (term length, hit count, output size, question length) are enforced
  here, so a client cannot ask for an unbounded amount of work or output.
- The stdio transport uses stdout for the protocol itself, and the agent prints
  (for example a retry notice from the Gemini client). mcp 2.x already diverts stray
  stdout writes away from the protocol stream; ask_repo also redirects them to stderr
  itself, so the server stays correct on older SDKs and the notices still reach the
  server log.
- Anticipated failures (bad input, a refused path, no index, no API key, a failed
  model request) are raised as ToolError, so the client sees a plain, actionable
  message. Anything unexpected is left to the SDK, which reports a generic error and
  keeps the exception details (paths, stack traces) away from the client.
"""
import contextlib
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional

import requests
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import agent as hand_rolled
import tools as repo_tools
from ingest import find_source_files

MAX_TERM_CHARS = 200
MAX_HITS_LIMIT = 50
MAX_QUESTION_CHARS = 2000
MAX_FILE_CHARS = 60_000
MAX_LISTED_FILES = 500


def load_config(env=None) -> Dict[str, Any]:
    """Reads and validates the server configuration from environment variables."""
    env = os.environ if env is None else env
    repo_root = env.get("REPO_QA_REPO_ROOT")
    if not repo_root:
        raise SystemExit("REPO_QA_REPO_ROOT is not set: point it at the repository the tools should read.")
    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"REPO_QA_REPO_ROOT is not a directory: {root}")
    return {
        "repo_root": root,
        "store_dir": Path(env.get("REPO_QA_STORE", "./chroma_store")).expanduser().resolve(),
        "collection": env.get("REPO_QA_COLLECTION", "repo_qa"),
    }


def _indexed_files(repo_root: Path) -> List[str]:
    return sorted(p.relative_to(repo_root).as_posix() for p in find_source_files(repo_root))


def build_server(
    repo_root,
    store_dir,
    collection: str = "repo_qa",
    answerer: Optional[Callable] = None,
) -> MCPServer:
    """Builds the MCP server. `answerer` defaults to the hand-rolled agent and is
    injectable so tests can run ask_repo without Gemini or Chroma."""
    repo_root = Path(repo_root).resolve()
    store_dir = Path(store_dir).resolve()
    answerer = answerer or hand_rolled.answer

    mcp = MCPServer(
        "repo-qa",
        instructions=(
            "Read-only tools over one indexed code repository. Use list_files to see what exists, "
            "search_keyword for exact identifiers, fetch_file to read a file or line range, and "
            "ask_repo for a cited natural-language answer that declines when the repo does not say."
        ),
    )

    @mcp.tool()
    def list_files() -> List[str]:
        """List the repository files the other tools can read (relative paths, indexed file types only)."""
        return _indexed_files(repo_root)[:MAX_LISTED_FILES]

    @mcp.tool()
    def search_keyword(term: str, max_hits: int = 8) -> List[Dict[str, Any]]:
        """Exact, case-insensitive substring search. Returns [{source, line_number, line_text}].
        Best for identifiers and config values that semantic search under-ranks, e.g. MAX_RETRIES."""
        term = (term or "").strip()
        if not term:
            raise ToolError("term must not be empty")
        if len(term) > MAX_TERM_CHARS:
            raise ToolError(f"term is too long ({len(term)} characters; the limit is {MAX_TERM_CHARS})")
        if not 1 <= max_hits <= MAX_HITS_LIMIT:
            raise ToolError(f"max_hits must be between 1 and {MAX_HITS_LIMIT}")
        return repo_tools.search_keyword(repo_root, term, max_hits=max_hits)

    @mcp.tool()
    def fetch_file(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        """Read an indexed file (relative path as returned by list_files), optionally lines
        start_line..end_line (1-based, inclusive). Output is capped; the cap is announced in the text."""
        wanted = PurePosixPath((path or "").replace("\\", "/")).as_posix()
        if wanted not in set(_indexed_files(repo_root)):
            raise ToolError(f"not an indexed file: {path!r}. Call list_files to see what can be read.")
        if start_line is not None and start_line < 1:
            raise ToolError("start_line must be 1 or greater")
        if start_line is not None and end_line is not None and end_line < start_line:
            raise ToolError("end_line must not be before start_line")
        text = repo_tools.fetch_file(repo_root, wanted, start_line=start_line, end_line=end_line)
        if len(text) > MAX_FILE_CHARS:
            text = text[:MAX_FILE_CHARS] + f"\n[truncated at {MAX_FILE_CHARS} characters; request a narrower line range]"
        return text

    @mcp.tool()
    def ask_repo(question: str, k: int = 5) -> Dict[str, Any]:
        """Answer a question about the repository using retrieval plus a keyword fallback.
        Returns {answer, sources, tools_used}. Cites [file:line] and declines when the repo does not
        contain the answer. Requires the index (run ingest.py) and GEMINI_API_KEY on the server."""
        question = (question or "").strip()
        if not question:
            raise ToolError("question must not be empty")
        if len(question) > MAX_QUESTION_CHARS:
            raise ToolError(f"question is too long ({len(question)} characters; the limit is {MAX_QUESTION_CHARS})")
        if not 1 <= k <= 20:
            raise ToolError("k must be between 1 and 20")
        if not store_dir.exists():
            raise ToolError(f"no index found at {store_dir}; run ingest.py on the repository first")
        if answerer is hand_rolled.answer and not os.environ.get("GEMINI_API_KEY"):
            raise ToolError("GEMINI_API_KEY is not set on the server, so ask_repo cannot generate answers; search_keyword and fetch_file still work")
        try:
            # stdout carries the protocol on the stdio transport, so keep the agent's prints off it
            with contextlib.redirect_stdout(sys.stderr):
                result = answerer(question, repo_root, store_dir, collection, k)
        except requests.exceptions.RequestException as exc:
            raise ToolError(f"the language-model request failed ({type(exc).__name__}); try again shortly") from exc
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "tools_used": result["tools_used"],
        }

    return mcp


if __name__ == "__main__":
    cfg = load_config()
    build_server(cfg["repo_root"], cfg["store_dir"], cfg["collection"]).run()
