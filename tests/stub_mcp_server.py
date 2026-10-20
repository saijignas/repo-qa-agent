"""A real MCP server process for the protocol-level tests. The agent is replaced by
a stub that PRINTS TO STDOUT, as the real Gemini client does when it retries, to
prove the server keeps that noise off the protocol stream."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server import build_server, load_config


def noisy_answerer(question, repo_root, store_dir, collection, k):
    print("  [HTTP 503, attempt 1/5 -- waiting 5s]", flush=True)  # would corrupt stdio if it reached stdout
    return {"answer": f"stub answer to: {question} [a.py:1-2]", "sources": ["a.py"], "tools_used": []}


if __name__ == "__main__":
    cfg = load_config()
    cfg["store_dir"].mkdir(parents=True, exist_ok=True)
    build_server(cfg["repo_root"], cfg["store_dir"], cfg["collection"], answerer=noisy_answerer).run()
