"""End-to-end smoke test of the MCP server with the REAL agent: launches mcp_server.py
as a subprocess (as Claude Desktop or Claude Code would), lists its tools, runs a
keyword search, tries to read a secret, and asks one real question through Gemini.

Needs the index (python ingest.py <repo>) and GEMINI_API_KEY in the environment.
Usage: python eval/mcp_smoke.py <repo_root> [--store ./chroma_store]
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters

ROOT = Path(__file__).parent.parent


async def main(repo_root, store):
    env = dict(os.environ)
    env.update({"REPO_QA_REPO_ROOT": str(repo_root), "REPO_QA_STORE": str(store)})
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_server.py")], env=env, cwd=str(ROOT))
    async with Client(params) as c:
        print("tools:", [t.name for t in (await c.list_tools()).tools])
        hits = await c.call_tool("search_keyword", {"term": "MAX_RETRIES"})
        print("search_keyword MAX_RETRIES ->", [(h["source"], h["line_number"]) for h in hits.structured_content["result"]][:3])
        refused = await c.call_tool("fetch_file", {"path": "../.env"})
        print("fetch_file ../.env -> is_error:", refused.is_error, "|", refused.content[0].text)
        asked = await c.call_tool("ask_repo", {"question": "What value is MAX_RETRIES set to, and where is it defined?"})
        sc = asked.structured_content
        sc = sc.get("result", sc)
        print("ask_repo is_error:", asked.is_error)
        print("answer:", sc["answer"])
        print("sources:", sc["sources"], "| tools_used:", sc["tools_used"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo_root")
    ap.add_argument("--store", default=str(ROOT / "chroma_store"))
    a = ap.parse_args()
    asyncio.run(main(Path(a.repo_root).resolve(), Path(a.store).resolve()))
