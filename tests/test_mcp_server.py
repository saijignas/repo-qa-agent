"""Tests for the MCP server. Two levels:

- in-process: an MCP client connected straight to the server object, to exercise
  every tool, every limit and every refusal quickly;
- protocol: a real server subprocess talking JSON-RPC over stdio, which is how
  Claude Desktop or Claude Code actually runs it. That level also proves that
  noise printed to stdout by the agent does not corrupt the protocol stream.

No test calls Chroma or Gemini: ask_repo gets a stub agent.
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from mcp import Client
from mcp.client.stdio import StdioServerParameters

import mcp_server

HERE = Path(__file__).parent


def stub_answerer(question, repo_root, store_dir, collection, k):
    return {"answer": f"stub: {question}", "sources": ["a.py"], "tools_used": []}


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "nested").mkdir()
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / "a.py").write_text("MAX_RETRIES = 3\nline2\nline3\nline4\n")
    (root / "docs" / "README.md").write_text("# Title\nretry policy\n")
    (root / "nested" / "b.yaml").write_text("replicas: 1\n")
    (root / ".env").write_text("GEMINI_API_KEY=should-never-be-readable\n")
    (root / "secrets.txt").write_text("not an indexed type\n")
    (root / ".venv" / "lib" / "x.py").write_text("MAX_RETRIES = 99\n")
    (tmp_path / "outside.py").write_text("OUTSIDE = 1\n")
    return root


def make_server(repo, answerer=stub_answerer):
    store = repo.parent / "store"
    store.mkdir(exist_ok=True)
    return mcp_server.build_server(repo, store, "repo_qa", answerer=answerer)


def call(server, tool, args=None):
    async def run():
        async with Client(server) as c:
            return await c.call_tool(tool, args or {})

    return asyncio.run(run())


def text(result):
    return result.content[0].text


def items(result):
    sc = result.structured_content
    return sc["result"] if "result" in sc else sc


def test_exposes_four_tools_each_with_a_description(repo):
    async def run():
        async with Client(make_server(repo)) as c:
            return (await c.list_tools()).tools

    tools = asyncio.run(run())
    assert sorted(t.name for t in tools) == ["ask_repo", "fetch_file", "list_files", "search_keyword"]
    assert all(t.description and t.input_schema for t in tools)


def test_list_files_shows_only_indexed_files(repo):
    result = call(make_server(repo), "list_files")
    assert not result.is_error
    assert items(result) == ["a.py", "docs/README.md", "nested/b.yaml"]


def test_search_keyword_finds_hits_and_ignores_the_virtualenv(repo):
    result = call(make_server(repo), "search_keyword", {"term": "max_retries"})
    hits = items(result)
    assert [h["source"] for h in hits] == ["a.py"]
    assert hits[0]["line_number"] == 1


@pytest.mark.parametrize("args,fragment", [
    ({"term": ""}, "must not be empty"),
    ({"term": "   "}, "must not be empty"),
    ({"term": "x" * 201}, "too long"),
    ({"term": "a", "max_hits": 0}, "max_hits"),
    ({"term": "a", "max_hits": 51}, "max_hits"),
])
def test_search_keyword_rejects_bad_input_with_a_clear_message(repo, args, fragment):
    result = call(make_server(repo), "search_keyword", args)
    assert result.is_error
    assert fragment in text(result)


def test_fetch_file_returns_a_file_and_a_line_range(repo):
    server = make_server(repo)
    assert text(call(server, "fetch_file", {"path": "a.py"})) == "MAX_RETRIES = 3\nline2\nline3\nline4"
    assert text(call(server, "fetch_file", {"path": "a.py", "start_line": 2, "end_line": 3})) == "line2\nline3"
    assert text(call(server, "fetch_file", {"path": "nested/b.yaml"})) == "replicas: 1"


@pytest.mark.parametrize("path", [
    ".env",
    "secrets.txt",
    ".venv/lib/x.py",
    "../outside.py",
    "docs/../../outside.py",
    "..\\outside.py",
    "nope.py",
    "",
])
def test_fetch_file_refuses_anything_that_is_not_an_indexed_file(repo, path):
    result = call(make_server(repo), "fetch_file", {"path": path})
    assert result.is_error
    assert "not an indexed file" in text(result)
    assert "should-never-be-readable" not in text(result)


def test_fetch_file_refuses_an_absolute_path(repo):
    result = call(make_server(repo), "fetch_file", {"path": str((repo.parent / "outside.py").resolve())})
    assert result.is_error
    assert "OUTSIDE" not in text(result)


@pytest.mark.parametrize("args,fragment", [
    ({"path": "a.py", "start_line": 0}, "start_line"),
    ({"path": "a.py", "start_line": 3, "end_line": 2}, "end_line"),
])
def test_fetch_file_rejects_bad_line_ranges(repo, args, fragment):
    result = call(make_server(repo), "fetch_file", args)
    assert result.is_error and fragment in text(result)


def test_fetch_file_truncates_very_large_output_and_says_so(repo):
    (repo / "big.py").write_text("x = 1\n" * 20_000)
    out = text(call(make_server(repo), "fetch_file", {"path": "big.py"}))
    assert len(out) < mcp_server.MAX_FILE_CHARS + 200
    assert "truncated" in out


def test_ask_repo_returns_the_agents_answer_sources_and_tools(repo):
    result = call(make_server(repo), "ask_repo", {"question": "How are retries handled?"})
    assert not result.is_error
    sc = result.structured_content
    sc = sc["result"] if "result" in sc else sc
    assert sc == {"answer": "stub: How are retries handled?", "sources": ["a.py"], "tools_used": []}


@pytest.mark.parametrize("args,fragment", [
    ({"question": ""}, "must not be empty"),
    ({"question": "q" * 2001}, "too long"),
    ({"question": "ok", "k": 0}, "k must be"),
    ({"question": "ok", "k": 21}, "k must be"),
])
def test_ask_repo_rejects_bad_input(repo, args, fragment):
    result = call(make_server(repo), "ask_repo", args)
    assert result.is_error and fragment in text(result)


def test_ask_repo_reports_a_missing_index(repo):
    server = mcp_server.build_server(repo, repo.parent / "no-such-store", "repo_qa", answerer=stub_answerer)
    result = call(server, "ask_repo", {"question": "anything"})
    assert result.is_error and "ingest.py" in text(result)


def test_ask_repo_with_the_real_agent_and_no_key_says_so_without_crashing(repo, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = call(make_server(repo, answerer=None), "ask_repo", {"question": "anything"})
    assert result.is_error
    assert "GEMINI_API_KEY is not set" in text(result)
    assert "search_keyword and fetch_file still work" in text(result)


def test_ask_repo_turns_a_failed_model_request_into_a_clear_tool_error(repo):
    def failing(*a, **k):
        raise requests.exceptions.ReadTimeout("slow")

    result = call(make_server(repo, answerer=failing), "ask_repo", {"question": "anything"})
    assert result.is_error
    assert "language-model request failed" in text(result) and "ReadTimeout" in text(result)


def test_an_unexpected_crash_does_not_leak_details_to_the_client(repo):
    def boom(*a, **k):
        raise RuntimeError("secret internal path C:/Users/someone/private")

    result = call(make_server(repo, answerer=boom), "ask_repo", {"question": "anything"})
    assert result.is_error
    assert "private" not in text(result)


def test_load_config_requires_a_real_directory(tmp_path):
    with pytest.raises(SystemExit):
        mcp_server.load_config({})
    with pytest.raises(SystemExit):
        mcp_server.load_config({"REPO_QA_REPO_ROOT": str(tmp_path / "missing")})
    cfg = mcp_server.load_config({"REPO_QA_REPO_ROOT": str(tmp_path)})
    assert cfg["repo_root"] == tmp_path.resolve()
    assert cfg["collection"] == "repo_qa"


# ---- protocol level: a real server process over stdio -------------------------

def stdio_params(repo):
    env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
    env.update({"REPO_QA_REPO_ROOT": str(repo), "REPO_QA_STORE": str(repo.parent / "store")})
    return StdioServerParameters(command=sys.executable, args=[str(HERE / "stub_mcp_server.py")], env=env)


def test_a_real_stdio_server_lists_tools_and_serves_calls(repo):
    async def run():
        async with Client(stdio_params(repo)) as c:
            tools = [t.name for t in (await c.list_tools()).tools]
            hits = await c.call_tool("search_keyword", {"term": "MAX_RETRIES"})
            refused = await c.call_tool("fetch_file", {"path": ".env"})
            return tools, hits, refused

    tools, hits, refused = asyncio.run(run())
    assert sorted(tools) == ["ask_repo", "fetch_file", "list_files", "search_keyword"]
    assert [h["source"] for h in items(hits)] == ["a.py"]
    assert refused.is_error and "not an indexed file" in text(refused)


def test_stdout_noise_from_the_agent_does_not_corrupt_the_stdio_protocol(repo):
    async def run():
        async with Client(stdio_params(repo)) as c:
            first = await c.call_tool("ask_repo", {"question": "first"})
            second = await c.call_tool("list_files", {})  # the stream is still healthy afterwards
            return first, second

    first, second = asyncio.run(run())
    assert not first.is_error and "stub answer to: first" in text(first)
    assert not second.is_error and "a.py" in text(second)
