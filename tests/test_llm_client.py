"""Tests for the retry behaviour of llm_client.call_gemini. No network: the
shared requests session is replaced with a scripted fake."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import llm_client


class FakeResponse:
    def __init__(self, status, headers=None, body=None):
        self.status_code = status
        self.headers = headers or {}
        self._body = body or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._body


OK = FakeResponse(200, body={"candidates": [{"content": {"parts": [{"text": "hello"}]}}]})


def script(monkeypatch, responses):
    calls = {"n": 0, "sleeps": []}

    def post(*args, **kwargs):
        r = responses[calls["n"]]
        calls["n"] += 1
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm_client.SESSION, "post", post)
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: calls["sleeps"].append(s))
    return calls


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_retries_transient_errors_then_succeeds(monkeypatch, status):
    calls = script(monkeypatch, [FakeResponse(status), OK])
    assert llm_client.call_gemini("hi") == "hello"
    assert calls["n"] == 2
    assert calls["sleeps"] == [5.0]


def test_honors_retry_after_header(monkeypatch):
    calls = script(monkeypatch, [FakeResponse(429, headers={"Retry-After": "2"}), OK])
    llm_client.call_gemini("hi")
    assert calls["sleeps"] == [2.0]


def test_does_not_retry_client_errors(monkeypatch):
    calls = script(monkeypatch, [FakeResponse(400), OK])
    with pytest.raises(RuntimeError):
        llm_client.call_gemini("hi")
    assert calls["n"] == 1


def test_gives_up_after_the_retry_budget(monkeypatch):
    calls = script(monkeypatch, [FakeResponse(503)] * llm_client.MAX_RETRIES_ON_RATE_LIMIT)
    with pytest.raises(RuntimeError):
        llm_client.call_gemini("hi")
    assert calls["n"] == llm_client.MAX_RETRIES_ON_RATE_LIMIT


def test_retries_a_read_timeout_then_succeeds(monkeypatch):
    import requests

    calls = script(monkeypatch, [requests.exceptions.ReadTimeout("slow"), OK])
    assert llm_client.call_gemini("hi") == "hello"
    assert calls["n"] == 2
    assert calls["sleeps"] == [5]


def test_a_timeout_on_every_attempt_is_finally_raised(monkeypatch):
    import requests

    script(monkeypatch, [requests.exceptions.ConnectionError("down")] * llm_client.MAX_RETRIES_ON_RATE_LIMIT)
    with pytest.raises(requests.exceptions.ConnectionError):
        llm_client.call_gemini("hi")
