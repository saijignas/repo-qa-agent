"""Gemini API client. Reuses the exact pattern from
Consumer-Complaint-Triage/src/llm_compare.py: key read from the
GEMINI_API_KEY environment variable (never hardcoded), sent via the
x-goog-api-key HEADER rather than a ?key=... query string (a query
string can end up embedded in an exception's own text -- e.g.
requests' HTTPError includes the full URL -- and get logged somewhere
it shouldn't). gemini-3.1-flash-lite is used for its free-tier quota
headroom, same reason as that project."""
import os
import time
from pathlib import Path

import requests

GEMINI_MODEL = "gemini-3.1-flash-lite"
SESSION = requests.Session()
MAX_RETRIES_ON_RATE_LIMIT = 5
# 429 is the free-tier rate limit; 500/502/503/504 are transient server-side errors
# (a 503 'overloaded' response failed a live test run before this was added).
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
REQUEST_TIMEOUT_SECONDS = 60


def _load_dotenv_fallback():
    """If GEMINI_API_KEY isn't in the process environment (e.g. it was
    set in a different terminal/process tree than the one running this
    script), fall back to reading a git-ignored .env file next to this
    script. Never overwrites a value that's already in os.environ, and
    never prints or logs the value it loads."""
    if os.environ.get("GEMINI_API_KEY"):
        return
    env_path = Path(__file__).parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "GEMINI_API_KEY":
            os.environ["GEMINI_API_KEY"] = value.strip()
            return


_load_dotenv_fallback()


def call_gemini(prompt, max_output_tokens=1024):
    """Retries on 429 (rate limit), transient 5xx errors and read timeouts with exponential backoff, honoring a
    Retry-After header when the API sends one rather than guessing.
    This is a real free-tier constraint (a requests-per-minute cap, not
    just the daily one) discovered while running the eval harness, so
    it's handled properly rather than papered over by just re-running
    the script by hand."""
    key = os.environ["GEMINI_API_KEY"]
    delay = 5
    for attempt in range(1, MAX_RETRIES_ON_RATE_LIMIT + 1):
        try:
            resp = SESSION.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                headers={"x-goog-api-key": key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": max_output_tokens},
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            # A read timeout is the same transient-service-trouble family as a 503
            # (it ended a live eval run before this was handled).
            if attempt == MAX_RETRIES_ON_RATE_LIMIT:
                raise
            print(f"  [{type(exc).__name__}, attempt {attempt}/{MAX_RETRIES_ON_RATE_LIMIT} -- waiting {delay:.0f}s]", flush=True)
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES_ON_RATE_LIMIT:
            wait = float(resp.headers.get("Retry-After", delay))
            print(f"  [HTTP {resp.status_code}, attempt {attempt}/{MAX_RETRIES_ON_RATE_LIMIT} -- waiting {wait:.0f}s]", flush=True)
            time.sleep(wait)
            delay *= 2
            continue
        resp.raise_for_status()
        parts = resp.json()["candidates"][0]["content"]["parts"]
        return next(p["text"] for p in parts if "text" in p)
