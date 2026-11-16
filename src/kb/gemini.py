"""Gemini through the user's local web reverse proxy (gemini_reverse_api, OpenAI-compatible, 127.0.0.1:3086), the
model `delegate` runs on. The proxy is its own project with its own login and cookie refresh; here only its URL,
bearer (an account group: pool, pro, an id or an email) and model:

    GEMINI_PROXY_URL=http://127.0.0.1:3086   GEMINI_PROXY_TOKEN=pool   GEMINI_MODEL=gemini-4-hydrogen

The web app has no tool calling and no system role: the proxy glues the messages into one prompt, and delegate
asks for tool calls as JSON blocks in the text.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

QUEUE_WAIT = 900.0   # how long a request waits for the one in flight (a completion times out at 240 s)
# A PRO account answers about 2400 messages per 5 hours; quick bursts get 429: keep a gap, and wait when refused.
MIN_GAP = float(os.environ.get("KB_GEMINI_GAP", "1.0"))
RATE_WAITS = (8.0, 15.0, 30.0, 60.0)


class Unavailable(Exception):
    """The proxy is down, refuses or answers nothing: the caller does the work itself."""


def url() -> str:
    return os.environ.get("GEMINI_PROXY_URL", "http://127.0.0.1:3086").rstrip("/")


def token() -> str:
    return os.environ.get("GEMINI_PROXY_TOKEN", "pool")


def model() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-4-hydrogen")


def status() -> dict[str, Any]:
    """The proxy's /health: accounts, which are alive, quota left."""
    try:
        r = httpx.get(url() + "/health", timeout=3)
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"{url()}: {type(e).__name__}"}
    if r.status_code != 200:
        return {"ok": False, "error": f"{url()}/health: HTTP {r.status_code}"}
    data = r.json()
    return {"ok": bool(data.get("usable")), "url": url(), "model": model(), "accounts": data.get("accounts"),
            "usable": data.get("usable"),
            "slots": [{k: s.get(k) for k in ("id", "label", "status", "cooldownUntil", "lastError")}
                      | {"quota_5h": (s.get("quota") or {}).get("window5h", {}).get("remaining")}
                      for s in data.get("slots", [])]}


@contextlib.contextmanager
def _one_at_a_time(wait: float = QUEUE_WAIT) -> Iterator[None]:
    """One completion at a time across threads and research-mcp processes, MIN_GAP apart."""
    try:
        import fcntl
    except ImportError:  # Windows: no flock, just go ahead
        yield
        return
    from kb.config import settings

    data = Path(os.environ.get("KB_DATA_DIR") or settings.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    with open(data / "gemini.lock", "w") as fh:
        deadline = time.monotonic() + wait
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise Unavailable(f"another Gemini request has been running for over {wait:.0f} s") from None
                time.sleep(0.5)
        last = data / "gemini.last"   # when the previous completion started, in any process
        try:
            time.sleep(max(0.0, float(last.read_text()) + MIN_GAP - time.time()))
        except (OSError, ValueError):
            pass
        last.write_text(str(time.time()))
        yield


def chat(messages: list[dict[str, Any]], timeout: float = 240.0) -> str:
    """One completion; returns its text. Completions run one at a time; a refusal (429) is retried after a wait."""
    body = {"model": model(), "messages": messages, "temperature": 0.2}
    for pause in (*RATE_WAITS, None):
        try:
            with _one_at_a_time():
                r = httpx.post(url() + "/v1/chat/completions", json=body, timeout=timeout,
                               headers={"Authorization": f"Bearer {token()}"})
        except httpx.HTTPError as e:
            raise Unavailable(f"Gemini proxy {url()}: {type(e).__name__}: {e} — is gemini-proxy.service running?"
                              ) from e
        except OSError as e:   # the lock or pacing file in the data folder
            raise Unavailable(f"cannot queue the Gemini request: {e}") from e
        if r.status_code != 429 or pause is None:
            break
        time.sleep(pause)
    if r.status_code != 200:
        raise Unavailable(f"Gemini proxy HTTP {r.status_code}: {r.text[:300]}")
    try:
        text = r.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise Unavailable(f"unexpected reply from the Gemini proxy: {r.text[:300]}") from e
    if not isinstance(text, str) or not text.strip():
        raise Unavailable(f"Gemini returned an empty reply: {r.text[:300]}")
    return text
