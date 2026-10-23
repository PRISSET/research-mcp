"""DeepSeek through its free web chat: the bundled proxy (deepseek_proxy/, TypeScript) speaks the OpenAI chat API
on 127.0.0.1 and forwards to chat.deepseek.com with the user's web login token, so `delegate` can run a research
subagent without a paid API.

    uv run kb deepseek install        # npm build; systemd user service research-deepseek
    uv run kb keys set DEEPSEEK_CHAT_TOKEN <token>   # chat.deepseek.com → DevTools → Local Storage → userToken

The token stays with research-mcp and goes in each request's Authorization header, never into the proxy's
environment (see _proxy_env).

Several accounts: tokens joined with "|" are rotated by the proxy and a throttled one is skipped.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

PROXY = Path(__file__).resolve().parents[2] / "deepseek_proxy"
PORT = int(os.environ.get("KB_DEEPSEEK_PORT", "3085"))
UNIT = Path("~/.config/systemd/user/research-deepseek.service").expanduser()
MODEL = "deepseek-chat"
QUEUE_WAIT = 900.0   # how long a request waits for the one in flight (a completion times out at 240 s)


class Unavailable(Exception):
    """No token, no build, or the proxy does not answer: the caller does the work itself."""


def url() -> str:
    return os.environ.get("DEEPSEEK_PROXY_URL", f"http://127.0.0.1:{PORT}").rstrip("/")


def token() -> str:
    """The token as saved from the browser: either the bare value or Local Storage's {"value": …} JSON."""
    raw = os.environ.get("DEEPSEEK_CHAT_TOKEN", "").strip()
    if raw.startswith("{"):
        try:
            return str(json.loads(raw).get("value", ""))
        except (ValueError, AttributeError):
            return ""
    return raw


def _run(*cmd: str) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROXY)


def install() -> str:
    """Build the proxy with npm and keep it running as a systemd user service when possible."""
    if not shutil.which("npm") or not shutil.which("node"):
        sys.exit("needs Node.js 18+ with npm on PATH")
    _run("npm", "ci", "--no-audit", "--no-fund")
    _run("npm", "run", "build")
    if not shutil.which("systemctl"):
        return "started on demand by research-mcp"
    UNIT.parent.mkdir(parents=True, exist_ok=True)
    UNIT.write_text(f"""[Unit]
Description=DeepSeek web proxy for research-mcp (127.0.0.1:{PORT})

[Service]
Environment=PROXY_PORT={PORT}
WorkingDirectory={PROXY}
ExecStart={shutil.which("node")} {PROXY / "dist" / "server.js"}
Restart=on-failure

[Install]
WantedBy=default.target
""", encoding="utf-8")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", "research-deepseek.service"], check=True)
    subprocess.run(["systemctl", "--user", "restart", "research-deepseek.service"], check=True)
    return "systemd user service research-deepseek (starts with your session)"


def uninstall() -> None:
    if UNIT.exists():
        subprocess.run(["systemctl", "--user", "disable", "--now", "research-deepseek.service"], check=False)
        UNIT.unlink()
    shutil.rmtree(PROXY / "dist", ignore_errors=True)


def _proxy_env() -> dict[str, str]:
    """The proxy answers any origin (CORS *) and falls back to DEEPSEEK_CHAT_TOKEN from its environment, so a web
    page could use the account through it: it never gets the token, every request brings its own."""
    env = {k: v for k, v in os.environ.items() if k != "DEEPSEEK_CHAT_TOKEN"}
    return {**env, "PROXY_PORT": str(PORT)}


def healthy() -> bool:
    try:
        return httpx.get(url() + "/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def ensure_running(wait: float = 8.0) -> None:
    """Start the built proxy in the background when no service keeps it up."""
    if not token():
        raise Unavailable("DEEPSEEK_CHAT_TOKEN is not set: `uv run kb keys set DEEPSEEK_CHAT_TOKEN <token>` "
                          "(chat.deepseek.com → DevTools → Application → Local Storage → userToken)")
    if healthy():
        return
    server = PROXY / "dist" / "server.js"
    if not server.exists() or not shutil.which("node"):
        raise Unavailable("the DeepSeek proxy is not built: `uv run kb deepseek install`")
    subprocess.Popen(["node", str(server)], cwd=PROXY, env=_proxy_env(),
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if healthy():
            return
        time.sleep(0.3)
    raise Unavailable(f"the DeepSeek proxy did not start on {url()}")


@contextlib.contextmanager
def _one_at_a_time(wait: float = QUEUE_WAIT) -> Iterator[None]:
    """The proxy reuses one web chat session per token, and parallel completions in it fail: every caller
    (threads, other research-mcp processes) waits until the running completion has finished, then goes."""
    try:
        import fcntl
    except ImportError:  # Windows: no flock, just go ahead
        yield
        return
    from kb.config import settings

    data = Path(os.environ.get("KB_DATA_DIR") or settings.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    with open(data / "deepseek.lock", "w") as fh:
        deadline = time.monotonic() + wait
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise Unavailable(f"another DeepSeek request has been running for over {wait:.0f} s") from None
                time.sleep(0.5)
        yield


def chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
         timeout: float = 240.0) -> dict[str, Any]:
    """One completion; returns the assistant message (content, tool_calls). Completions run one at a time."""
    ensure_running()
    # client_prompt: keep our system message, leave out the proxy's DeepSeek Harness workflow
    body: dict[str, Any] = {"model": MODEL, "messages": messages, "stream": False, "client_prompt": True}
    if tools:
        body["tools"] = tools
    try:
        with _one_at_a_time():
            r = httpx.post(url() + "/v1/chat/completions", json=body, timeout=timeout,
                           headers={"Authorization": f"Bearer {token()}"})
    except httpx.HTTPError as e:
        raise Unavailable(f"DeepSeek proxy: {type(e).__name__}: {e}") from e
    if r.status_code != 200:
        hint = ("; the token is wrong: take userToken from DevTools → Application → Local Storage → "
                "https://chat.deepseek.com (not a cookie), then `uv run kb keys set DEEPSEEK_CHAT_TOKEN …`"
                if "invalid token" in r.text else "")
        raise Unavailable(f"DeepSeek proxy HTTP {r.status_code}: {r.text[:300]}{hint}")
    return r.json()["choices"][0]["message"]
