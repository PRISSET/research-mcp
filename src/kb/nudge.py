"""PreToolUse hook: hand heavy reading to the Gemini subagent (delegate) instead of pulling it into Claude's context.

The first time in a session that Claude goes to read something big — a WebFetch, a fetch of several pages, a Read
of a large file without a line range — the call is refused once with a pointer to delegate. Calling again with the
same target goes through, so raw text is still there when it is really needed. Nothing is refused while the Gemini
proxy is down.

    uv run kb gemini hook install      # adds it to ~/.claude/settings.json
"""

from __future__ import annotations

import fcntl
import json
import sys
from pathlib import Path
from typing import Any

from kb.config import ROOT

MARK = "kb.nudge"
MATCHER = "WebFetch|Read|mcp__research__fetch"
BIG_FILE = 40_000      # bytes: a Read of more than this without offset / limit goes to delegate first
KEEP_SESSIONS = 50
REASON = ("This pulls a lot of text into your context. Hand it to delegate(task=<what you need from it>, "
          "context=<what you already know>{project}): the Gemini subagent reads it and returns only the facts with "
          "their sources, for a few hundred tokens. If you need the raw text itself, make the same call again: it "
          "will go through.")


def _command() -> str:
    python = ROOT / ".venv" / "bin" / "python"   # every Read passes here: skip uv's start-up
    return f'"{python}" -m {MARK}' if python.exists() else f'uv run --directory "{ROOT}" --no-sync python -m {MARK}'


def install() -> str:
    from kb import guard

    return guard.add_hooks(MARK, {"PreToolUse": (_command(), MATCHER)}, timeout=5)


def uninstall() -> bool:
    from kb import guard

    return guard.remove_hooks(MARK)


def target(tool: str, args: dict[str, Any]) -> str | None:
    """What a heavy read would pull in (a URL, several, a big file), or None for a light call."""
    if tool == "WebFetch":
        return args.get("url") or None
    if tool == "mcp__research__fetch":
        many = list(args.get("urls") or []) + list(args.get("paper_ids") or [])
        return " ".join(sorted(map(str, many))) if len(many) >= 2 else None
    if tool == "Read":
        path = Path(str(args.get("file_path") or ""))
        if args.get("offset") or args.get("limit") or path.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif",
                                                                               ".webp", ".pdf", ".ipynb"):
            return None
        try:
            return str(path) if path.is_file() and path.stat().st_size > BIG_FILE else None
        except OSError:
            return None
    return None


def _proxy_up() -> bool:
    import httpx

    from kb import gemini

    try:
        r = httpx.get(gemini.url() + "/health", timeout=1.0)
        return r.status_code == 200 and bool(r.json().get("usable", True))
    except (httpx.HTTPError, ValueError):
        return False


def _first_time(session: str, what: str) -> bool:
    """True the first time this session reads `what` (and remember it); False on a repeat."""
    from kb.memory import llm

    home = llm.home().parent
    home.mkdir(parents=True, exist_ok=True)
    state_file = home / "nudge.json"
    with open(home / "nudge.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        seen = state.pop(session, [])
        first = what not in seen
        state[session] = (seen + [what])[-200:] if first else seen   # re-inserted: this session is the newest
        state = dict(list(state.items())[-KEEP_SESSIONS:])
        tmp = state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(state_file)
    return first


def hook(data: dict[str, Any]) -> dict[str, Any] | None:
    what = target(str(data.get("tool_name") or ""), data.get("tool_input") or {})
    if not what or not _proxy_up() or not _first_time(str(data.get("session_id") or "-"), what):
        return None
    project = f', project="{data["cwd"]}"' if data.get("tool_name") == "Read" and data.get("cwd") else ""
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                   "permissionDecisionReason": REASON.format(project=project)}}


def main() -> int:
    try:
        out = hook(json.loads(sys.stdin.read() or "{}"))
    except Exception as e:   # never block a tool because the nudge broke
        print(f"{MARK}: {type(e).__name__}: {e}", file=sys.stderr)
        return 0
    if out:
        print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
