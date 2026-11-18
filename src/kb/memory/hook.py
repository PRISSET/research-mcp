"""Claude Code hooks of the memory: what earlier chats left (topics only, a few dozen tokens) at session start,
and a background digest of the chat by the local model before compaction and at session end."""

from __future__ import annotations

import json
import sys
from typing import Any

from kb.config import ROOT

MARK = "kb.memory.hook"   # how our hooks are recognised in settings.json
HOOKS = {"SessionStart": ("session", None), "PreCompact": ("end", None), "SessionEnd": ("end", None)}


def _command(event: str) -> str:
    return f'uv run --directory "{ROOT}" --no-sync python -m {MARK} {event}'


def install() -> str:
    from kb import guard

    return guard.add_hooks(MARK, {name: (_command(event), matcher) for name, (event, matcher) in HOOKS.items()})


def uninstall() -> bool:
    from kb import guard

    return guard.remove_hooks(MARK)


def context() -> str | None:
    from kb.memory import store
    from kb.store import Database

    with Database() as db:
        ts = store.topics(db, limit=30)
    if not ts:
        return None
    listed = "; ".join(f"{t['topic']} ({t['notes']})" for t in ts)
    return ("Memory from earlier chats and RESEARCH.md, by topic (notes): " + listed + ". When the task touches "
            "one, call recall(query) for the notes instead of reading RESEARCH.md; save a lasting fact with "
            "remember(topic, text). The user's ComfyUI workflows on disk: workflows(goal).")


def start_digest(data: dict[str, Any]) -> None:
    """Only put the chat in the queue: the model (~8 GB of RAM, all its threads busy) never starts on its own
    when a session ends; `kb memory digest-pending` runs the queue when the user chooses."""
    from kb.memory import digest

    if data.get("transcript_path") and data.get("session_id"):
        digest.enqueue(data["transcript_path"], data["session_id"], data.get("cwd") or "")


def hook(event: str, data: dict[str, Any]) -> dict[str, Any] | None:
    if event == "session":
        text = context()
        return {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": text}} if text else None
    if event == "end":
        start_digest(data)
    return None


def main(argv: list[str]) -> int:
    try:
        data = json.loads(sys.stdin.read() or "{}")
        out = hook(argv[0] if argv else "", data)
    except Exception as e:   # memory must never break the session
        print(f"{MARK}: {type(e).__name__}: {e}", file=sys.stderr)
        return 0
    if out:
        print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
