"""Claude Code hooks of the memory: what earlier chats left (topics only) at session start, a nudge to save
lasting facts with remember() after a substantial stretch of conversation, and the chat queued for the local
model before compaction and at session end."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from kb.config import ROOT

MARK = "kb.memory.hook"   # how our hooks are recognised in settings.json
HOOKS = {"SessionStart": ("session", None), "PreCompact": ("end", None), "SessionEnd": ("end", None),
         "Stop": ("stop", None)}
REMIND_CHARS = 8_000   # of new conversation text before Claude is asked again to save what lasts
REMIND = ("Memory: if this stretch of the conversation produced lasting facts (a preference or rule of the user, a "
          "decision and why, a setting or result that worked or failed, where something is), save each with "
          "remember(topic, text) — at most 3, short, with the concrete values. Nothing worth keeping: just stop.")


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
        ts = store.topics(db, limit=15)
    if not ts:
        return None
    listed = "; ".join(f"{t['topic']} ({t['notes']})" for t in ts)
    return f"Memory topics (recall(query) gives the notes): {listed}."


def start_digest(data: dict[str, Any]) -> None:
    """Only put the chat in the queue: the model never starts on its own when a session ends (closing the editor
    ends many at once); `kb memory digest-pending` runs the queue when the user chooses."""
    from kb.memory import digest

    if data.get("transcript_path") and data.get("session_id"):
        digest.enqueue(data["transcript_path"], data["session_id"], data.get("cwd") or "")


def remind(data: dict[str, Any]) -> dict[str, Any] | None:
    """Ask Claude, once per REMIND_CHARS of new user/assistant text, to save what should outlast the chat: it
    knows best what mattered, and it costs a short extra turn instead of a model run."""
    from kb.memory import digest, llm

    transcript, session = data.get("transcript_path"), data.get("session_id")
    if data.get("stop_hook_active") or not transcript or not session:
        return None
    state_file = llm.home() / "remind.json"
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    messages, n_lines = digest.read_chat(Path(transcript), int(state.get(session, 0)))
    if sum(len(t) for _, t in messages) < REMIND_CHARS:
        return None
    state[session] = n_lines
    llm.home().mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state), encoding="utf-8")
    return {"decision": "block", "reason": REMIND}


def hook(event: str, data: dict[str, Any]) -> dict[str, Any] | None:
    if event == "session":
        text = context()
        return {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": text}} if text else None
    if event == "end":
        start_digest(data)
    if event == "stop":
        return remind(data)
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
