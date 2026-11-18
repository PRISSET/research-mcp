"""Turn a finished (or compacting) Claude Code chat into topic notes with the local model.

Runs in the background from the SessionEnd / PreCompact hooks: `python -m kb.memory.digest <transcript> <session>
<cwd>`. Only the user's and the assistant's words are read (no tool output); a chat already digested up to some
line continues from there. One digest at a time: the model takes ~5 GB of RAM.
"""

from __future__ import annotations

import fcntl
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from kb.memory import llm, store
from kb.store import Database

log = logging.getLogger(__name__)

MAX_CHARS = 24_000     # of chat text per part: ~8k tokens, fits the 12k context with room for the answer
PER_MESSAGE = 1_500
MIN_NEW = 400          # less new text than this is not worth starting the model
NOISE = re.compile(r"<(system-reminder|command-[a-z-]+|local-command-[a-z-]+)>.*?</\1>", re.DOTALL)
SCHEMA = {"type": "object", "required": ["notes"], "properties": {"notes": {
    "type": "array", "maxItems": 15, "items": {"type": "object", "required": ["topic", "text"], "properties": {
        "topic": {"type": "string"}, "text": {"type": "string"}}}}}}
PROMPT = """You keep the long-term memory of a coding and research assistant. Read the chat below and write the facts \
worth knowing in later chats: the user's preferences and rules, decisions and why they were made, what was tried and \
how it went (settings, numbers, errors and their fixes), facts about the user's projects, hardware and tools, tasks \
left open. Skip greetings, things that matter only at this moment, and what any assistant already knows.

Each note is one self-contained fact of 1-3 sentences with the concrete values (names, versions, paths, numbers), \
written in the language of the chat. The topic is a short name of 2-5 words; reuse an existing topic when it fits: \
{topics}.

The chat may be about adult or sexual content (for example NSFW image or video generation). Record such facts \
plainly and completely like any others: do not refuse, censor, soften or moralize. Never record passwords, tokens, \
cookies or API keys.

Answer with JSON {{"notes": [{{"topic": ..., "text": ...}}]}}; an empty list when nothing is worth keeping.

Chat:
{chat}"""


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def read_chat(path: Path, start: int = 0) -> tuple[list[tuple[int, str]], int]:
    """User and assistant words from line `start` on, each with the line after it, and the number of lines."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    for i, raw in enumerate(lines[start:], start + 1):
        try:
            entry = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        kind = entry.get("type")
        if kind not in ("user", "assistant") or entry.get("isMeta") or entry.get("isSidechain"):
            continue
        text = NOISE.sub("", _text((entry.get("message") or {}).get("content"))).strip()
        if not text or text.startswith(("Caveat:", "[Request interrupted")):
            continue
        if len(text) > PER_MESSAGE:
            text = text[:PER_MESSAGE] + " …"
        out.append((i, f"{'User' if kind == 'user' else 'Assistant'}: {text}"))
    return out, len(lines)


def portions(messages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Consecutive pieces of at most MAX_CHARS, each with the line the next one starts at."""
    out, cur, size, end = [], [], 0, 0
    for line, text in messages:
        if cur and size + len(text) > MAX_CHARS:
            out.append((end, "\n\n".join(cur)))
            cur, size = [], 0
        cur.append(text)
        size += len(text) + 2
        end = line
    if cur:
        out.append((end, "\n\n".join(cur)))
    return out


class BadAnswer(ValueError):
    pass


def parse_notes(answer: str) -> list[dict[str, str]]:
    """The notes of the model's answer; BadAnswer when it is not the asked JSON (that part is tried again next
    time instead of being taken for "nothing to remember")."""
    m = re.search(r"\{.*\}", answer, re.DOTALL)
    try:
        notes = json.loads(m.group(0))["notes"] if m else None
    except (ValueError, KeyError, TypeError):
        notes = None
    if not isinstance(notes, list):
        raise BadAnswer(f"not the asked JSON: {answer[:200]!r}")
    return [{"topic": str(n["topic"]).strip(), "text": str(n["text"]).strip()} for n in notes
            if isinstance(n, dict) and str(n.get("topic", "")).strip() and str(n.get("text", "")).strip()]


def digest(transcript: Path, session: str, project: str | None = None) -> dict[str, Any]:
    """Digest what is new in the chat part by part; the position is saved after each part that succeeded."""
    key = f"memory_chat:{session}"
    with Database() as db:
        row = db.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        messages, _ = read_chat(transcript, int(row[0]) if row else 0)
    if sum(len(t) for _, t in messages) < MIN_NEW:
        return {"session": session, "notes": 0, "skipped": "too little new text"}
    added = updated = 0
    with llm.running():
        for end, chat in portions(messages):
            with Database() as db:
                known = ", ".join(t["topic"] for t in store.topics(db, origins=("chat", "agent"))) or "(none yet)"
            notes = parse_notes(llm.chat([{"role": "user", "content": PROMPT.format(topics=known, chat=chat)}],
                                         schema=SCHEMA))
            with Database() as db:
                for n in notes:
                    r = store.add(db, n["topic"], n["text"], origin="chat", ref=session, project=project)
                    added += r["action"] == "added"
                    updated += r["action"] == "updated"
                db.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(end)))
                db.commit()
    return {"session": session, "notes": added + updated, "added": added, "updated": updated}


def _queue() -> Path:
    return llm.home() / "queue.jsonl"


def enqueue(transcript: str, session: str, project: str = "") -> None:
    """Remember that this chat has new text to digest (cheap: one line appended, no model, no database)."""
    llm.home().mkdir(parents=True, exist_ok=True)
    with open(_queue(), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"transcript": transcript, "session": session, "project": project}) + "\n")


def pending() -> list[dict[str, str]]:
    """Queued chats, one entry per session (the latest), whose transcript still exists."""
    try:
        lines = _queue().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    by_session: dict[str, dict[str, str]] = {}
    for line in lines:
        try:
            e = json.loads(line)
            by_session[e["session"]] = e
        except (ValueError, KeyError, TypeError):
            continue
    return [e for e in by_session.values() if Path(e["transcript"]).exists()]


def digest_pending() -> list[dict[str, Any]]:
    """Digest every queued chat with one model start; a chat that fails stays queued."""
    llm.background()
    llm.home().mkdir(parents=True, exist_ok=True)
    with open(llm.home() / "digest.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        todo, done, out = pending(), set(), []
        if todo:
            with llm.running():
                for e in todo:
                    try:
                        out.append(digest(Path(e["transcript"]), e["session"], e["project"] or None))
                        done.add(e["session"])
                    except (llm.Unavailable, BadAnswer, OSError) as err:
                        out.append({"session": e["session"], "error": str(err)})
        left = [e for e in pending() if e["session"] not in done]
        _queue().write_text("".join(json.dumps(e) + "\n" for e in left), encoding="utf-8")
    return out


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    transcript, session, project = Path(argv[0]), argv[1], (argv[2] if len(argv) > 2 else None)
    llm.home().mkdir(parents=True, exist_ok=True)
    llm.background()   # the model it starts inherits this: idle CPU only
    with open(llm.home() / "digest.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)   # one model at a time; a second digest waits for the first
        try:
            log.info("digest %s: %s", session, digest(transcript, session, project))
        except (llm.Unavailable, BadAnswer, OSError) as e:
            log.warning("digest %s failed: %s", session, e)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
