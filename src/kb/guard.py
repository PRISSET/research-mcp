"""Code guard: a task ledger per project and Claude Code hooks that keep the agent to it.

The ledger holds what the user asked for while coding — requirements, optimizations, constraints, facts and
decisions — outside the conversation, so compaction cannot drop them. Requirements and optimizations stay open
until they are marked done (with where: file:line) or deferred (with the reason the user hears).

Hooks (`uv run kb guard install`, written to ~/.claude/settings.json; run by Claude Code itself, so they cannot be
forgotten):
- UserPromptSubmit: the open items and standing facts go into context, with a reminder to record new ones;
- SessionStart after compaction or resume: the whole ledger again;
- Stop: finishing is refused once while ledger items are open, or while code changed since the last review_code.

Kept light: the hooks import only this module (they run on every prompt).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from kb.config import ROOT, settings

KINDS = ("requirement", "optimization", "constraint", "fact", "decision")
CLOSING = ("requirement", "optimization")     # these must end done or deferred
CODE_EXT = (".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".go", ".rs", ".java", ".kt", ".c", ".cc",
            ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".scala", ".lua", ".sh", ".sql", ".vue", ".svelte")
ARCHIVE_DAYS = 14
MARK = "kb.guard"   # how our hooks are recognised in settings.json


# ---------------- project and ledger ----------------

def _git(root: str | Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def project_root(path: str | None) -> Path:
    p = Path(path or os.environ.get("CLAUDE_PROJECT_DIR") or os.environ.get("PWD") or os.getcwd()).expanduser()
    top = _git(p, "rev-parse", "--show-toplevel")
    return Path(top.strip()) if top else p.resolve()


def _path(root: Path) -> Path:
    return settings.data_dir / "ledgers" / f"{hashlib.sha1(str(root).encode()).hexdigest()[:12]}.json"


def load(root: Path) -> dict[str, Any]:
    try:
        led = json.loads(_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        led = {}
    led.setdefault("root", str(root))
    led.setdefault("items", [])
    return led


def save(led: dict[str, Any]) -> None:
    p = _path(Path(led["root"]))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(led, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def open_items(led: dict[str, Any]) -> list[dict[str, Any]]:
    return [i for i in led["items"] if i["kind"] in CLOSING and i["status"] == "open"]


def standing(led: dict[str, Any]) -> list[dict[str, Any]]:
    return [i for i in led["items"] if i["kind"] not in CLOSING and i["status"] == "kept"]


def _line(i: dict[str, Any]) -> str:
    tail = f" — done: {i['where']}" if i.get("where") else f" — deferred: {i['why']}" if i.get("why") else ""
    return f"#{i['id']} {i['kind']}: {i['text']}{tail}"


def update(project: str | None, add: list[dict[str, str]] | None = None, done: list[dict[str, Any]] | None = None,
           defer: list[dict[str, Any]] | None = None, drop: list[int] | None = None,
           show_all: bool = False) -> dict[str, Any]:
    root = project_root(project)
    led = load(root)
    today = date.today().isoformat()
    by_id = {i["id"]: i for i in led["items"]}
    notes = []
    for a in add or []:
        kind = (a.get("kind") or "requirement").lower()
        text = " ".join(str(a.get("text") or "").split())
        if kind not in KINDS or not text:
            notes.append(f"skipped {a}: kind must be one of {', '.join(KINDS)}, text non-empty")
            continue
        if any(i["text"].lower() == text.lower() for i in led["items"] if i["status"] in ("open", "kept")):
            continue
        nid = max(by_id, default=0) + 1
        item = {"id": nid, "kind": kind, "text": text[:400], "status": "open" if kind in CLOSING else "kept",
                "created": today}
        led["items"].append(item)
        by_id[nid] = item
    for d in done or []:
        i = by_id.get(int(d.get("id", 0)))
        if not i:
            notes.append(f"no item #{d.get('id')}")
        elif not d.get("where"):
            notes.append(f"#{i['id']}: say where it is done (file:line or function)")
        else:
            i.update(status="done", where=str(d["where"])[:200], closed=today)
            i.pop("why", None)
    for d in defer or []:
        i = by_id.get(int(d.get("id", 0)))
        if not i:
            notes.append(f"no item #{d.get('id')}")
        elif not d.get("why"):
            notes.append(f"#{i['id']}: say why it is deferred — the user will read it")
        else:
            i.update(status="deferred", why=str(d["why"])[:300], closed=today)
    for nid in drop or []:
        if nid in by_id:
            by_id[nid]["status"] = "dropped"
    cutoff = (date.today() - timedelta(days=ARCHIVE_DAYS)).isoformat()
    led["items"] = [i for i in led["items"] if i["status"] != "dropped" and
                    not (i["status"] in ("done", "deferred") and (i.get("closed") or today) < cutoff)]
    save(led)
    shown = led["items"] if show_all else [i for i in led["items"] if i["status"] in ("open", "kept")
                                           or (i.get("closed") == today)]
    return {"project": str(root), "open": [_line(i) for i in open_items(led)],
            "standing": [_line(i) for i in standing(led)], "items": shown, "notes": notes,
            "next": "Work through the open items; mark each done with where (file:line) or deferred with the reason "
                    "the user should hear. Constraints, facts and decisions stay in force for the project."}


# ---------------- what changed ----------------

def _base(root: Path) -> str:
    return "HEAD" if _git(root, "rev-parse", "--verify", "-q", "HEAD") else "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def changed_code(root: Path, base: str | None = None) -> list[str]:
    """Code files changed against `base` (default HEAD) plus untracked ones, relative to the root."""
    tracked = _git(root, "diff", "--name-only", base or _base(root)) or ""
    untracked = _git(root, "ls-files", "--others", "--exclude-standard") or ""
    names = dict.fromkeys(n for n in (tracked + untracked).splitlines() if n.endswith(CODE_EXT))
    return [n for n in names if (root / n).exists()]


def fingerprint(root: Path) -> str | None:
    """Hash of the uncommitted code changes; None when there are none (or no git)."""
    if _git(root, "rev-parse", "--git-dir") is None:
        return None
    files = changed_code(root)
    if not files:
        return None
    h = hashlib.sha256((_git(root, "diff", _base(root), "--", *files) or "").encode())
    for f in files:
        try:
            h.update(f.encode() + (root / f).read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:20]


def mark_reviewed(root: Path) -> None:
    led = load(root)
    led["reviewed"] = fingerprint(root)
    save(led)


# ---------------- hooks ----------------

def _context(led: dict[str, Any], full: bool) -> str:
    op, st = open_items(led), standing(led)
    if not op and not st:
        return ("Task ledger: when the user gives requirements, optimizations, constraints or facts to keep while "
                f"coding, record them with task_ledger(project=\"{led['root']}\", add=[{{kind, text}}]).")
    lines = ["Task ledger of this project (task_ledger; keep to it):"]
    lines += [f"- OPEN {_line(i)}" for i in op]
    lines += [f"- {_line(i)}" for i in (st if full else st[-8:])]
    lines.append("Record new requirements / optimizations / facts from the user with task_ledger(add=…); mark an "
                 "item done with where it is, or deferred with why, when you finish it.")
    return "\n".join(lines)


def hook(event: str, data: dict[str, Any]) -> dict[str, Any] | None:
    root = project_root(data.get("cwd"))
    led = load(root)
    if event == "prompt":
        return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": _context(led, False)}}
    if event == "session":
        if not led["items"]:
            return None
        return {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": _context(led, True)}}
    if event == "stop":
        if data.get("stop_hook_active"):   # refused once already: the agent has been told, do not loop
            return None
        reasons = []
        op = open_items(led)
        if op:
            reasons.append("These task-ledger items are still open:\n" + "\n".join(f"- {_line(i)}" for i in op) +
                           "\nDo them now, or mark each done (with where) or deferred (with why, and tell the user) "
                           "using task_ledger.")
        fp = fingerprint(root)
        if fp and fp != led.get("reviewed"):
            reasons.append(f"Code changed since the last review: call review_code(project=\"{root}\") and fix what "
                           "applies (or say why not) before finishing.")
        if reasons:
            return {"decision": "block", "reason": "\n\n".join(reasons)}
    return None


def main(argv: list[str]) -> int:
    event = argv[0] if argv else ""
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        data = {}
    try:
        out = hook(event, data)
    except Exception as e:   # a broken guard must never break the session
        print(f"kb.guard: {type(e).__name__}: {e}", file=sys.stderr)
        return 0
    if out:
        print(json.dumps(out, ensure_ascii=False))
    return 0


# ---------------- install ----------------

def settings_path() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude").expanduser() / "settings.json"


def _command(event: str) -> str:
    return f'uv run --directory "{ROOT}" --no-sync python -m {MARK} {event}'


HOOKS = {"UserPromptSubmit": ("prompt", None), "SessionStart": ("session", "compact|resume"), "Stop": ("stop", None)}


def _without_ours(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for g in groups:
        hs = [h for h in g.get("hooks") or [] if MARK not in str(h.get("command") or "")]
        if hs:
            out.append({**g, "hooks": hs})
    return out


def install() -> str:
    path = settings_path()
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    hooks = cfg.setdefault("hooks", {})
    for name, (event, matcher) in HOOKS.items():
        groups = _without_ours(hooks.get(name) or [])
        groups.append({**({"matcher": matcher} if matcher else {}),
                       "hooks": [{"type": "command", "command": _command(event), "timeout": 20}]})
        hooks[name] = groups
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.with_suffix(f".json.bak-{datetime.now():%Y%m%d%H%M%S}").write_text(path.read_text(encoding="utf-8"),
                                                                                encoding="utf-8")
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def uninstall() -> bool:
    path = settings_path()
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    hooks = cfg.get("hooks") or {}
    found = False
    for name in list(hooks):
        kept = _without_ours(hooks[name])
        found = found or kept != hooks[name]
        if kept:
            hooks[name] = kept
        else:
            hooks.pop(name)
    if not hooks:
        cfg.pop("hooks", None)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return found


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
