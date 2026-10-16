"""Self-update on start: every time the agent (Claude Code, Codex) launches the server, fetch the repository and
fast-forward to the published version, sync dependencies if they changed, and restart on the new code.

Safe by construction: only a clean checkout on a branch that tracks a remote, only fast-forward (local commits or
edits are never touched), a short network timeout (offline start is not delayed), one updater at a time when
several sessions start together. Nothing goes to stdout — that is the MCP channel. KB_AUTO_UPDATE=0 turns it off.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from importlib.util import find_spec
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
FETCH_TIMEOUT = float(os.environ.get("KB_UPDATE_TIMEOUT", "8"))
DEP_FILES = ("pyproject.toml", "uv.lock")
# The published history was rewritten on 2026-10-15 (squash) and 2026-10-16 (commit messages). Installs still on
# one of these commits are moved to the new history (only when they have no local changes); anything else that is
# not an ancestor counts as the user's own work.
REWRITTEN = {"726f6e4", "ae887a6", "bb4a69e", "8ee6676", "8292257", "e66bbaa", "682d1c4", "dbb7e9d", "08d1e73",
             "02e864f", "6a20dbe", "e17fe76", "e1bdafb"}


def _git(*args: str, timeout: float = 15) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "true", "LC_ALL": "C"}
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=timeout, env=env,
                       stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(f"git {args[0]}: {r.stderr.strip()[:200]}")
    return r.stdout.strip()


def _deps_hash() -> str:
    h = hashlib.sha256()
    for name in DEP_FILES:
        f = ROOT / name
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()


def _sync() -> str:
    uv = os.environ.get("UV") or shutil.which("uv")
    if not uv:
        return "dependencies changed but uv was not found: run `uv sync` in " + str(ROOT)
    cmd = [uv, "--directory", str(ROOT), "sync", "--inexact", "--extra", "fulltext"]
    if find_spec("model2vec") is not None or find_spec("torch") is not None:   # keep semantic search if installed
        cmd += ["--extra", "embed"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    return "dependencies synced" if r.returncode == 0 else f"uv sync failed: {r.stderr.strip()[-300:]}"


def _lock() -> object | None:
    """One updater at a time: Claude Code and Codex sessions may all start at once."""
    try:
        import fcntl
    except ImportError:          # Windows: no flock, just go ahead
        return object()
    from kb.config import settings

    data = Path(os.environ.get("KB_DATA_DIR") or settings.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    f = open(data / "update.lock", "w")  # noqa: SIM115 — closed by check() when it is done
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def check(apply: bool = True) -> dict[str, str]:
    """Fetch and, when possible, fast-forward. Returns what happened (never raises)."""
    if not (ROOT / ".git").exists():
        return {"status": "skipped: not a git checkout"}
    lock = _lock()
    if lock is None:
        return {"status": "skipped: another session is updating"}
    try:
        return _check(apply)
    finally:
        close = getattr(lock, "close", None)
        if close:
            close()


def _check(apply: bool) -> dict[str, str]:
    try:
        try:
            upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        except RuntimeError:
            return {"status": "skipped: branch does not track a remote"}
        remote, _, branch = upstream.partition("/")
        _git("fetch", "--quiet", remote, branch, timeout=FETCH_TIMEOUT)
        head, new = _git("rev-parse", "HEAD"), _git("rev-parse", "@{u}")
        if head == new:
            return {"status": "up to date", "version": head[:7]}
        rewritten = head[:7] in REWRITTEN
        if not rewritten:
            try:
                _git("merge-base", "--is-ancestor", "HEAD", "@{u}")
            except RuntimeError:
                return {"status": "skipped: local commits not in the published version", "version": head[:7]}
        if _git("status", "--porcelain", "--untracked-files=no"):
            return {"status": "skipped: local changes in the checkout", "version": head[:7]}
        changes = "history rewritten upstream" if rewritten else _git("log", "--format=%s", f"{head}..{new}")
        if not apply:
            return {"status": "update available", "version": head[:7], "latest": new[:7], "changes": changes}
        deps = _deps_hash()
        _git("reset", "--hard", "--quiet", "@{u}") if rewritten else _git("merge", "--ff-only", "--quiet", "@{u}", timeout=60)
        out = {"status": "updated", "from": head[:7], "to": new[:7], "changes": changes}
        if _deps_hash() != deps:
            out["dependencies"] = _sync()
        _record(out)
        return out
    except subprocess.TimeoutExpired:
        return {"status": "skipped: network timeout"}
    except Exception as e:
        return {"status": f"skipped: {type(e).__name__}: {str(e)[:200]}"}


def _record(info: dict[str, str]) -> None:
    try:
        from kb.config import settings

        (Path(os.environ.get("KB_DATA_DIR") or settings.data_dir) / "last_update.json").write_text(json.dumps({**info, "at": time.strftime("%Y-%m-%d %H:%M")},
                                                                        ensure_ascii=False, indent=1))
    except OSError:
        pass


def update_and_restart() -> None:
    """Called first thing in the server's main(): after an update, re-exec so the new code is what runs."""
    if os.environ.get("KB_AUTO_UPDATE", "1") == "0" or os.environ.get("KB_UPDATED"):
        return
    info = check()
    if info["status"] != "updated":
        if not info["status"].startswith(("up to date", "skipped: not a git")):
            log.warning("research-mcp update: %s", info["status"])
        return
    os.environ["KB_UPDATED"] = json.dumps(info, ensure_ascii=False)
    sys.stderr.write(f"research-mcp updated {info['from']} -> {info['to']}, restarting\n")
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, *sys.argv])


def update_note() -> str:
    """For the server instructions: tell the agent once what changed in this session's update."""
    raw = os.environ.get("KB_UPDATED")
    if not raw:
        return ""
    try:
        info = json.loads(raw)
    except ValueError:
        return ""
    lines = "; ".join(info.get("changes", "").splitlines()[:8])
    return f"\nThis server was just updated ({info.get('from')} -> {info.get('to')}): {lines}\n"
