"""A local Redlib for reading Reddit, built without Docker: `uv run kb redlib install`.

Reddit closed its API to new apps and answers 403 to every non-browser client. Redlib (a private Reddit front-end)
talks to Reddit the way the official app does; a private instance on 127.0.0.1 serves the threads and Reddit
search as plain HTML that fetch and research_sweep read. Built from source with cargo; on Linux a systemd user
service keeps it running. The source is Silvenga's fork: upstream main fails at start since 2025-09 ("Failed to
create OAuth client", redlib-org/redlib#485) and its last release is from 2025-03, while the fork's build runs
(reported in #485, checked here 2026-10). KB_REDLIB_REPO / KB_REDLIB_BRANCH switch back when upstream is fixed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = os.environ.get("KB_REDLIB_REPO", "https://github.com/Silvenga/redlib")
BRANCH = os.environ.get("KB_REDLIB_BRANCH", "main")
PORT = int(os.environ.get("KB_REDLIB_PORT", "8889"))
HOME = Path(os.environ.get("KB_REDLIB_DIR", "~/.local/share/research-mcp/redlib")).expanduser()
UNIT = Path("~/.config/systemd/user/research-redlib.service").expanduser()


def _run(*cmd: str) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def url() -> str:
    return f"http://127.0.0.1:{PORT}"


def install() -> str:
    """Build (or rebuild) Redlib with cargo; start it as a service if possible."""
    if not shutil.which("cargo"):
        sys.exit("needs Rust's cargo on PATH (https://rustup.rs, or your package manager's rust)")
    HOME.mkdir(parents=True, exist_ok=True)
    _run("cargo", "install", "--locked", "--force", "--git", REPO, "--branch", BRANCH, "--root", str(HOME), "redlib")
    exe = HOME / "bin" / "redlib"
    if shutil.which("systemctl"):
        UNIT.parent.mkdir(parents=True, exist_ok=True)
        UNIT.write_text(f"""[Unit]
Description=Redlib for research-mcp (127.0.0.1:{PORT})

[Service]
Environment=REDLIB_DEFAULT_SHOW_NSFW=off
Environment=REDLIB_ROBOTS_DISABLE_INDEXING=on
ExecStart={exe} --address 127.0.0.1 --port {PORT}
Restart=on-failure

[Install]
WantedBy=default.target
""", encoding="utf-8")
        _run("systemctl", "--user", "daemon-reload")
        _run("systemctl", "--user", "enable", "--now", "research-redlib.service")
        _run("systemctl", "--user", "restart", "research-redlib.service")
        how = "systemd user service research-redlib (starts with your session)"
    else:
        how = f"start it with: {exe} --address 127.0.0.1 --port {PORT}"
    from kb import keys

    keys.set_key("REDLIB_URL", url())
    return how


def uninstall() -> None:
    if UNIT.exists():
        subprocess.run(["systemctl", "--user", "disable", "--now", "research-redlib.service"], check=False)
        UNIT.unlink()
    shutil.rmtree(HOME, ignore_errors=True)
    from kb import keys

    keys.set_key("REDLIB_URL", "")
