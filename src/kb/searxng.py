"""A local SearXNG for keyless web search, installed without Docker: `uv run kb searxng install`.

SearXNG is a metasearch engine: it asks Google, Bing, DuckDuckGo, Brave and others and merges the results, so the
built-in web search works without keys or limits. Installed from source into its own venv, listening on
127.0.0.1 only, with the JSON API on (off in SearXNG's defaults); on Linux a systemd user service keeps it running.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/searxng/searxng.git"
PORT = int(os.environ.get("KB_SEARXNG_PORT", "8888"))
HOME = Path(os.environ.get("KB_SEARXNG_DIR", "~/.local/share/research-mcp/searxng")).expanduser()
UNIT = Path("~/.config/systemd/user/research-searxng.service").expanduser()
SETTINGS = """\
use_default_settings: true
server:
  secret_key: "{secret}"
  bind_address: "127.0.0.1"
  port: {port}
  limiter: false
  public_instance: false
search:
  formats: [html, json]
"""


def _run(*cmd: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, check=True)


def url() -> str:
    return f"http://127.0.0.1:{PORT}"


def install() -> str:
    """Clone (or update) SearXNG, install it in its own venv, write settings; start it as a service if possible."""
    if not shutil.which("uv") or not shutil.which("git"):
        sys.exit("needs uv and git on PATH")
    HOME.mkdir(parents=True, exist_ok=True)
    src, venv = HOME / "src", HOME / "venv"
    if (src / ".git").exists():
        _run("git", "-C", str(src), "pull", "--ff-only", "-q")
    else:
        _run("git", "clone", "-q", "--depth", "1", REPO, str(src))
    if not venv.exists():
        _run("uv", "venv", "-q", "-p", "3.13", str(venv))   # SearXNG declares support up to 3.13
    env = {"VIRTUAL_ENV": str(venv)}
    _run("uv", "pip", "install", "-q", "setuptools", "wheel", "-r", str(src / "requirements.txt"), env=env)
    _run("uv", "pip", "install", "-q", "--no-build-isolation", "-e", str(src), env=env)
    settings = HOME / "settings.yml"
    if not settings.exists():
        settings.write_text(SETTINGS.format(secret=secrets.token_hex(24), port=PORT), encoding="utf-8")
    exe = venv / "bin" / "searxng-run"
    if shutil.which("systemctl"):
        UNIT.parent.mkdir(parents=True, exist_ok=True)
        UNIT.write_text(f"""[Unit]
Description=SearXNG for research-mcp (127.0.0.1:{PORT})

[Service]
Environment=SEARXNG_SETTINGS_PATH={settings}
ExecStart={exe}
Restart=on-failure

[Install]
WantedBy=default.target
""", encoding="utf-8")
        _run("systemctl", "--user", "daemon-reload")
        _run("systemctl", "--user", "enable", "--now", "research-searxng.service")
        how = "systemd user service research-searxng (starts with your session)"
    else:
        how = f"start it with: SEARXNG_SETTINGS_PATH={settings} {exe}"
    from kb import keys

    keys.set_key("SEARXNG_URL", url())
    return how


def uninstall() -> None:
    if UNIT.exists():
        subprocess.run(["systemctl", "--user", "disable", "--now", "research-searxng.service"], check=False)
        UNIT.unlink()
    shutil.rmtree(HOME, ignore_errors=True)
    from kb import keys

    keys.set_key("SEARXNG_URL", "")
