"""Registration in OpenCode: `uv run kb opencode register` (the installer calls it for --agent opencode / all).

OpenCode keeps MCP servers in its global config, ~/.config/opencode/opencode.json. Version 2 puts them under
`mcp.servers` with `disabled` and a `timeout` object; version 1 put them directly under `mcp` with `enabled` and a
millisecond `timeout` for listing tools. The entry is merged into the existing file, everything else stays as it
was. A config with comments (JSONC) cannot be rewritten safely: then `opencode mcp add --global` does it.

Global rules: OpenCode reads ~/.config/opencode/AGENTS.md, or ~/.claude/CLAUDE.md when that file does not exist.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

STARTUP_MS = 120_000   # the first start may still be syncing dependencies


def config_path() -> Path:
    if os.environ.get("OPENCODE_CONFIG"):
        return Path(os.environ["OPENCODE_CONFIG"]).expanduser()
    base = Path(os.environ.get("XDG_CONFIG_HOME") or "~/.config").expanduser()
    return base / "opencode" / "opencode.json"


def rules_path() -> Path:
    """Where OpenCode reads global rules: its own AGENTS.md, else Claude Code's CLAUDE.md (its fallback)."""
    own = config_path().parent / "AGENTS.md"
    claude = Path("~/.claude/CLAUDE.md").expanduser()
    return own if own.exists() or not claude.exists() else claude


def major_version() -> int:
    try:
        out = subprocess.run(["opencode", "--version"], capture_output=True, text=True, timeout=20).stdout
        return int(re.search(r"(\d+)\.\d+", out).group(1))
    except Exception:
        return 2


def entry(directory: str, env: dict[str, str], version: int) -> dict[str, Any]:
    e: dict[str, Any] = {"type": "local", "command": ["uv", "--directory", directory, "run", "research-mcp"],
                         "environment": env}
    if version >= 2:
        e["timeout"] = {"startup": STARTUP_MS}
    else:
        e.update(enabled=True, timeout=STARTUP_MS)
    return e


def register(name: str, directory: str, env: dict[str, str]) -> str:
    version = major_version()
    path = config_path()
    try:
        cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except ValueError:   # JSONC with comments / trailing commas: let OpenCode itself edit it
        if not shutil.which("opencode"):
            raise
        flags = [x for k, v in env.items() for x in ("--env", f"{k}={v}")]
        subprocess.run(["opencode", "mcp", "add", name, "--global", *flags, "--",
                        "uv", "--directory", directory, "run", "research-mcp"], check=True)
        return f"{path} (via opencode mcp add)"
    cfg.setdefault("$schema", "https://opencode.ai/config.json")
    mcp = cfg.setdefault("mcp", {})
    if version >= 2:
        mcp.setdefault("servers", {})[name] = entry(directory, env, version)
        mcp.pop(name, None)   # a v1-style entry left from before
    else:
        mcp[name] = entry(directory, env, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_suffix(".json.bak"))
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def unregister(name: str) -> bool:
    path = config_path()
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    mcp = cfg.get("mcp") or {}
    found = mcp.pop(name, None) is not None
    found = (mcp.get("servers") or {}).pop(name, None) is not None or found
    if found:
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return found
