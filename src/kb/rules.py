"""Install / remove the research-MCP rules block in an agent's global instructions (CLAUDE.md, AGENTS.md).

The block sits between markers, so re-running replaces it in place and `remove` takes it out cleanly.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

START, END = "<!-- research-mcp:rules -->", "<!-- /research-mcp:rules -->"
RULES = Path(__file__).with_name("assets") / "agent_rules.md"


def install(path: Path) -> str:
    block = f"{START}\n{RULES.read_text(encoding='utf-8').strip()}\n{END}"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if START in text and END in text:
        new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
        status = "updated"
    else:
        new = (text.rstrip("\n") + "\n\n" if text.strip() else "") + block + "\n"
        status = "added"
    path.write_text(new, encoding="utf-8")
    return status


def remove(path: Path) -> str:
    if not path.exists():
        return "absent"
    text = path.read_text(encoding="utf-8")
    new = re.sub(r"\n*" + re.escape(START) + r".*?" + re.escape(END) + r"\n?", "\n", text, flags=re.S)
    if new == text:
        return "absent"
    path.write_text(new.strip("\n") + "\n" if new.strip() else "", encoding="utf-8")
    return "removed"


def main() -> None:
    action, *files = sys.argv[1:]
    for f in files:
        p = Path(f).expanduser()
        print(f"{p}: {(install if action == 'install' else remove)(p)}")


if __name__ == "__main__":
    main()
