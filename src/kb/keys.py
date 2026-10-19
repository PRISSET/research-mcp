"""Optional API keys: one file the user fills once, read by every server process.

    ~/.config/research-mcp/keys.env      (or $RESEARCH_KEYS)
    GITHUB_TOKEN=ghp_…
    TAVILY_API_KEY=tvly-…

Values already in the environment (e.g. set in the MCP registration) win over the file. Nothing works worse without
a key; each one turns on or improves one platform, and `kb keys` shows which and where to get it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

KEYS: list[dict[str, str]] = [
    {"name": "TAVILY_API_KEY", "improves": "web search built for agents instead of scraping DuckDuckGo / Bing; free "
                                           "plan without a card", "get": "https://app.tavily.com"},
    {"name": "BRAVE_API_KEY", "improves": "web search (Brave index); needs a card, monthly free credit",
     "get": "https://api-dashboard.search.brave.com"},
    {"name": "GITHUB_TOKEN", "improves": "GitHub search and issues: 5000 requests / hour instead of 60; a token "
                                         "with no scopes is enough", "get": "https://github.com/settings/tokens"},
    {"name": "HF_TOKEN", "improves": "Hugging Face: gated models (Llama, Gemma, some FLUX…) — their cards, configs "
                                     "and weight sizes for fetch, model_variants and estimate_resources, which answer "
                                     "401 without it; higher API limits (anonymous: 500 requests / 5 min); a free "
                                     "read token", "get": "https://huggingface.co/settings/tokens"},
    {"name": "S2_API_KEY", "improves": "Semantic Scholar in paper search, citation graph and 'what others say' (off "
                                       "without a key: the shared pool answers 429 almost always); free",
     "get": "https://www.semanticscholar.org/product/api#api-key-form"},
    {"name": "OPENALEX_API_KEY", "improves": "OpenAlex paper search: higher daily limits; free",
     "get": "https://openalex.org/settings/api"},
    {"name": "CIVITAI_API_KEY", "improves": "Civitai models / LoRAs: content behind login and higher limits; free",
     "get": "https://civitai.com/user/account"},
]
NAMES = {k["name"] for k in KEYS}
_HINTED: set[str] = set()
LINE = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*(.*?)\s*$")


def path() -> Path:
    return Path(os.environ.get("RESEARCH_KEYS", "~/.config/research-mcp/keys.env")).expanduser()


def _parse(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        m = LINE.match(line)
        if m and not line.lstrip().startswith("#"):
            out[m.group(1)] = m.group(2).strip().strip("'\"")
    return out


def load() -> None:
    """Put the file's values into the environment (never overriding what is already set)."""
    try:
        values = _parse(path().read_text(encoding="utf-8"))
    except OSError:
        return
    for k, v in values.items():
        if v:
            os.environ.setdefault(k, v)


def status() -> list[dict[str, Any]]:
    return [{**k, "set": bool(os.environ.get(k["name"]))} for k in KEYS]


def missing_hint(names: list[str]) -> str | None:
    """One line for a tool result: which keys would have made this search better."""
    miss = [k for k in KEYS if k["name"] in names and not os.environ.get(k["name"]) and k["name"] not in _HINTED]
    if not miss:
        return None
    _HINTED.update(k["name"] for k in miss)   # once per server process, not in every result
    return ("Better results with optional keys (tell the user once; they add them with `uv run kb keys set NAME "
            "VALUE` in the research-mcp folder): " + "; ".join(f"{k['name']} — {k['improves']} ({k['get']})"
                                                               for k in miss))


def set_key(name: str, value: str) -> Path:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else [
        "# research-mcp keys: NAME=value, one per line. Run `uv run kb keys` to see what each one improves."]
    lines = [ln for ln in lines if not ((m := LINE.match(ln)) and m.group(1) == name)]
    if value:
        lines.append(f"{name}={value}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    p.chmod(0o600)
    return p
