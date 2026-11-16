"""A research subagent on Gemini: it searches, reads pages, issues and the project's files with this server's own
tools and hands back only the distilled answer, so the calling agent (Claude Code) never pays for the raw text.

Gemini runs through the user's web reverse proxy (kb.gemini), which has no tool calling: the tools are described in
the prompt, the model writes calls as ```json blocks and the results come back as the next message. What it read
stays in the local cache (search_fulltext, notebook_add_source work on it afterwards); links in the answer that no
tool returned are reported as unverified, since the calling agent cannot see what the subagent read.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from kb import gemini

# This server's tools the subagent may call: read-only searching and reading, nothing that writes the notebook.
SERVER_TOOLS = ("research_sweep", "github_search", "hf_search", "find_papers", "fetch", "search_fulltext",
                "evidence", "recipes", "check_versions", "latest_models", "diagnose_error", "dataset_info")
RESULT_CHARS = 9000        # one tool result as the subagent sees it
CONTEXT_CHARS = 150_000    # the whole conversation; older results shrink first
OLD_RESULT_CHARS = 700
URL = re.compile(r"https?://[^\s)\]>'\"`]+")
# source ids as this server's tools return them (web:…, gh:…, arxiv:…): citable like URLs
SOURCE_ID = re.compile(r"\b(?:web|gh|hf|arxiv|civitai|doi|hn|issues|openalex|pack|reddit|s2|yt):[^\s()\[\]>'\"`,;]+")
CALL_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
RETRY = ("That was not run: use one of the listed commands ({names}) as a ```json block "
         '{{"name": "<tool>", "arguments": {{...}}}}, or write the final answer without any json block.')
MAX_RETRIES = 3
IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")

SYSTEM = """\
You are the research assistant of another AI agent (Claude Code) that has a small context budget. You do not
browse or search yourself in this conversation; I run commands on my machine for you and send you their output.
Your job: decide what to search and read (pages, issues, papers, docs; the project's code), read the outputs I
send, and stop as soon as they cover the question (most questions need 3-6 rounds, at most {steps}). Prefer
primary sources (official docs, the repository, maintainers' comments, papers) and recent ones; today is {today}.

To have commands run, reply with nothing but one or more blocks like this (several blocks run together):
```json
{{"name": "fetch", "arguments": {{"url": "https://…"}}}}
```
I send the outputs in my next message. Never write outputs yourself.

Commands:
{tools}

When the outputs cover the question, write the final answer (no json blocks), at most {words} words, in the
language of the task:
- the answer first, with concrete values (versions, flags, numbers, commands, code locations as path:line);
- after each fact, the source it came from (URL or source id exactly as an output gave it) — never invent one;
- what you could not find or verify, and disagreements between sources, in one line each.
No preamble, no restating the task, no generic advice, no copies of long passages.
"""

LOCAL_TOOLS = [
    {"name": "web_search", "description": "Search the web (SearXNG / Tavily / Brave / DuckDuckGo). Returns titles, "
     "URLs and snippets; read the pages with fetch.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["query"]}},
    {"name": "read_file", "description": "Read a file of the project (absolute, or relative to the project root), "
     "with line numbers.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer"},
                                                     "lines": {"type": "integer"}}, "required": ["path"]}},
    {"name": "grep", "description": "Search the project's files with a regular expression (git grep); returns "
     "path:line: text.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                    "required": ["pattern"]}},
]


def _server_tools() -> dict[str, Any]:
    from kb.mcp.server import mcp

    manager = mcp._tool_manager
    return {name: manager.get_tool(name) for name in SERVER_TOOLS}


def tool_specs(project: Path | None) -> list[dict[str, Any]]:
    specs = [{"name": t.name, "description": (t.description or "")[:900], "parameters": t.parameters}
             for t in _server_tools().values()]
    return specs + (LOCAL_TOOLS if project else LOCAL_TOOLS[:1])


def _describe(specs: list[dict[str, Any]]) -> str:
    """The tools as the prompt shows them: name(arguments) and what it does."""
    lines = []
    for t in specs:
        props = t["parameters"].get("properties", {})
        need = set(t["parameters"].get("required", []))
        args = ", ".join(f"{k}{'' if k in need else '?'}: {v.get('type') or 'any'}" for k, v in props.items())
        lines.append(f"- {t['name']}({args}): {' '.join(t['description'].split())}")
    return "\n".join(lines)


def parse_calls(text: str) -> list[dict[str, Any]]:
    """Tool calls written as ```json blocks: {"name", "arguments"} objects, or a list of them."""
    calls = []
    for block in CALL_BLOCK.findall(text):
        try:
            data = json.loads(block)
        except ValueError:
            continue
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                args = item.get("arguments") or item.get("parameters") or {}
                calls.append({"name": item["name"], "arguments": args if isinstance(args, dict) else {}})
    return calls


def _inside(project: Path, rel: str) -> Path:
    p = (project / rel).resolve()
    if not p.is_relative_to(project):
        raise ValueError(f"{rel} is outside the project")
    return p


def _read_file(project: Path, path: str, start: int = 1, lines: int = 200) -> str:
    text = _inside(project, path).read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, start)
    part = text[start - 1:start - 1 + min(max(lines, 1), 400)]
    return "\n".join(f"{start + i}: {line}" for i, line in enumerate(part)) + f"\n({len(text)} lines in the file)"


def _grep(project: Path, pattern: str, path: str = ".") -> str:
    where = str(_inside(project, path).relative_to(project)) or "."
    r = subprocess.run(["git", "-C", str(project), "grep", "-n", "-I", "-E", "--max-count=20", pattern, "--", where],
                       check=False, capture_output=True, text=True, timeout=30)
    return r.stdout[:RESULT_CHARS] or r.stderr[:300] or "no matches"


def run_tool(name: str, args: dict[str, Any], project: Path | None) -> str:
    """One tool call as text; errors are returned to the subagent instead of ending the run."""
    try:
        if name == "web_search":
            from kb.research.web import web_search
            from kb.store import Database

            db = Database()
            try:
                out: Any = web_search(db, str(args.get("query", "")), int(args.get("limit") or 8))
            finally:
                db.close()
        elif name in ("read_file", "grep") and project:
            fn = _read_file if name == "read_file" else _grep
            out = fn(project, **{k: v for k, v in args.items() if k in fn.__code__.co_varnames})
        elif name in SERVER_TOOLS:
            tool = _server_tools()[name]
            out = tool.fn(**{k: v for k, v in args.items() if k in tool.parameters.get("properties", {})})
        else:
            return f"unknown tool {name}"
    except Exception as e:
        return f"error: {type(e).__name__}: {str(e)[:300]}"
    text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)
    text = IMAGE.sub(lambda m: f"[image: {m.group(1)}]", text)   # README badges and pictures are noise here
    return text if len(text) <= RESULT_CHARS else text[:RESULT_CHARS] + f"\n… cut ({len(text)} chars)"


def _fit(messages: list[dict[str, Any]]) -> None:
    """Keep the conversation inside the web chat's window: shrink the oldest tool results first."""
    size = sum(len(str(m.get("content") or "")) for m in messages)
    for m in messages:
        if size <= CONTEXT_CHARS:
            return
        content = m.get("content") or ""
        if m.get("result") and len(content) > OLD_RESULT_CHARS:
            size -= len(content) - OLD_RESULT_CHARS
            m["content"] = content[:OLD_RESULT_CHARS] + "\n… (older result shortened)"


def _links(text: str) -> list[str]:
    """URLs and source ids (web:…, gh:…) in a text, in order, without trailing punctuation."""
    found = URL.findall(text) + [m.group(0) for m in SOURCE_ID.finditer(URL.sub(" ", text))]
    return list(dict.fromkeys(u.rstrip(".,;:") for u in found))


def _report(answer: str, seen: set[str], calls: list[str], read_chars: int, steps: int) -> dict[str, Any]:
    links = _links(answer)
    unverified = [u for u in links if u not in seen]
    out: dict[str, Any] = {"answer": answer.strip(), "sources": [u for u in links if u in seen], "steps": steps,
                           "tool_calls": calls, "read_chars": read_chars}
    if unverified:
        out["unverified_links"] = unverified
        out["note"] = "no tool returned these links: check them before relying on them"
    return out


def delegate(task: str, context: str | None = None, project: str | None = None, max_steps: int = 12,
             words: int = 400) -> dict[str, Any]:
    """Run the subagent on `task` and return its distilled answer with the sources it actually read."""
    import datetime

    root = Path(project).expanduser().resolve() if project else None
    if root and not root.is_dir():
        return {"error": f"project {project} is not a directory"}
    specs = tool_specs(root)
    names = sorted(t["name"] for t in specs)
    prompt = task + (f"\n\nContext from the agent:\n{context}" if context else "")
    if root:
        prompt += f"\n\nThe project is at {root}; read_file / grep take paths inside it."
    prompt += "\n\nStart now: reply with only the ```json command blocks for your first searches or reads."
    max_steps = max(1, min(max_steps, 30))
    system = SYSTEM.format(today=datetime.date.today().isoformat(), words=words, steps=max_steps,
                           tools=_describe(specs))
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    seen: set[str] = set()
    done: list[str] = []
    read_chars = retries = 0
    for step in range(1, max_steps + 1):
        _fit(messages)
        last = step == max_steps
        ask = messages + [{"role": "user", "content": "Stop reading now and write the final answer from the outputs "
                                                      "you have, without json blocks."}] if last else messages
        try:
            reply = gemini.chat([{"role": m["role"], "content": m["content"]} for m in ask])
        except gemini.Unavailable as e:
            return {"error": str(e), "tool_calls": done, "next": "do the task yourself with this server's tools"}
        calls = parse_calls(reply)
        if last and calls:
            return {"error": "the Gemini subagent ran out of steps without writing the answer", "tool_calls": done,
                    "next": "do the task yourself with this server's tools (what it read is in search_fulltext)"}
        unknown = [c["name"] for c in calls if c["name"] not in names]
        if unknown or (not calls and CALL_BLOCK.search(reply) and '"name"' in reply and not last):
            if retries >= MAX_RETRIES:
                return {"error": "the Gemini subagent kept writing tool calls that cannot run", "tool_calls": done,
                        "next": "do the task yourself with this server's tools"}
            retries += 1
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": RETRY.format(names=", ".join(names))}]
            continue
        if not calls:
            return _report(reply, seen, done, read_chars, step)
        retries = 0
        results = []
        for call in calls:
            result = run_tool(call["name"], call["arguments"], root)
            seen.update(_links(result))
            read_chars += len(result)
            label = f"{call['name']}({json.dumps(call['arguments'], ensure_ascii=False)[:120]})"
            done.append(label)
            results.append(f"### {label}\n```\n{result}\n```")
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "Outputs:\n\n" + "\n\n".join(results), "result": True}]
    return {"error": "the subagent did not finish", "tool_calls": done}
