"""A research subagent on DeepSeek: it searches, reads pages, issues and the project's files with this server's own
tools and hands back only the distilled answer, so the calling agent (Claude Code) never pays for the raw text.

The loop is ordinary OpenAI tool calling through the bundled web proxy (kb.deepseek). What it read stays in the
local cache (search_fulltext, notebook_add_source work on it afterwards); links in the answer that no tool returned
are reported as unverified, since the calling agent cannot see what the subagent read.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from kb import deepseek

# This server's tools the subagent may call: read-only searching and reading, nothing that writes the notebook.
SERVER_TOOLS = ("research_sweep", "github_search", "hf_search", "find_papers", "fetch", "search_fulltext",
                "evidence", "recipes", "check_versions", "latest_models", "diagnose_error", "dataset_info")
RESULT_CHARS = 9000        # one tool result as the subagent sees it
CONTEXT_CHARS = 110_000    # the whole conversation; older results shrink first (the web chat has ~64k tokens)
OLD_RESULT_CHARS = 700
URL = re.compile(r"https?://[^\s)\]>'\"`]+")
# source ids as this server's tools return them (web:…, gh:…, arxiv:…): citable like URLs
SOURCE_ID = re.compile(r"\b(?:web|gh|hf|arxiv|civitai|doi|hn|issues|openalex|pack|reddit|s2|yt):[^\s)\]>'\"`,;]+")

# DeepSeek's web chat sometimes starts a call in its native tool format, which never reaches us: the proxy cuts it
# and puts a placeholder `bash` echo call in its place (meant for DeepSeek Harness, which has bash). Any reply that
# is only a call to a tool we do not have is such a lost call, not an answer.
LOST_CALL = re.compile(r'```(?:json)?\s*\{\s*"name"\s*:\s*"([^"]+)"')
RETRY = ("Your tool call did not arrive (the chat's own tool-call format does not reach the harness, and there is no "
         "bash or shell tool). Call one of the listed tools, written ONLY as a ```json block: "
         '{{"name": "<tool>", "arguments": {{...}}}}. Available: {names}.')
MAX_RETRIES = 3
IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")

SYSTEM = """\
You are a research subagent working for another AI agent (Claude Code) that has a small context budget. Do the
reading for it: search, open the promising sources (fetch pages, issues, papers, docs; read_file / grep the
project's code) until you can answer with specifics. You have at most {steps} rounds of tool calls; most
questions need 3-6: stop reading and answer as soon as the sources cover the question. Prefer primary sources
(official docs, the repository, maintainers' comments, papers) and recent ones; today is {today}.

Then answer ONLY with what the agent needs to act, at most {words} words, in the language of the task:
- the answer first, with concrete values (versions, flags, numbers, commands, code locations as path:line);
- after each fact, the source it came from (URL or source id exactly as a tool returned it) — never invent one;
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
    local = LOCAL_TOOLS if project else LOCAL_TOOLS[:1]
    return [{"type": "function", "function": s} for s in specs + local]


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
        if m["role"] == "tool" and len(content) > OLD_RESULT_CHARS:
            size -= len(content) - OLD_RESULT_CHARS
            m["content"] = content[:OLD_RESULT_CHARS] + "\n… (older result shortened)"


def _args(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("function", {}).get("arguments") or "{}"
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return {}
    return args if isinstance(args, dict) else {}


def _report(answer: str, seen: str, calls: list[str], read_chars: int, steps: int) -> dict[str, Any]:
    found = URL.findall(answer) + [m.group(0) for m in SOURCE_ID.finditer(URL.sub(" ", answer))]
    links = list(dict.fromkeys(u.rstrip(".,;:") for u in found))
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
    try:
        deepseek.ensure_running()
    except deepseek.Unavailable as e:
        return {"error": str(e), "next": "do the task yourself with this server's tools"}
    tools = tool_specs(root)
    prompt = task + (f"\n\nContext from the agent:\n{context}" if context else "")
    if root:
        prompt += f"\n\nThe project is at {root}; read_file / grep take paths inside it."
    max_steps = max(1, min(max_steps, 30))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM.format(today=datetime.date.today().isoformat(), words=words, steps=max_steps)},
        {"role": "user", "content": prompt}]
    names = sorted(t["function"]["name"] for t in tools)
    seen, calls, read_chars, retries = "", [], 0, 0
    for step in range(1, max_steps + 1):
        _fit(messages)
        last = step == max_steps
        try:
            msg = deepseek.chat(messages if not last else messages + [
                {"role": "user", "content": "Stop reading now and write the final answer from what you have."}],
                None if last else tools)
        except deepseek.Unavailable as e:
            return {"error": str(e), "tool_calls": calls, "next": "do the task yourself with this server's tools"}
        tool_calls, content = msg.get("tool_calls") or [], msg.get("content") or ""
        lost = LOST_CALL.search(content)
        if not tool_calls and lost and lost.group(1) not in names:
            if last or retries >= MAX_RETRIES:
                return {"error": "the DeepSeek subagent kept answering with tool calls in a format the proxy cannot "
                        "pass on", "tool_calls": calls, "next": "do the task yourself with this server's tools"}
            retries += 1
            messages += [{"role": "assistant", "content": content},
                         {"role": "user", "content": RETRY.format(names=", ".join(names))}]
            continue
        if not tool_calls and not content.strip():
            return {"error": "DeepSeek returned an empty reply", "tool_calls": calls,
                    "next": "do the task yourself with this server's tools (what it read is in search_fulltext)"}
        if not tool_calls:
            return _report(content, seen, calls, read_chars, step)
        retries = 0
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
        for call in tool_calls:
            name, args = call.get("function", {}).get("name", ""), _args(call)
            result = run_tool(name, args, root)
            seen += result
            read_chars += len(result)
            calls.append(f"{name}({json.dumps(args, ensure_ascii=False)[:120]})")
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "name": name, "content": result})
    return {"error": "the subagent did not finish", "tool_calls": calls}
