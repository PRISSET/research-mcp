"""RESEARCH.md: one Markdown file that collects every research topic, its papers and conclusions.

The file is meant to be read and edited by people. The server only touches what sits inside its HTML-comment
markers (invisible when rendered) and leaves everything else alone:

    <!-- research:toc --> … <!-- /research:toc -->                       table of contents, regenerated
    <!-- research:section id="…" --> … <!-- /research:section -->         one topic
      <!-- research:header {json} --> … <!-- /research:header -->        question, queries, dates (regenerated)
      <!-- research:summary --> … <!-- /research:summary -->             conclusions, written by the agent
      <!-- research:playbook --> … <!-- /research:playbook -->           what a specialist knows: concepts, decisions,
                                                                          practices, pitfalls — each with its sources
      <!-- research:experiments --> … <!-- /research:experiments -->     what was tried and what came of it (a table)
      <!-- research:paper id="…" --> … <!-- /research:paper -->          one paper: generated title line + notes

Writes are atomic (temp file + rename) and serialised with a lock file, so two agents cannot interleave.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

LABELS = {
    "ru": {"title": "Исследования", "intro": "Статьи и выводы по темам. Файл ведёт research MCP; правьте свободно — "
                                             "служебные метки `<!-- research:… -->` лучше не удалять.",
           "toc_head": "| Тема | Вопрос | Статей | Обновлено |", "question": "Вопрос", "queries": "Поисковые запросы",
           "created": "Создано", "updated": "Обновлено", "summary": "Выводы", "papers": "Источники",
           "no_summary": "_Выводов пока нет._", "grade": "оценка", "citations": "цит.", "empty": "_Пока пусто._",
           "kinds": {"paper": "статья", "repo": "репозиторий", "issue": "issue", "pr": "pull request", "model": "модель HF",
                     "dataset": "датасет HF", "space": "Space HF", "web": "страница", "hn": "Hacker News",
                     "so": "Stack Overflow", "civitai": "модель Civitai", "node": "нода ComfyUI",
                     "discussion": "обсуждение HF"},
           "stars": "★", "downloads": "загрузок", "pushed": "коммит", "upd": "обновлено", "state": "статус",
           "merged": "влит", "playbook": "Знания специалиста", "sources": "источники",
           "experiments": "Эксперименты", "exp_head": "| Дата | Что меняли | Настройки | Результат | Итог |",
           "verdicts": {"better": "лучше", "worse": "хуже", "same": "без разницы", "failed": "не сработало",
                        "unclear": "непонятно"},
           "knowledge": {"concept": "Понятия", "decision": "Решения и компромиссы", "practice": "Практики",
                         "pitfall": "Ошибки и ловушки", "metric": "Метрики и контроль качества",
                         "tool": "Инструменты"}},
    "en": {"title": "Research", "intro": "Papers and conclusions by topic. Maintained by the research MCP; edit freely — "
                                         "keep the `<!-- research:… -->` markers.",
           "toc_head": "| Topic | Question | Papers | Updated |", "question": "Question", "queries": "Search queries",
           "created": "Created", "updated": "Updated", "summary": "Conclusions", "papers": "Sources",
           "no_summary": "_No conclusions yet._", "grade": "grade", "citations": "citations", "empty": "_Nothing yet._",
           "kinds": {"paper": "paper", "repo": "repo", "issue": "issue", "pr": "pull request", "model": "HF model",
                     "dataset": "HF dataset", "space": "HF space", "web": "page", "hn": "Hacker News",
                     "so": "Stack Overflow", "civitai": "Civitai model", "node": "ComfyUI node",
                     "discussion": "HF discussion"},
           "stars": "★", "downloads": "downloads", "pushed": "last commit", "upd": "updated", "state": "state",
           "merged": "merged", "playbook": "Specialist knowledge", "sources": "sources",
           "experiments": "Experiments", "exp_head": "| Date | Change | Settings | Result | Verdict |",
           "verdicts": {"better": "better", "worse": "worse", "same": "no difference", "failed": "failed",
                        "unclear": "unclear"},
           "knowledge": {"concept": "Concepts", "decision": "Decisions and trade-offs", "practice": "Practices",
                         "pitfall": "Pitfalls", "metric": "Metrics and quality control", "tool": "Tools"}},
}
KNOWLEDGE_KINDS = ("concept", "decision", "practice", "pitfall", "metric", "tool")

SECTION_RE = re.compile(r'<!-- research:section id="(?P<id>[^"]+)" -->(?P<body>.*?)<!-- /research:section -->', re.DOTALL)
HEADER_RE = re.compile(r"<!-- research:header (?P<meta>\{.*?\}) -->(?P<body>.*?)<!-- /research:header -->", re.DOTALL)
SUMMARY_RE = re.compile(r"<!-- research:summary -->(?P<body>.*?)<!-- /research:summary -->", re.DOTALL)
PAPER_RE = re.compile(r'<!-- research:paper id="(?P<id>[^"]+)" -->(?P<body>.*?)<!-- /research:paper -->', re.DOTALL)
PLAYBOOK_RE = re.compile(r"<!-- research:playbook -->(?P<body>.*?)<!-- /research:playbook -->", re.DOTALL)
EXPERIMENTS_RE = re.compile(r"<!-- research:experiments -->(?P<body>.*?)<!-- /research:experiments -->", re.DOTALL)
TOC_RE = re.compile(r"<!-- research:toc -->.*?<!-- /research:toc -->", re.DOTALL)
TITLE_RE = re.compile(r"^## (.+)$", re.MULTILINE)


class NotebookError(ValueError):
    pass


def notebook_path() -> Path:
    return Path(os.environ.get("RESEARCH_NOTEBOOK", "~/research/RESEARCH.md")).expanduser()


def labels() -> dict[str, str]:
    return LABELS.get(os.environ.get("RESEARCH_LANG", "ru"), LABELS["ru"])


def slug(title: str) -> str:
    """GitHub-style heading anchor (keeps Cyrillic), also used as the section id."""
    s = re.sub(r"[^\w\- ]", "", title.strip().lower())
    return re.sub(r" ", "-", s) or "topic"


def _today() -> str:
    return date.today().isoformat()


def _meta_comment(meta: dict[str, Any]) -> str:
    # "-->" inside the JSON would end the comment early.
    return json.dumps(meta, ensure_ascii=False).replace(">", "\\u003e")


# ---------------- file access ----------------

@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    with open(lock, "w") as fh:
        try:
            import fcntl

            fcntl.flock(fh, fcntl.LOCK_EX)
        except ImportError:  # Windows: no advisory locks, single-writer assumed
            pass
        yield


def _read(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    L = labels()
    return f"# {L['title']}\n\n{L['intro']}\n\n<!-- research:toc -->\n<!-- /research:toc -->\n"


def _write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".research-", suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text.rstrip("\n") + "\n")
    os.replace(tmp, path)


# ---------------- parsing ----------------

def _sections(text: str) -> list[dict[str, Any]]:
    out = []
    for m in SECTION_RE.finditer(text):
        body = m.group("body")
        h = HEADER_RE.search(body)
        meta = json.loads(h.group("meta")) if h else {}
        t = TITLE_RE.search(body)
        s = SUMMARY_RE.search(body)
        out.append({
            "id": m.group("id"), "title": t.group(1).strip() if t else m.group("id"),
            "question": meta.get("question", ""), "queries": meta.get("queries", []),
            "created": meta.get("created"), "updated": meta.get("updated"),
            "papers": [p.group("id") for p in PAPER_RE.finditer(body)],
            "summary": _summary_text(s.group("body") if s else ""), "span": m.span(), "body": body,
            "knowledge": _parse_playbook(body), "experiments": _parse_experiments(body),
        })
    return out


def _parse_playbook(body: str) -> dict[str, list[dict[str, Any]]]:
    """Items under each kind heading: "- text — sources: `id`, `id`"."""
    m = PLAYBOOK_RE.search(body)
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in KNOWLEDGE_KINDS}
    if not m:
        return out
    by_heading = {v.lower(): k for L in LABELS.values() for k, v in L["knowledge"].items()}
    kind = None
    for line in m.group("body").splitlines():
        if line.startswith("#### "):
            kind = by_heading.get(line[5:].strip().lower())
        elif kind and line.startswith("- "):
            text, _, tail = line[2:].rpartition(" — ")
            ids = re.findall(r"`([^`]+)`", tail)
            out[kind].append({"text": (text if ids else line[2:]).strip(), "sources": ids})
    return out


def _parse_experiments(body: str) -> list[dict[str, str]]:
    m = EXPERIMENTS_RE.search(body)
    if not m:
        return []
    verdict = {v: k for L in LABELS.values() for k, v in L["verdicts"].items()}
    out = []
    for line in m.group("body").splitlines():
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
        if len(cells) == 5 and not set(cells[0]) <= set("-: ") and re.match(r"\d{4}-\d\d-\d\d", cells[0]):
            out.append({"date": cells[0], "change": cells[1], "settings": cells[2], "result": cells[3],
                        "verdict": verdict.get(cells[4].strip("*").lower(), cells[4])})
    return out


def _cell(text: str) -> str:
    return " ".join(str(text or "—").split()).replace("|", "\\|")


def _playbook_block(items: dict[str, list[dict[str, Any]]]) -> str:
    L = labels()
    parts = [f"### {L['playbook']}", "<!-- research:playbook -->"]
    for kind in KNOWLEDGE_KINDS:
        if items.get(kind):
            parts.append(f"#### {L['knowledge'][kind]}")
            parts += [f"- {it['text']} — {L['sources']}: " + ", ".join(f"`{i}`" for i in it["sources"])
                      for it in items[kind]]
    parts.append("<!-- /research:playbook -->")
    return "\n".join(parts)


def _summary_text(raw: str) -> str:
    text = raw.strip()
    return "" if text in {L["no_summary"] for L in LABELS.values()} else text


def _find(text: str, section: str) -> dict[str, Any]:
    secs = _sections(text)
    for s in secs:
        if section in (s["id"], s["title"]) or slug(section) == s["id"] or section.lower() == s["title"].lower():
            return s
    known = ", ".join(f'"{s["id"]}"' for s in secs) or "none yet"
    raise NotebookError(f"no section {section!r}; existing: {known}. Create one with notebook_write(title=…, question=…).")


# ---------------- rendering ----------------

def _header(meta: dict[str, Any]) -> str:
    L = labels()
    lines = [f"**{L['question']}:** {meta.get('question') or '—'}  "]
    if meta.get("queries"):
        lines.append(f"**{L['queries']}:** " + ", ".join(f"`{q}`" for q in meta["queries"]) + "  ")
    lines.append(f"**{L['created']}:** {meta.get('created')} · **{L['updated']}:** {meta.get('updated')}")
    return f"<!-- research:header {_meta_comment(meta)} -->\n" + "\n".join(lines) + "\n<!-- /research:header -->"


def _toc(text: str) -> str:
    L = labels()
    rows = [L["toc_head"], "|---|---|---|---|"]
    for s in sorted(_sections(text), key=lambda s: s["updated"] or "", reverse=True):
        q = (s["question"] or "").replace("|", "\\|").replace("\n", " ")
        rows.append(f"| [{s['title']}](#{slug(s['title'])}) | {q[:120]} | {len(s['papers'])} | {s['updated']} |")
    body = "\n".join(rows) if len(rows) > 2 else L["empty"]
    return f"<!-- research:toc -->\n{body}\n<!-- /research:toc -->"


def _with_toc(text: str) -> str:
    if TOC_RE.search(text):
        return TOC_RE.sub(lambda _: _toc(text), text, count=1)
    # Markers removed by hand: put the TOC back right after the first heading.
    m = re.search(r"^# .+$", text, re.MULTILINE)
    at = m.end() if m else 0
    return text[:at] + "\n\n" + _toc(text) + "\n" + text[at:]


def paper_block(src: dict[str, Any], notes: str) -> str:
    """Title line generated from the stored source (any kind), then the agent's notes."""
    L = labels()
    kind = src.get("kind") or "paper"
    extra = src.get("extra") or {}
    bits = [] if kind == "paper" else [L["kinds"].get(kind, kind)]
    bits.append(str(src.get("year") or ""))
    if src.get("venue"):
        bits.append(src["venue"])
    if src.get("grade"):
        bits.append(f"{L['grade']} **{src['grade']}**")
    if kind == "paper" and src.get("citation_count") is not None:
        bits.append(f"{src['citation_count']} {L['citations']}")
    if extra.get("stars") is not None:
        bits.append(f"{L['stars']} {extra['stars']}")
    if extra.get("downloads"):
        bits.append(f"{extra['downloads']:,} {L['downloads']}".replace(",", " "))
    if extra.get("license"):
        bits.append(str(extra["license"]))
    if extra.get("pushed"):
        bits.append(f"{L['pushed']} {extra['pushed']}")
    elif extra.get("updated") and kind != "paper":
        bits.append(f"{L['upd']} {extra['updated']}")
    if kind in ("issue", "pr") and extra.get("state"):
        bits.append(f"{L['state']}: {extra['state']}" + (f", {L['merged']}" if extra.get("merged") else ""))
    bits.append(f"`{src['id']}`")
    title = src["title"].replace("[", "(").replace("]", ")")
    link = f"[{title}]({src['url']})" if src.get("url") else title
    authors = src.get("authors") or []
    who = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "") if kind == "paper" else ""
    head = f"#### {link}\n" + (f"{who}  \n" if who else "") + " · ".join(b for b in bits if b)
    return f'<!-- research:paper id="{src["id"]}" -->\n{head}\n\n{notes.strip()}\n<!-- /research:paper -->'


def _touch(body: str) -> str:
    h = HEADER_RE.search(body)
    if not h:
        return body
    meta = json.loads(h.group("meta"))
    meta["updated"] = _today()
    return body[: h.start()] + _header(meta) + body[h.end():]


def _replace_section(text: str, sec: dict[str, Any], body: str) -> str:
    a, b = sec["span"]
    return text[:a] + f'<!-- research:section id="{sec["id"]}" -->' + body + "<!-- /research:section -->" + text[b:]


# ---------------- public API ----------------

def overview() -> dict[str, Any]:
    path = notebook_path()
    text = _read(path) if path.exists() else ""
    return {"path": str(path), "exists": path.exists(), "sections": [
        {k: s[k] for k in ("id", "title", "question", "updated")} | {"papers": len(s["papers"]),
                                                                     "has_summary": bool(s["summary"])}
        for s in _sections(text)]}


def read_section(section: str) -> dict[str, Any]:
    text = _read(notebook_path())
    s = _find(text, section)
    return {k: s[k] for k in ("id", "title", "question", "queries", "created", "updated", "papers")} | {
        "knowledge_counts": {k: len(v) for k, v in s["knowledge"].items()}, "experiments": s["experiments"],
        "markdown": s["body"].strip()}


def knowledge(section: str) -> dict[str, Any]:
    s = _find(_read(notebook_path()), section)
    return {"section": s["id"], "papers": s["papers"], "knowledge": s["knowledge"]}


def search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Keyword search over earlier research: which topics and paper notes mention these words."""
    terms = [t for t in re.findall(r"\w{3,}", query.lower())]
    if not terms:
        return []
    hits = []
    for s in _sections(_read(notebook_path())):
        blocks = [("summary", "", s["summary"])] + [("paper", p.group("id"), p.group("body"))
                                                    for p in PAPER_RE.finditer(s["body"])]
        blocks += [("knowledge", "", f"[{kind}] {it['text']}") for kind, items in s["knowledge"].items() for it in items]
        blocks += [("experiment", "", f"[{e['verdict']}] {e['change']} · {e['settings']} → {e['result']}")
                   for e in s["experiments"]]
        head = f"{s['title']} {s['question']} {' '.join(s['queries'])}".lower()
        for kind, pid, body in blocks:
            low = body.lower()
            score = sum(low.count(t) for t in terms) + 3 * sum(t in head for t in terms)
            if score:
                i = min((low.find(t) for t in terms if t in low), default=0)
                hits.append({"section": s["id"], "title": s["title"], "kind": kind, "paper_id": pid or None,
                             "score": score, "snippet": " ".join(body[max(0, i - 150): i + 350].split())})
    return sorted(hits, key=lambda h: h["score"], reverse=True)[:limit]


def cited_ids() -> set[str]:
    """Every source id the notebook cites: these are never pruned from the base."""
    path = notebook_path()
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8")
    ids = {p.group("id") for p in PAPER_RE.finditer(text)}
    for m in PLAYBOOK_RE.finditer(text):
        ids.update(re.findall(r"`([^`\s]+:[^`\s]+)`", m.group("body")))
    return ids


def sections_with(paper_id: str) -> list[str]:
    path = notebook_path()
    if not path.exists():
        return []
    return [s["id"] for s in _sections(path.read_text(encoding="utf-8")) if paper_id in s["papers"]]


def start(title: str, question: str, queries: list[str] | None = None) -> dict[str, Any]:
    title = " ".join(title.split()).lstrip("#").strip()
    if not title:
        raise NotebookError("title is empty")
    path = notebook_path()
    with _locked(path):
        text = _read(path)
        sid = slug(title)
        for s in _sections(text):
            if s["id"] == sid:
                return {"status": "exists", "section": sid, "papers": s["papers"], "summary": s["summary"][:1500],
                        "note": "This topic was researched before: build on it (add papers, update conclusions)."}
        L = labels()
        meta = {"question": question.strip(), "queries": list(dict.fromkeys(queries or [])),
                "created": _today(), "updated": _today()}
        block = (f'\n<!-- research:section id="{sid}" -->\n## {title}\n\n{_header(meta)}\n\n'
                 f"### {L['summary']}\n<!-- research:summary -->\n{L['no_summary']}\n<!-- /research:summary -->\n\n"
                 f"### {L['papers']}\n\n<!-- /research:section -->\n")
        text = _with_toc(text.rstrip("\n") + "\n" + block)
        _write(path, text)
    return {"status": "created", "section": sid, "path": str(path)}


def add_paper(section: str, paper: dict[str, Any], notes: str, replace: bool = False) -> dict[str, Any]:
    if not notes.strip():
        raise NotebookError("notes are empty: write what the paper does, its key numbers and how it applies")
    path = notebook_path()
    with _locked(path):
        text = _read(path)
        s = _find(text, section)
        body = s["body"]
        block = paper_block(paper, notes)
        existing = next((m for m in PAPER_RE.finditer(body) if m.group("id") == paper["id"]), None)
        if existing and not replace:
            return {"status": "exists", "section": s["id"], "paper": paper["id"],
                    "current_notes": existing.group("body").strip()[:3000],
                    "note": "Already in this section. Pass replace=true to rewrite its notes."}
        if existing:
            body = body[: existing.start()] + block + body[existing.end():]
        else:
            body = body.rstrip() + "\n\n" + block + "\n\n"
        text = _with_toc(_replace_section(text, s, _touch(body)))
        _write(path, text)
    return {"status": "replaced" if existing else "added", "section": s["id"], "paper": paper["id"],
            "papers_in_section": len(s["papers"]) + (0 if existing else 1)}


def write(section: str, summary: str | None = None, append: bool = False, question: str | None = None,
          add_queries: list[str] | None = None, remove_papers: list[str] | None = None) -> dict[str, Any]:
    path = notebook_path()
    changed = []
    with _locked(path):
        text = _read(path)
        s = _find(text, section)
        body = s["body"]
        if summary is not None:
            m = SUMMARY_RE.search(body)
            old = m.group("body").strip() if m else ""
            new = summary.strip()
            if append and _summary_text(old):
                new = old + "\n\n" + new
            block = f"<!-- research:summary -->\n{new}\n<!-- /research:summary -->"
            body = body[: m.start()] + block + body[m.end():] if m else body.rstrip() + "\n\n" + block + "\n\n"
            changed.append("summary")
        h = HEADER_RE.search(body)
        if h and (question or add_queries):
            meta = json.loads(h.group("meta"))
            if question:
                meta["question"] = question.strip()
            if add_queries:
                meta["queries"] = list(dict.fromkeys([*meta.get("queries", []), *add_queries]))
            body = body[: h.start()] + _header(meta) + body[h.end():]
            changed.append("header")
        for pid in remove_papers or []:
            m = next((m for m in PAPER_RE.finditer(body) if m.group("id") == pid), None)
            if m:
                body = body[: m.start()].rstrip() + "\n\n" + body[m.end():].lstrip()
                changed.append(f"removed {pid}")
        if not changed:
            return {"status": "nothing to change", "section": s["id"]}
        text = _with_toc(_replace_section(text, s, _touch(body)))
        _write(path, text)
    return {"status": "updated", "section": s["id"], "changed": changed}


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))[:90]


def add_knowledge(section: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Add what a specialist must know (concepts, decisions, practices, pitfalls, metrics, tools) to a topic.
    Every item cites the sources it comes from; an item already there (same opening words) is not repeated."""
    path = notebook_path()
    with _locked(path):
        text = _read(path)
        s = _find(text, section)
        current = s["knowledge"]
        added, skipped = 0, []
        for it in items:
            kind = str(it.get("kind", "")).strip().lower()
            body = " ".join(str(it.get("text", "")).split()).replace(" — ", " – ")
            ids = [str(i).strip() for i in it.get("sources") or [] if str(i).strip()]
            if kind not in KNOWLEDGE_KINDS:
                skipped.append(f"{body[:40]!r}: kind must be one of {', '.join(KNOWLEDGE_KINDS)}")
            elif len(body) < 15:
                skipped.append(f"{body[:40]!r}: too short — state the fact and what it means for the work")
            elif not ids:
                skipped.append(f"{body[:40]!r}: no sources — cite the ids it comes from")
            elif any(_norm(body) == _norm(x["text"]) for x in current[kind]):
                skipped.append(f"{body[:40]!r}: already there")
            else:
                current[kind].append({"text": body, "sources": list(dict.fromkeys(ids))})
                added += 1
        if added:
            body = s["body"]
            block = _playbook_block(current)
            m = PLAYBOOK_RE.search(body)
            if m:   # replace it together with its heading line
                start = m.start()
                for L in LABELS.values():
                    if body[:start].endswith(f"### {L['playbook']}\n"):
                        start -= len(f"### {L['playbook']}\n")
                body = body[:start] + block + body[m.end():]
            else:   # right before the sources list (or the first source), else at the end
                at = body.find(f"\n### {labels()['papers']}")
                if at < 0:
                    first = PAPER_RE.search(body)
                    at = first.start() - 1 if first else len(body.rstrip())
                body = body[:at].rstrip() + "\n\n" + block + "\n\n" + body[at:].lstrip("\n")
            text = _with_toc(_replace_section(text, s, _touch(body)))
            _write(path, text)
    counts = {k: len(v) for k, v in current.items()}
    return {"status": "updated" if added else "nothing added", "section": s["id"], "added": added,
            "skipped": skipped, "knowledge_counts": counts}


VERDICTS = ("better", "worse", "same", "failed", "unclear")


def add_experiment(section: str, change: str, result: str, verdict: str, settings: str = "",
                   when: str | None = None) -> dict[str, Any]:
    """Log one try: what was changed, with which settings, what came out and whether it helped."""
    verdict = verdict.strip().lower()
    if verdict not in VERDICTS:
        raise NotebookError(f"verdict must be one of {', '.join(VERDICTS)}")
    if len(change.strip()) < 3 or len(result.strip()) < 2:
        raise NotebookError("experiment needs `change` (what was tried) and `result` (what came out, with numbers)")
    path = notebook_path()
    L = labels()
    with _locked(path):
        text = _read(path)
        s = _find(text, section)
        body = s["body"]
        mark = {"better": "**", "failed": "", "worse": ""}.get(verdict, "")
        row = (f"| {when or _today()} | {_cell(change)} | {_cell(settings)} | {_cell(result)} | "
               f"{mark}{L['verdicts'][verdict]}{mark} |")
        m = EXPERIMENTS_RE.search(body)
        if m:
            inner = m.group("body").rstrip("\n") + "\n" + row + "\n"
            body = body[: m.start("body")] + inner + body[m.end("body"):]
        else:
            block = (f"### {L['experiments']}\n<!-- research:experiments -->\n{L['exp_head']}\n|---|---|---|---|---|\n"
                     f"{row}\n<!-- /research:experiments -->")
            at = body.find(f"\n### {L['papers']}")
            if at < 0:
                first = PAPER_RE.search(body)
                at = first.start() - 1 if first else len(body.rstrip())
            body = body[:at].rstrip() + "\n\n" + block + "\n\n" + body[at:].lstrip("\n")
        text = _with_toc(_replace_section(text, s, _touch(body)))
        _write(path, text)
    exps = _parse_experiments(body)
    return {"status": "logged", "section": s["id"], "experiments": len(exps),
            "failed_or_worse": [e["change"] for e in exps if e["verdict"] in ("failed", "worse", "same")]}
