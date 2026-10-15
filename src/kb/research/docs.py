"""Non-paper sources (repos, issues, model cards, web pages, forum threads) stored next to papers.

They share the `papers` table (kind != 'paper', platform metadata in `extra`) and the full-text index, so
search_fulltext, find_paper_numbers and the notebook treat a README or an issue thread like a paper.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from kb.process.parse import Block
from kb.store import Database

KIND_LABEL = {
    "paper": "paper", "repo": "GitHub repo", "issue": "GitHub issue", "pr": "GitHub pull request",
    "model": "HF model", "dataset": "HF dataset", "space": "HF space", "web": "web page",
    "hn": "Hacker News thread", "so": "Stack Overflow question",
}


@dataclass
class Doc:
    id: str
    kind: str
    title: str
    url: str
    platform: str                         # github | huggingface | web | hackernews | stackoverflow
    summary: str = ""
    published: str | None = None          # ISO date
    authors: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def upsert_doc(db: Database, d: Doc) -> str:
    year = int(d.published[:4]) if d.published and d.published[:4].isdigit() else None
    db.conn.execute(
        """INSERT INTO papers (id, title, abstract, authors, year, published, url, source, kind, extra)
           VALUES (:id, :title, :summary, :authors, :year, :published, :url, :platform, :kind, :extra)
           ON CONFLICT(id) DO UPDATE SET title = excluded.title,
               abstract = COALESCE(NULLIF(excluded.abstract, ''), abstract),
               authors = CASE WHEN excluded.authors = '[]' THEN authors ELSE excluded.authors END,
               year = COALESCE(excluded.year, year), published = COALESCE(excluded.published, published),
               url = excluded.url, kind = excluded.kind, extra = json_patch(extra, excluded.extra)""",
        {"id": d.id, "title": d.title[:500], "summary": d.summary[:4000], "authors": json.dumps(d.authors[:20]),
         "year": year, "published": d.published, "url": d.url, "platform": d.platform, "kind": d.kind,
         "extra": json.dumps(d.extra, ensure_ascii=False, default=str)},
    )
    db._reindex(d.id)
    return d.id


# ---------------- Markdown -> blocks ----------------

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def parse_markdown(md: str, title: str = "Body") -> list[Block]:
    """Split Markdown into blocks: paragraphs and code under their heading path, tables as their own blocks."""
    md = re.sub(r"\A---\n.*?\n---\n", "", md, flags=re.S)        # trafilatura front matter
    path: list[tuple[int, str]] = []
    blocks: list[Block] = []
    buf: list[str] = []
    kind = "text"
    in_code = False

    def section() -> str:
        return " > ".join(h for _, h in path) or title

    def flush() -> None:
        nonlocal buf, kind
        text = "\n".join(buf).strip()
        if text:
            blocks.append(Block(section(), kind, text if kind == "table" or in_code else " ".join(text.split())))
        buf, kind = [], "text"

    for line in md.splitlines():
        if line.strip().startswith("```"):
            if in_code:
                buf.append(line)
                flush()
                in_code = False
            else:
                flush()
                in_code = True
                buf.append(line)
            continue
        if in_code:
            buf.append(line)
            continue
        m = HEADING.match(line)
        if m:
            flush()
            level, text = len(m.group(1)), re.sub(r"[*_`]", "", m.group(2)).strip()
            path = [(lv, h) for lv, h in path if lv < level] + [(level, text[:120])]
            continue
        is_table = line.lstrip().startswith("|")
        if not line.strip() or (is_table != (kind == "table") and buf):
            flush()
            if not line.strip():
                continue
        kind = "table" if is_table else "text"
        buf.append(line)
    flush()
    return [b for b in blocks if len(b.text) > 20 or b.kind == "table"]


def index_text(db: Database, doc_id: str, markdown: str, source: str, title: str = "Body",
               embed: bool = True) -> dict[str, Any]:
    """Chunk + store + extract numbers (+ vectors when available) for one document."""
    from kb.process.claims import extract_all
    from kb.process.fulltext import _store

    blocks = parse_markdown(markdown, title)
    if not blocks:
        return {"chunks": 0}
    n = _store(db, doc_id, source, blocks)
    db.commit()
    extract_all(db, rebuild=True, paper_ids=[doc_id])
    vectors: Any = "skipped"
    if embed:
        try:
            from kb.process.embed import embed_later

            vectors = embed_later([doc_id])
        except Exception as e:  # no torch: keyword search still works
            vectors = f"unavailable ({type(e).__name__})"
    return {"chunks": n, "vectors": vectors}
