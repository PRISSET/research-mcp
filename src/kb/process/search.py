"""Hybrid retrieval over full-text chunks: BM25 (FTS5) + dense vectors, fused with RRF.

No cross-encoder reranker: on the eval corpus it only moved the right paper from 2nd-5th to 1st place (recall@5
unchanged, worse for some queries) for ~1.5 GB of RAM, and the agent reads all k hits anyway.

Results carry citations (paper, year, section, arXiv link) and the paper's evidence grade, and at most
`per_paper` chunks per paper so one long paper cannot crowd out the rest.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from kb.store import Database

log = logging.getLogger(__name__)

RRF_K = 60
STOP = {"the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "is", "are", "how", "what", "which", "with",
        "does", "do", "can", "best", "way", "use", "using", "vs", "my", "i", "it", "be", "by", "at", "from", "as"}
GRADE_BOOST = {"A": 0.06, "B": 0.03, "C": 0.0, "D": -0.05}


def fts_query(text: str) -> str:
    terms = [t for t in re.findall(r"[\w][\w.+-]*", text.lower()) if t not in STOP and len(t) > 1]
    return " OR ".join(f'"{t}"' for t in terms) or '""'


def bm25(db: Database, query: str, k: int = 100, paper_ids: set[str] | None = None) -> list[tuple[int, float]]:
    sql = """SELECT c.id, bm25(chunks_fts, 1.0, 0.4) AS r FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
             WHERE chunks_fts MATCH ?"""
    params: list[Any] = [fts_query(query)]
    if paper_ids is not None:
        sql += f" AND c.paper_id IN ({','.join('?' * len(paper_ids))})"
        params += list(paper_ids)
    sql += " ORDER BY r LIMIT ?"
    params.append(k)
    try:
        return [(r[0], -r[1]) for r in db.conn.execute(sql, params)]
    except Exception as e:  # malformed query
        log.debug("fts: %s", e)
        return []


def _paper_filter(db: Database, paper_ids: set[str] | None, year_from: int | None, grades: str | None) -> set[str] | None:
    if not (paper_ids is not None or year_from or grades):
        return None
    sql = "SELECT DISTINCT p.id FROM papers p LEFT JOIN critique c ON c.source_id = p.id WHERE 1=1"
    params: list[Any] = []
    if paper_ids is not None:
        if not paper_ids:
            return set()
        sql += f" AND p.id IN ({','.join('?' * len(paper_ids))})"
        params += list(paper_ids)
    if year_from:
        sql += " AND p.year >= ?"
        params.append(year_from)
    if grades:
        sql += f" AND COALESCE(c.grade, 'C') IN ({','.join('?' * len(grades))})"
        params += list(grades)
    return {r[0] for r in db.conn.execute(sql, params)}


def search(
    db: Database,
    query: str,
    k: int = 8,
    paper_ids: set[str] | None = None,
    year_from: int | None = None,
    grades: str | None = None,
    kinds: tuple[str, ...] = ("text", "table", "figure"),
    per_paper: int = 2,
    candidates: int = 60,
    dense: bool = True,
) -> list[dict[str, Any]]:
    from kb.process.embed import vector_search

    allowed = _paper_filter(db, paper_ids, year_from, grades)
    if allowed is not None and not allowed:
        return []
    rankings = [bm25(db, query, 100, allowed)]
    if dense:
        try:
            rankings.append(vector_search(db, query, 100, allowed))
        except Exception as e:  # no model / no vectors yet: BM25 only
            log.warning("vector search unavailable: %s", e)
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    if not fused:
        return []
    top = sorted(fused, key=fused.get, reverse=True)[:candidates]  # type: ignore[arg-type]
    rows = {r["id"]: dict(r) for r in db.conn.execute(
        f"""SELECT c.id, c.paper_id, c.section, c.kind, c.page, c.text, p.title, p.year, p.arxiv_id, p.url AS doc_url, p.source,
                   p.citation_count, COALESCE(cr.grade, 'C') AS grade
            FROM chunks c JOIN papers p ON p.id = c.paper_id LEFT JOIN critique cr ON cr.source_id = p.id
            WHERE c.id IN ({','.join('?' * len(top))})""", top)}
    cands = [rows[c] for c in top if c in rows and rows[c]["kind"] in kinds]
    for c in cands:
        c["score"] = fused[c["id"]] * 30 + GRADE_BOOST.get(c["grade"], 0.0)
    cands.sort(key=lambda c: c["score"], reverse=True)
    out, per = [], {}
    for c in cands:
        if per.get(c["paper_id"], 0) >= per_paper:
            continue
        per[c["paper_id"]] = per.get(c["paper_id"], 0) + 1
        aid = c["arxiv_id"]
        out.append({
            "score": round(c["score"], 4), "chunk_id": c["id"], "paper_id": c["paper_id"], "title": c["title"],
            "year": c["year"], "grade": c["grade"], "section": c["section"], "kind": c["kind"], "page": c["page"],
            "text": c["text"], "url": f"https://arxiv.org/abs/{aid}" if aid else c["doc_url"],
            "citation": f"{c['title']} ({c['year']}), {c['section']}" + (f", p.{c['page']}" if c["page"] else ""),
        })
        if len(out) >= k:
            break
    db.touch({h["paper_id"] for h in out})
    return out


def context(db: Database, chunk_id: int, window: int = 1) -> list[dict[str, Any]]:
    """Neighbouring chunks of a hit, for reading a passage in context."""
    row = db.conn.execute("SELECT paper_id, ord FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    if not row:
        return []
    db.touch({row["paper_id"]})
    return [dict(r) for r in db.conn.execute(
        "SELECT id, ord, section, kind, page, text FROM chunks WHERE paper_id = ? AND ord BETWEEN ? AND ? ORDER BY ord",
        (row["paper_id"], row["ord"] - window, row["ord"] + window))]
