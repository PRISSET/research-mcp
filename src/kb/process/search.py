"""Hybrid retrieval over full-text chunks: BM25 (FTS5) + dense vectors, fused with RRF, then a cross-encoder reranker.

Results carry citations (paper, year, section, arXiv link) and the paper's evidence grade, and at most
`per_paper` chunks per paper so one long paper cannot crowd out the rest.
"""

from __future__ import annotations

import logging
import os
import threading
import re
from functools import cache
from typing import Any

from kb.store import Database

log = logging.getLogger(__name__)

RERANK_MODEL = os.environ.get("KB_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RRF_K = 60
CPU_CANDIDATES = int(os.environ.get("KB_CPU_RERANK", "16"))
# Final score = reranker probability + FUSION_WEIGHT * RRF. Pure reranking loses some precise English
# lexical hits; pure RRF is weak cross-lingually. Tuned on eval/search_questions.yaml.
FUSION_WEIGHT = float(os.environ.get("KB_FUSION_WEIGHT", "15"))
STOP = {"the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "is", "are", "how", "what", "which", "with",
        "does", "do", "can", "best", "way", "use", "using", "vs", "my", "i", "it", "be", "by", "at", "from", "as"}
GRADE_BOOST = {"A": 0.06, "B": 0.03, "C": 0.0, "D": -0.05}
# Related-work passages describe *other* papers second-hand: prefer the primary source.
SECOND_HAND = re.compile(r"related work|background|literature", re.IGNORECASE)


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


_LOAD_LOCK = threading.Lock()


def reranker(device: str | None = None) -> Any:
    from kb.process.embed import pick_device, touch

    with _LOAD_LOCK:
        touch()
        return _reranker(device or pick_device())


@cache
def _reranker(device: str) -> Any:
    import torch
    from sentence_transformers import CrossEncoder

    from kb.process.embed import load_local_first

    kw = {"torch_dtype": torch.float16} if device == "cuda" else {}
    # On the CPU a shorter window keeps reranking to a few seconds; passages are ~300 words anyway.
    return load_local_first(CrossEncoder, RERANK_MODEL, device=device, max_length=512 if device == "cuda" else 384,
                            model_kwargs=kw)


def _rerank(pairs: list[tuple[str, str]]) -> Any:
    import torch

    from kb.process.embed import in_use, use_cpu_threads

    use_cpu_threads()
    with in_use():
        try:
            return reranker().predict(pairs, batch_size=16, show_progress_bar=False)
        except torch.OutOfMemoryError:
            log.warning("GPU out of memory: reranking on the CPU")
            torch.cuda.empty_cache()
            return reranker("cpu").predict(pairs, batch_size=16, show_progress_bar=False)


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
    rerank: bool = True,
    paper_ids: set[str] | None = None,
    year_from: int | None = None,
    grades: str | None = None,
    kinds: tuple[str, ...] = ("text", "table", "figure"),
    per_paper: int = 2,
    candidates: int = 60,
) -> list[dict[str, Any]]:
    from kb.process.embed import vector_search, wait_idle

    wait_idle(float(os.environ.get("KB_EMBED_WAIT", "15")))   # pages read just before this search
    allowed = _paper_filter(db, paper_ids, year_from, grades)
    if allowed is not None and not allowed:
        return []
    lexical = bm25(db, query, 100, allowed)
    try:
        dense = vector_search(db, query, 100, allowed)
    except Exception as e:  # no torch / no vectors yet: BM25 only
        log.warning("vector search unavailable: %s", e)
        dense = []
    fused: dict[int, float] = {}
    for ranking in (lexical, dense):
        for rank, (cid, _) in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    if not fused:
        return []
    if rerank:
        from kb.process.embed import pick_device

        if pick_device() == "cpu":   # the cross-encoder dominates CPU latency: rerank only the head
            candidates = min(candidates, CPU_CANDIDATES)
    top = sorted(fused, key=fused.get, reverse=True)[:candidates]  # type: ignore[arg-type]
    rows = {r["id"]: dict(r) for r in db.conn.execute(
        f"""SELECT c.id, c.paper_id, c.section, c.kind, c.page, c.text, p.title, p.year, p.arxiv_id, p.url AS doc_url, p.source,
                   p.citation_count, COALESCE(cr.grade, 'C') AS grade
            FROM chunks c JOIN papers p ON p.id = c.paper_id LEFT JOIN critique cr ON cr.source_id = p.id
            WHERE c.id IN ({','.join('?' * len(top))})""", top)}
    cands = [rows[c] for c in top if c in rows and rows[c]["kind"] in kinds]
    if rerank and cands:
        try:
            scores = _rerank([(query, f"{c['title']}\n{c['section']}\n{c['text']}") for c in cands])
            for c, s in zip(cands, scores):
                c["score"] = (float(s) + FUSION_WEIGHT * fused[c["id"]] + GRADE_BOOST.get(c["grade"], 0.0)
                              - (0.08 if SECOND_HAND.search(c["section"]) else 0.0))
        except Exception as e:
            log.warning("reranker unavailable: %s", e)
            rerank = False
    if not rerank:
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
    return out


def context(db: Database, chunk_id: int, window: int = 1) -> list[dict[str, Any]]:
    """Neighbouring chunks of a hit, for reading a passage in context."""
    row = db.conn.execute("SELECT paper_id, ord FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    if not row:
        return []
    return [dict(r) for r in db.conn.execute(
        "SELECT id, ord, section, kind, page, text FROM chunks WHERE paper_id = ? AND ord BETWEEN ? AND ? ORDER BY ord",
        (row["paper_id"], row["ord"] - window, row["ord"] + window))]


def warm_up() -> None:
    """Load the models and the vector matrix ahead of the first query (run in a background thread)."""
    try:
        from kb.process.embed import vector_search
        from kb.store import Database

        from kb.process.embed import resume_queue

        db = Database()
        try:
            vector_search(db, "warm up", k=1)
        finally:
            db.close()
        reranker()
        resume_queue()   # pages read just before the previous server stopped
    except Exception as e:  # no torch: keyword search only, nothing to warm
        log.info("warm-up skipped: %s", e)
