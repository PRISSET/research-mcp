"""find_papers: search all sources at once, merge duplicates, rank, store and grade.

Ranking: reciprocal-rank fusion over the sources (a paper several sources put on top wins), plus a small boost
for citations. Every result is stored locally, so full texts, numbers and notes can refer to it later.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

from kb.config import settings
from kb.critic.scoring import grade_and_save
from kb.research import sources
from kb.schemas import Paper
from kb.store import Database
from kb.textutil import normalize

log = logging.getLogger(__name__)

RRF_K = 60
SOURCE_DEADLINE = float(__import__("os").environ.get("KB_SOURCE_DEADLINE", "25"))
COOLDOWN = 300.0                      # seconds a throttled source is left alone
_COOLDOWN: dict[str, float] = {}      # source -> monotonic time it may be asked again (per server process)
SOURCES = ("openalex", "arxiv", "s2", "local")
# arXiv's own search matches words, it does not rank by relevance well: its votes count less.
WEIGHT = {"openalex": 1.0, "s2": 1.0, "arxiv": 0.5, "local": 0.8}
CITATION_BOOST = 0.008        # per decade of citations; 1000 citations ~ a top-3 place in one source


def _key(p: Paper) -> str:
    return f"arxiv:{p.arxiv_id}" if p.arxiv_id else (f"doi:{p.doi.lower()}" if p.doi else f"t:{normalize(p.title)}")


def _run_source(name: str, query: str, limit: int, year_from: int | None) -> list[Paper]:
    if name == "openalex":
        c = sources.OpenAlex()
        try:
            return c.search(query, limit, year_from)
        finally:
            c.close()
    if name == "s2":
        c2 = sources.SemanticScholar()
        try:
            return c2.search(query, limit, year_from)
        finally:
            c2.close()
    if name == "arxiv":
        return sources.arxiv_search(query, limit, year_from)
    raise ValueError(name)


def _local(db: Database, query: str, limit: int, year_from: int | None) -> list[Paper]:
    out = []
    # Every word must match: an OR over common words ("models", "large") drags in unrelated cached papers.
    for r in db.search(query, year_from=year_from, limit=limit, match_all=True):
        out.append(Paper(id=r["id"], title=r["title"], arxiv_id=r.get("arxiv_id"), doi=r.get("doi"), year=r.get("year"),
                         citation_count=r.get("citation_count"), source="local"))
    return out


def enrich_s2(db: Database, ids: list[str]) -> int:
    """Citation counts, influential citations and TL;DRs for papers not yet checked. Best effort without a key."""
    from kb.collect.semantic_scholar import PAPER_FIELDS, to_db_fields

    rows = db.papers_missing_s2(ids=ids)
    if not rows or _COOLDOWN.get("s2", 0) > time.monotonic():   # throttled a moment ago: no 3 s wait for nothing
        return 0
    client = sources.SemanticScholar()
    client.http.max_retries = 1
    found = 0
    try:
        refs = [f"ARXIV:{r['arxiv_id']}" if r["arxiv_id"] else f"DOI:{r['doi']}" for r in rows]
        resp = client.http.post(f"{settings.s2_api}/paper/batch", params={"fields": PAPER_FIELDS},
                                json={"ids": refs})
        for row, item in zip(rows, resp.json(), strict=True):
            if item:
                db.update_s2(row["id"], to_db_fields(item))
                found += 1
            else:
                db.mark_s2_checked(row["id"])
    except Exception as e:
        log.info("s2 enrichment skipped: %s", e)
        if "429" in str(e):
            _COOLDOWN["s2"] = time.monotonic() + COOLDOWN
    finally:
        client.close()
    return found


def brief(db: Database, paper_id: str, abstract_chars: int = 500) -> dict[str, Any]:
    p = db.get_paper(paper_id) or {"id": paper_id}
    crit = db.conn.execute("SELECT grade, flags FROM critique WHERE source_id = ?", (p["id"],)).fetchone()
    ft = db.conn.execute("SELECT status, n_chunks FROM paper_fulltext WHERE paper_id = ?", (p["id"],)).fetchone()
    return {
        "id": p["id"], "title": p.get("title"), "year": p.get("year"), "venue": p.get("venue"),
        "authors": (p.get("authors") or [])[:4], "citations": p.get("citation_count"),
        "grade": crit["grade"] if crit else None, "flags": json.loads(crit["flags"]) if crit else [],
        "tldr": p.get("tldr"), "abstract": (p.get("abstract") or "")[:abstract_chars], "url": p.get("url"),
        "fulltext": (ft["status"] if ft else ("available" if p.get("arxiv_id") or p.get("pdf_url") else "none")),
    }


def find_papers(db: Database, query: str, limit: int = 15, year_from: int | None = None,
                use: tuple[str, ...] = SOURCES, on_paper: Callable[[str], list[str]] | None = None) -> dict[str, Any]:
    per_source = max(limit * 2, 20)
    results: dict[str, list[Paper]] = {}
    status: dict[str, str] = {}
    now = time.monotonic()
    remote = []
    for s in use:
        if s == "local":
            continue
        if _COOLDOWN.get(s, 0) > now:   # recently throttled: do not wait on it again
            status[s] = f"skipped: throttled {int(COOLDOWN - (_COOLDOWN[s] - now))}s ago, retry in {int(_COOLDOWN[s] - now)}s"
        else:
            remote.append(s)
    # Each source gets a deadline: a throttled one (arXiv after a burst) must not hold the others' results.
    pool = ThreadPoolExecutor(max_workers=max(1, len(remote)))
    futures = {s: pool.submit(_run_source, s, query, per_source, year_from) for s in remote}
    deadline = time.monotonic() + SOURCE_DEADLINE
    for s, f in futures.items():
        try:
            results[s] = f.result(timeout=max(0.1, deadline - time.monotonic()))
            status[s] = f"ok ({len(results[s])})"
        except FuturesTimeout:
            status[s] = f"timeout after {SOURCE_DEADLINE:.0f}s (rate-limited or slow) — skipped"
            _COOLDOWN[s] = time.monotonic() + COOLDOWN
        except Exception as e:  # one source down must not fail the search
            status[s] = f"failed: {type(e).__name__}: {str(e)[:120]}"
            log.info("source %s failed: %s", s, e)
            if "429" in str(e) or "503" in str(e):
                _COOLDOWN[s] = time.monotonic() + COOLDOWN
    pool.shutdown(wait=False, cancel_futures=True)
    if "local" in use:
        results["local"] = _local(db, query, per_source, year_from)
        status["local"] = f"ok ({len(results['local'])})"

    score: dict[str, float] = {}
    best: dict[str, Paper] = {}
    cites: dict[str, int] = {}
    found_in: dict[str, list[str]] = {}
    for s, papers in results.items():
        for rank, p in enumerate(papers):
            k = _key(p)
            score[k] = score.get(k, 0.0) + WEIGHT.get(s, 1.0) / (RRF_K + rank + 1)
            cites[k] = max(cites.get(k, 0), p.citation_count or 0)
            found_in.setdefault(k, []).append(s)
            if s != "local" and (k not in best or len(p.abstract) > len(best[k].abstract)):
                best[k] = p
    for k, p in best.items():  # keep the highest count any source reported (OpenAlex undercounts preprints)
        p.citation_count = cites[k] or p.citation_count
    ranked = sorted(score, key=lambda k: score[k] + CITATION_BOOST * math.log10(1 + cites.get(k, 0)), reverse=True)

    stored: list[tuple[str, str]] = []           # (key, stored id)
    for k in ranked[: limit * 2]:
        if k in best:
            pid, _ = db.upsert_paper(best[k])
        else:  # only in the local cache
            local = next(p for p in results["local"] if _key(p) == k)
            pid = local.id
        stored.append((k, pid))
    db.commit()
    top = stored[:limit]
    if not status.get("s2", "").startswith(("failed", "skipped", "timeout")):  # S2 throttled us: do not wait again
        enrich_s2(db, [pid for k, pid in top if "s2" not in found_in.get(k, [])])
    for _, pid in top:
        grade_and_save(db, pid)
    new = sum(1 for k, _ in top if "local" not in found_in.get(k, []))
    db.log_run("find_papers", query, len(score), new)
    db.commit()

    out = []
    for k, pid in top:
        b = brief(db, pid)
        b["found_in"] = found_in.get(k, [])
        if on_paper:
            b["in_notebook"] = on_paper(pid)
        out.append(b)
    return {"query": query, "sources": status, "results": out,
            "next": "fetch_fulltext(ids) for the promising ones, then search_fulltext / read_passage for details."}
