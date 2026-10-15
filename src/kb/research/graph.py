"""follow_citations: walk the citation graph from a paper (what it builds on / who built on it)."""

from __future__ import annotations

import logging
from typing import Any

from kb.critic.scoring import grade_and_save
from kb.research import sources
from kb.research.find import brief
from kb.schemas import Paper
from kb.store import Database

log = logging.getLogger(__name__)


def _s2_ref(p: dict[str, Any]) -> str | None:
    if p.get("arxiv_id"):
        return f"arXiv:{p['arxiv_id']}"
    if p.get("doi"):
        return f"DOI:{p['doi']}"
    return p.get("s2_id")


def _openalex_id(db: Database, oa: sources.OpenAlex, p: dict[str, Any]) -> str | None:
    if p.get("openalex_id"):
        return p["openalex_id"]
    doi = p.get("doi") or (f"10.48550/arXiv.{p['arxiv_id']}" if p.get("arxiv_id") else None)
    work = oa.work(f"doi:{doi}") if doi else None
    if not work:
        return None
    oa_id = work["id"].rsplit("/", 1)[-1]
    db.conn.execute("UPDATE papers SET openalex_id = ? WHERE id = ?", (oa_id, p["id"]))
    return oa_id


def _via_openalex(db: Database, p: dict[str, Any], direction: str, limit: int) -> list[tuple[Paper, bool]]:
    oa = sources.OpenAlex()
    try:
        oa_id = _openalex_id(db, oa, p)
        if not oa_id:
            return []
        if direction == "citations":
            return [(x, False) for x in oa.works_filter(f"cites:{oa_id}", limit)]
        work = oa.work(oa_id) or {}
        refs = [r.rsplit("/", 1)[-1] for r in work.get("referenced_works") or []][:100]
        if not refs:
            return []
        return [(x, False) for x in oa.works_filter(f"openalex_id:{'|'.join(refs)}", limit)]
    finally:
        oa.close()


def follow_citations(db: Database, paper_id: str, direction: str = "references", limit: int = 20) -> dict[str, Any]:
    if direction not in ("references", "citations"):
        return {"error": "direction must be 'references' (what it builds on) or 'citations' (who cites it)"}
    p = db.get_paper(paper_id)
    if p is None:
        return {"error": f"unknown paper {paper_id}: find it with find_papers first"}
    # OpenAlex can sort citing works by their own citations (S2 lists the newest first, mostly uncited);
    # for references S2 is better: it marks the influential ones.
    order = ("openalex", "s2") if direction == "citations" else ("s2", "openalex")
    links: list[tuple[Paper, bool]] = []
    used, errors = None, []
    for src in order:
        try:
            if src == "openalex":
                links = _via_openalex(db, p, direction, max(limit * 2, 25))
            elif ref := _s2_ref(p):
                s2 = sources.SemanticScholar()
                try:
                    links = s2.links(ref, direction, limit=1000 if direction == "citations" else max(limit * 3, 50))
                finally:
                    s2.close()
        except Exception as e:
            errors.append(f"{src}: {type(e).__name__}")
            log.info("%s %s failed: %s", src, direction, e)
        if links:
            used = src
            break
    if not links:
        return {"error": "no links found" + (f" ({'; '.join(errors)})" if errors else ""), "paper": p["title"]}
    # Influential links first, then the most cited.
    links.sort(key=lambda x: (x[1], x[0].citation_count or 0), reverse=True)
    out = []
    for paper, influential in links[:limit]:
        pid, _ = db.upsert_paper(paper)
        grade_and_save(db, pid)
        b = brief(db, pid, abstract_chars=300)
        b["influential"] = influential
        out.append(b)
    db.commit()
    return {"paper": p["title"], "direction": direction, "source": used, "results": out,
            "note": "References: what it builds on. Citations: follow-ups, reproductions and critiques."}
