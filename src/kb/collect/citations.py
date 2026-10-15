"""Enrich stored papers with Semantic Scholar data (citations, TL;DR, venue)."""

from __future__ import annotations

import logging

from kb.collect.semantic_scholar import SemanticScholarClient, to_db_fields
from kb.store import Database

log = logging.getLogger(__name__)


def enrich_citations(db: Database, limit: int | None = None) -> tuple[int, int]:
    rows = db.papers_missing_s2(limit)
    if not rows:
        return 0, 0
    ids = [f"ARXIV:{r['arxiv_id']}" if r["arxiv_id"] else f"DOI:{r['doi']}" for r in rows]
    client = SemanticScholarClient()
    found = 0
    try:
        for start in range(0, len(rows), 400):
            chunk_rows, chunk_ids = rows[start : start + 400], ids[start : start + 400]
            for row, item in zip(chunk_rows, client.batch(chunk_ids), strict=True):
                if item:
                    db.update_s2(row["id"], to_db_fields(item))
                    found += 1
                else:
                    db.mark_s2_checked(row["id"])
            db.commit()
            log.info("s2: %d/%d processed", min(start + 400, len(rows)), len(rows))
    finally:
        client.close()
    return len(rows), found
