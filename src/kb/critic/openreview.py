"""OpenReview acceptance status by title search (anonymous API). Review texts need login and are not fetched."""

from __future__ import annotations

import logging
import re

import httpx

from kb.collect.http import RateLimitedClient
from kb.store import Database
from kb.textutil import similarity

log = logging.getLogger(__name__)
API = "https://api2.openreview.net/notes/search"


def classify_venue(venue: str) -> str:
    v = venue.lower()
    if "withdrawn" in v or "desk reject" in v:
        return "withdrawn"
    if v.startswith("submitted to") or "reject" in v:
        return "rejected_or_pending"
    if re.search(r"\b(poster|oral|spotlight|accept|conference|workshop)\b", v) and re.search(r"iclr|neurips|icml|tmlr|colm|aaai|cvpr", v):
        return "accepted"
    return "not_found"


def check_openreview(db: Database, limit: int = 300, paper_ids: list[str] | None = None) -> dict[str, int]:
    if paper_ids is not None:
        rows = db.conn.execute(f"SELECT id, title FROM papers WHERE id IN ({','.join('?' * len(paper_ids))})",
                               paper_ids).fetchall()
    else:
        rows = db.conn.execute(
            """SELECT p.id, p.title FROM papers p LEFT JOIN paper_reviews r ON r.paper_id = p.id
               WHERE r.paper_id IS NULL ORDER BY COALESCE(p.citation_count, 0) DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    http = RateLimitedClient(1.2, max_retries=2 if paper_ids is not None else 4, max_backoff=10.0)
    counts: dict[str, int] = {}
    try:
        for row in rows:
            try:
                notes = http.get(API, params={"term": row["title"], "type": "terms", "content": "title",
                                              "group": "all", "source": "forum", "limit": 5}).json().get("notes", [])
            except httpx.HTTPError as e:
                log.warning("openreview failed for %s: %s", row["id"], e)
                continue
            status, forum, venue = "not_found", None, None
            for n in notes:
                title = ((n.get("content") or {}).get("title") or {}).get("value", "")
                v = ((n.get("content") or {}).get("venue") or {}).get("value", "")
                if similarity(title, row["title"]) >= 0.9 and v and not v.lower().startswith("corr"):
                    status, forum, venue = classify_venue(v), n.get("forum"), v
                    if status == "accepted":
                        break
            db.conn.execute(
                "INSERT OR REPLACE INTO paper_reviews (paper_id, or_forum, or_venue, status) VALUES (?, ?, ?, ?)",
                (row["id"], forum, venue, status),
            )
            counts[status] = counts.get(status, 0) + 1
            db.commit()
    finally:
        http.close()
    return counts
