"""Keep the base a cache of what the agent actually read, not a growing library.

- A full text (passages, vectors, extracted numbers) that the notebook does not cite and that no search or read
  has returned for `days` is dropped; its metadata row stays, so a later fetch re-reads it in a second.
- A metadata-only record (a search result never read) older than `days` and not cited is dropped.
- Tables left by earlier versions of the project are dropped.

The server runs this at most once a day in the background (KB_KEEP_DAYS, default 60; 0 turns it off).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from kb.store import Database

log = logging.getLogger(__name__)

KEEP_DAYS = int(os.environ.get("KB_KEEP_DAYS", "60"))
# Tables of the old MiniMax-H3 knowledge base and the bulk collector, unused by the code.
LEGACY_TABLES = ("h3_artifacts", "h3_artifacts_fts", "h3_facts", "cloud_prices", "provider_offers", "catalog_items",
                 "catalog_fts", "measurements", "consultations", "user_profile", "feedback", "paper_topics",
                 "unresolved_seeds", "embed_queue")


def _ids(rows: Any) -> list[str]:
    return [r[0] for r in rows]


def _delete(db: Database, table: str, column: str, ids: list[str]) -> None:
    for i in range(0, len(ids), 500):
        part = ids[i: i + 500]
        db.conn.execute(f"DELETE FROM {table} WHERE {column} IN ({','.join('?' * len(part))})", part)


def prune(db: Database, days: int = KEEP_DAYS, keep: set[str] | None = None, vacuum: bool = True) -> dict[str, Any]:
    if keep is None:
        from kb import notebook

        keep = notebook.cited_ids()
    cutoff = f"-{int(days)} days"
    stale_text = [i for i in _ids(db.conn.execute(
        "SELECT paper_id FROM paper_fulltext WHERE COALESCE(used_at, fetched_at) < datetime('now', ?)", (cutoff,)))
        if i not in keep]
    _delete(db, "paper_claims", "paper_id", stale_text)
    _delete(db, "chunks", "paper_id", stale_text)
    _delete(db, "paper_fulltext", "paper_id", stale_text)
    stale_meta = [i for i in _ids(db.conn.execute(
        """SELECT id FROM papers WHERE added_at < datetime('now', ?)
           AND id NOT IN (SELECT paper_id FROM paper_fulltext)""", (cutoff,))) if i not in keep]
    for table, column in (("critique", "source_id"), ("papers_fts", "paper_id"), ("citation_contexts", "paper_id"),
                          ("paper_reviews", "paper_id"), ("papers", "id")):
        _delete(db, table, column, stale_meta)
    existing = set(_ids(db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")))
    dropped = [t for t in LEGACY_TABLES if t in existing]
    for t in dropped:
        db.conn.execute(f"DROP TABLE IF EXISTS {t}")
    db.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('last_prune', datetime('now'))")
    db.commit()
    if vacuum and (stale_text or stale_meta or dropped):
        db.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
        db.commit()
        db.conn.execute("VACUUM")
    from kb.process.embed import _matrix

    _matrix.cache_clear()
    return {"full_texts_dropped": len(stale_text), "records_dropped": len(stale_meta), "tables_dropped": dropped,
            "kept_cited": len(keep)}


def prune_daily() -> None:
    """Server start: prune at most once a day (in a background thread, errors only logged)."""
    if KEEP_DAYS <= 0:
        return
    try:
        db = Database()
        try:
            row = db.conn.execute("SELECT 1 FROM meta WHERE key = 'last_prune' "
                                  "AND value > datetime('now', '-1 day')").fetchone()
            if not row:
                log.info("prune: %s", prune(db))
        finally:
            db.close()
    except Exception as e:
        log.warning("prune skipped: %s", e)
