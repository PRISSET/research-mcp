"""SQLite access: schema, upserts and full-text search."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from kb.config import settings
from kb.schemas import Paper

SCHEMA = Path(__file__).with_name("schema.sql")
# Columns added after the first release: (table, column, type). Old databases get them on open.
MIGRATIONS = [("papers", "openalex_id", "TEXT"), ("papers", "kind", "TEXT NOT NULL DEFAULT 'paper'"),
              ("papers", "extra", "TEXT NOT NULL DEFAULT '{}'")]
# Fields a later source may fill in but never blank out.
MERGE_FIELDS = ("abstract", "year", "published", "updated", "arxiv_id", "doi", "s2_id", "openalex_id", "venue",
                "url", "pdf_url", "comment", "influential_count", "tldr")


class Database:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.db_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Batch jobs and the MCP server may write concurrently: wait for the lock instead of failing.
        self.conn = sqlite3.connect(self.path, timeout=120)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA.read_text())
        self._migrate()

    def _migrate(self) -> None:
        for table, column, kind in MIGRATIONS:
            cols = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                try:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
                except sqlite3.OperationalError as e:  # another connection migrated first
                    if "duplicate column" not in str(e):
                        raise
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_openalex ON papers(openalex_id)")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.conn.commit()
        self.close()

    # ---- writes -------------------------------------------------------------

    def find_existing(self, p: Paper) -> str | None:
        """Id of a stored record for the same work: same arXiv id, DOI, OpenAlex or S2 id, or same title."""
        for col, val in (("arxiv_id", p.arxiv_id), ("doi", p.doi and p.doi.lower()), ("openalex_id", p.openalex_id),
                         ("s2_id", p.s2_id)):
            if val:
                expr = "lower(doi)" if col == "doi" else col
                row = self.conn.execute(f"SELECT id FROM papers WHERE {expr} = ?", (val,)).fetchone()
                if row:
                    return row[0]
        row = self.conn.execute(
            "SELECT id FROM papers WHERE kind = 'paper' AND lower(title) = ? "
            "AND (year IS NULL OR ? IS NULL OR abs(year - ?) <= 1)",
            (p.title.lower(), p.year, p.year)).fetchone()
        return row[0] if row else None

    def upsert_paper(self, p: Paper) -> tuple[str, bool]:
        """Insert or merge a paper. Returns (stored id, was_new). Known fields are never blanked by a poorer source."""
        existing = self.find_existing(p)
        row = {
            "title": p.title,
            "abstract": p.abstract,
            "authors": json.dumps(p.authors, ensure_ascii=False),
            "year": p.year,
            "published": p.published.isoformat() if p.published else None,
            "updated": p.updated.isoformat() if p.updated else None,
            "arxiv_id": p.arxiv_id,
            "doi": p.doi,
            "s2_id": p.s2_id,
            "openalex_id": p.openalex_id,
            "categories": json.dumps(p.categories),
            "venue": p.venue,
            "url": p.url,
            "pdf_url": p.pdf_url,
            "comment": p.comment,
            "citation_count": p.citation_count,
            "influential_count": p.influential_count,
            "tldr": p.tldr,
            "source": p.source,
        }
        if existing:
            sets = [f"{f} = COALESCE(NULLIF(:{f}, ''), {f})" for f in MERGE_FIELDS]
            sets.append("authors = CASE WHEN :authors = '[]' THEN authors ELSE :authors END")
            sets.append("categories = CASE WHEN :categories = '[]' THEN categories ELSE :categories END")
            sets.append("citation_count = MAX(COALESCE(citation_count, 0), COALESCE(:citation_count, 0))")
            # arXiv ids are unique: do not steal one already held by another record.
            if p.arxiv_id and self.conn.execute("SELECT 1 FROM papers WHERE arxiv_id = ? AND id != ?",
                                                (p.arxiv_id, existing)).fetchone():
                sets = [s for s in sets if not s.startswith("arxiv_id")]
            self.conn.execute(f"UPDATE papers SET {', '.join(sets)} WHERE id = :id", {**row, "id": existing})
            pid, new = existing, False
        else:
            row["id"] = p.id
            self.conn.execute(f"INSERT INTO papers ({', '.join(row)}) VALUES ({', '.join(':' + k for k in row)})", row)
            pid, new = p.id, True
        self._reindex(pid)
        return pid, new

    def update_s2(self, paper_id: str, data: dict[str, Any]) -> None:
        self.conn.execute(
            """UPDATE papers SET s2_id=COALESCE(:s2_id, s2_id),
                   citation_count=MAX(COALESCE(citation_count, 0), COALESCE(:citation_count, 0)),
                   influential_count=:influential_count, tldr=COALESCE(:tldr, tldr),
                   venue=COALESCE(NULLIF(:venue, ''), venue), doi=COALESCE(doi, :doi),
                   publication_types=:publication_types, s2_checked_at=datetime('now')
               WHERE id=:id""",
            {"id": paper_id, **data},
        )
        self._reindex(paper_id)

    def mark_s2_checked(self, paper_id: str) -> None:
        self.conn.execute("UPDATE papers SET s2_checked_at=datetime('now') WHERE id=?", (paper_id,))

    def log_run(self, collector: str, query: str, fetched: int, new: int) -> None:
        self.conn.execute(
            "INSERT INTO collection_runs (collector, query, fetched, new) VALUES (?, ?, ?, ?)",
            (collector, query, fetched, new),
        )

    def commit(self) -> None:
        self.conn.commit()

    def _reindex(self, paper_id: str) -> None:
        self.conn.execute("DELETE FROM papers_fts WHERE paper_id = ?", (paper_id,))
        self.conn.execute(
            """INSERT INTO papers_fts (paper_id, title, abstract, tldr)
               SELECT id, title, abstract, COALESCE(tldr, '') FROM papers WHERE id = ?""",
            (paper_id,),
        )

    # ---- reads --------------------------------------------------------------

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        """By stored id, bare arXiv id, DOI ("10.…" or "doi:10.…") or OpenAlex id."""
        key = paper_id.strip()
        bare = key.split(":", 1)[1] if key.lower().startswith(("arxiv:", "doi:", "openalex:")) else key
        row = self.conn.execute(
            "SELECT * FROM papers WHERE id = ? OR arxiv_id = ? OR lower(doi) = lower(?) OR openalex_id = ?",
            (key, bare, bare, bare)).fetchone()
        return _row_to_dict(row) if row else None

    def search(self, query: str, *, year_from: int | None = None, paper_ids: set[str] | None = None,
               limit: int = 10, match_all: bool = False) -> list[dict[str, Any]]:
        sql = """
            SELECT p.*, bm25(papers_fts, 0, 10.0, 1.0, 2.0) AS rank
            FROM papers_fts JOIN papers p ON p.id = papers_fts.paper_id
            WHERE papers_fts MATCH :q
        """
        params: dict[str, Any] = {"q": to_fts_query(query, match_all), "limit": limit}
        if year_from:
            sql += " AND p.year >= :year_from"
            params["year_from"] = year_from
        if paper_ids is not None:
            sql += f" AND p.id IN ({','.join(f':p{i}' for i in range(len(paper_ids)))})"
            params.update({f"p{i}": pid for i, pid in enumerate(paper_ids)})
        # BM25 is negative (lower is better); nudge well-cited papers up.
        sql += " ORDER BY rank - 0.6 * ln(1 + COALESCE(p.citation_count, 0)) / 10.0 LIMIT :limit"
        try:
            return [_row_to_dict(r) for r in self.conn.execute(sql, params)]
        except sqlite3.OperationalError:  # malformed FTS query
            return []

    def stats(self) -> dict[str, Any]:
        one = lambda sql: self.conn.execute(sql).fetchone()[0]
        return {
            "papers": one("SELECT COUNT(*) FROM papers"),
            "with_fulltext": one("SELECT COUNT(*) FROM paper_fulltext WHERE status = 'ok'"),
            "chunks": one("SELECT COUNT(*) FROM chunks"),
            "embedded_chunks": one("SELECT COUNT(*) FROM chunks WHERE emb IS NOT NULL"),
            "extracted_numbers": one("SELECT COUNT(*) FROM paper_claims"),
            "years": {
                r[0]: r[1]
                for r in self.conn.execute(
                    "SELECT year, COUNT(*) FROM papers WHERE year IS NOT NULL GROUP BY year ORDER BY year DESC LIMIT 8"
                )
            },
        }

    def papers_missing_s2(self, limit: int | None = None, ids: list[str] | None = None) -> list[sqlite3.Row]:
        sql = "SELECT id, arxiv_id, doi FROM papers WHERE s2_checked_at IS NULL AND (arxiv_id IS NOT NULL OR doi IS NOT NULL)"
        params: list[Any] = []
        if ids is not None:
            sql += f" AND id IN ({','.join('?' * len(ids))})"
            params += ids
        sql += " ORDER BY year DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return list(self.conn.execute(sql, params))

    def iter_papers(self) -> Iterable[sqlite3.Row]:
        return self.conn.execute("SELECT id, title, abstract FROM papers")


def to_fts_query(text: str, match_all: bool = False) -> str:
    """Turn free text into a safe FTS5 query: quoted terms joined with OR (or AND), phrases kept."""
    if any(op in text for op in ('"', " AND ", " OR ", " NOT ")):
        return text
    terms = [t for t in "".join(c if c.isalnum() or c in "-_" else " " for c in text).split() if len(t) > 1]
    return (" AND " if match_all else " OR ").join(f'"{t}"' for t in terms) or '""'


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for key in ("authors", "categories", "publication_types", "extra"):
        if isinstance(d.get(key), str):
            d[key] = json.loads(d[key])
    return d
