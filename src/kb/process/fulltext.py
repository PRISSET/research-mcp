"""Fetch full texts (arXiv HTML, arXiv PDF, open-access PDF), parse, chunk and store them.

`fetch_papers` is the on-demand path used by the MCP server; `collect` is the batch path for the CLI.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator

from kb.collect.http import RateLimitedClient
from kb.process.parse import Block, parse_arxiv_html, parse_pdf
from kb.store import Database

log = logging.getLogger(__name__)

TARGET_WORDS = 280     # ~300-word chunks retrieve far more precisely than whole sections
MAX_WORDS = 420


def queue(db: Database, limit: int | None = None, grades: str = "ABC", retry_failed: bool = False) -> list[str]:
    done = "SELECT paper_id FROM paper_fulltext" + (" WHERE status IN ('ok', 'no_source')" if retry_failed else "")
    sql = f"""
        SELECT p.id FROM papers p LEFT JOIN critique c ON c.source_id = p.id
        WHERE p.arxiv_id IS NOT NULL AND p.id NOT IN ({done})
          AND COALESCE(c.grade, 'C') IN ({",".join("?" * len(grades))})
        ORDER BY CASE COALESCE(c.grade, 'C') WHEN 'A' THEN 0 WHEN 'B' THEN 1 WHEN 'C' THEN 2 ELSE 3 END,
                 COALESCE(p.citation_count, 0) DESC, p.year DESC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [r[0] for r in db.conn.execute(sql, tuple(grades))]


def chunk_blocks(blocks: list[Block]) -> Iterator[tuple[str, str, str, int | None]]:
    """Pack consecutive text paragraphs of one section into ~TARGET_WORDS chunks; tables/figures stay alone.

    Each chunk is prefixed with its section path so both BM25 and the embedding see the context."""
    buf: list[str] = []
    buf_words = 0
    cur_section: str | None = None
    cur_page: int | None = None

    def flush() -> Iterator[tuple[str, str, str, int | None]]:
        nonlocal buf, buf_words
        if buf:
            yield cur_section or "Body", "text", " ".join(buf), cur_page
        buf, buf_words = [], 0

    for b in blocks:
        words = len(b.text.split())
        if b.kind != "text":
            yield from flush()
            text = b.text if words <= MAX_WORDS * 2 else " ".join(b.text.split()[: MAX_WORDS * 2])
            yield b.section, b.kind, text, b.page
            continue
        if b.section != cur_section or buf_words + words > MAX_WORDS:
            yield from flush()
            cur_section, cur_page = b.section, b.page
        if words > MAX_WORDS:  # very long paragraph: split on sentences
            sents = re.split(r"(?<=[.!?])\s+", b.text)
            part: list[str] = []
            n = 0
            for s in sents:
                part.append(s)
                n += len(s.split())
                if n >= TARGET_WORDS:
                    yield b.section, "text", " ".join(part), b.page
                    part, n = [], 0
            if part:
                buf, buf_words = part, n
            continue
        buf.append(b.text)
        buf_words += words
        if buf_words >= TARGET_WORDS:
            yield from flush()
    yield from flush()


def _store(db: Database, paper_id: str, source: str, blocks: list[Block]) -> int:
    db.conn.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
    n = 0
    for i, (section, kind, text, page) in enumerate(chunk_blocks(blocks)):
        db.conn.execute(
            "INSERT INTO chunks (paper_id, ord, section, kind, page, text, n_words) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (paper_id, i, section[:300], kind, page, text, len(text.split())),
        )
        n += 1
    db.conn.execute(
        """INSERT INTO paper_fulltext (paper_id, status, source, n_chars, n_chunks, used_at)
           VALUES (?, 'ok', ?, ?, ?, datetime('now'))
           ON CONFLICT(paper_id) DO UPDATE SET status='ok', source=excluded.source, n_chars=excluded.n_chars,
               n_chunks=excluded.n_chunks, error=NULL, fetched_at=datetime('now'), used_at=datetime('now')""",
        (paper_id, source, sum(len(b.text) for b in blocks), n),
    )
    return n


def _fail(db: Database, paper_id: str, status: str, error: str) -> None:
    db.conn.execute(
        """INSERT INTO paper_fulltext (paper_id, status, error) VALUES (?, ?, ?)
           ON CONFLICT(paper_id) DO UPDATE SET status=excluded.status, error=excluded.error, fetched_at=datetime('now')""",
        (paper_id, status, error[:500]),
    )


def _from_pdf(db: Database, http: RateLimitedClient, paper_id: str, url: str, source: str) -> str:
    r = http.get(url)
    if not r.content.startswith(b"%PDF"):
        _fail(db, paper_id, "no_source", f"not a PDF: {url}")
        return "none"
    blocks = parse_pdf(r.content)
    if sum(len(b.text) for b in blocks) < 3000:
        _fail(db, paper_id, "failed", "pdf text too short (scanned?)")
        return "none"
    _store(db, paper_id, source, blocks)
    return "pdf"


def fetch_one(db: Database, http: RateLimitedClient, paper_id: str, arxiv_id: str | None, pdf_url: str | None = None) -> str:
    if arxiv_id:
        base = re.sub(r"v\d+$", "", arxiv_id)
        try:
            r = http.get(f"https://arxiv.org/html/{base}")
            html = r.text
            if "ltx_document" in html or "ltx_page_main" in html:
                _, blocks = parse_arxiv_html(html)
                if sum(len(b.text) for b in blocks) > 3000:
                    _store(db, paper_id, "arxiv-html", blocks)
                    return "html"
        except Exception as e:  # 404 = no HTML rendering (older or conversion-failed papers)
            log.debug("html %s: %s", base, e)
        try:
            return _from_pdf(db, http, paper_id, f"https://arxiv.org/pdf/{base}", "arxiv-pdf")
        except Exception as e:
            if not pdf_url or "arxiv.org" in pdf_url:
                _fail(db, paper_id, "failed", f"{type(e).__name__}: {e}")
                return "none"
    if not pdf_url:
        _fail(db, paper_id, "no_source", "no arXiv id and no open-access PDF")
        return "none"
    try:
        return _from_pdf(db, http, paper_id, pdf_url, "oa-pdf")
    except Exception as e:
        _fail(db, paper_id, "failed", f"{type(e).__name__}: {e}")
        return "none"


def fetch_papers(db: Database, paper_ids: list[str], embed: bool = True, refresh: bool = False) -> dict[str, object]:
    """On demand: full text -> chunks -> numbers -> vectors for just these papers."""
    from kb.process.claims import extract_all

    http = RateLimitedClient(1.0, max_retries=2, max_backoff=10, use_cache=False)  # text lands in the base
    report: dict[str, object] = {}
    ok: list[str] = []
    try:
        for ref in paper_ids:
            p = db.get_paper(ref)
            if p is None:
                report[ref] = "unknown id: find it with find_papers first"
                continue
            done = db.conn.execute("SELECT status, n_chunks FROM paper_fulltext WHERE paper_id = ?", (p["id"],)).fetchone()
            if done and done["status"] == "ok" and not refresh:
                report[p["id"]] = f"already indexed ({done['n_chunks']} chunks)"
                db.touch([p["id"]])
                ok.append(p["id"])
                continue
            how = fetch_one(db, http, p["id"], p.get("arxiv_id"), p.get("pdf_url"))
            db.commit()
            if how == "none":
                err = db.conn.execute("SELECT error FROM paper_fulltext WHERE paper_id = ?", (p["id"],)).fetchone()
                report[p["id"]] = f"no full text: {err[0] if err else 'unknown'} — work from the abstract"
            else:
                n = db.conn.execute("SELECT n_chunks FROM paper_fulltext WHERE paper_id = ?", (p["id"],)).fetchone()[0]
                report[p["id"]] = f"indexed from {how} ({n} chunks)"
                ok.append(p["id"])
    finally:
        http.close()
    extract_all(db, paper_ids=ok)
    vectors = "skipped"
    if embed and ok:
        from kb.process.embed import embed_now

        vectors = embed_now(db, ok)
    return {"papers": report, "vectors": vectors}


def collect(db: Database, limit: int | None = None, grades: str = "ABC", interval: float = 1.0,
            retry_failed: bool = False, progress: bool = True) -> dict[str, int]:
    ids = queue(db, limit, grades, retry_failed)
    http = RateLimitedClient(interval, max_retries=4, max_backoff=120, use_cache=False)
    counts = {"html": 0, "pdf": 0, "none": 0}
    t0 = time.monotonic()
    try:
        for i, pid in enumerate(ids, 1):
            arxiv_id = db.conn.execute("SELECT arxiv_id FROM papers WHERE id = ?", (pid,)).fetchone()[0]
            counts[fetch_one(db, http, pid, arxiv_id)] += 1
            db.commit()
            if progress and (i % 25 == 0 or i == len(ids)):
                rate = i / (time.monotonic() - t0)
                log.info("fulltext %d/%d %s (%.2f papers/s)", i, len(ids), counts, rate)
    finally:
        http.close()
    return counts


def stats(db: Database) -> dict[str, object]:
    one = lambda sql: db.conn.execute(sql).fetchone()[0]
    return {
        "papers_with_arxiv": one("SELECT COUNT(*) FROM papers WHERE arxiv_id IS NOT NULL"),
        "fulltext": {r[0]: r[1] for r in db.conn.execute("SELECT COALESCE(source, status), COUNT(*) FROM paper_fulltext GROUP BY 1")},
        "chunks": one("SELECT COUNT(*) FROM chunks"),
        "embedded": one("SELECT COUNT(*) FROM chunks WHERE emb IS NOT NULL"),
        "claims": one("SELECT COUNT(*) FROM paper_claims"),
    }
