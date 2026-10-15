"""What other papers say about a paper: Semantic Scholar citation contexts, with critical ones flagged.

stance: critical (reports failure / limitation / contradiction), comparative (outperformed / compared),
neutral. Rule-based (evidence C): read the context before quoting it as criticism.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from kb.config import settings
from kb.store import Database

log = logging.getLogger(__name__)

CRITICAL = re.compile(
    r"\b(fails? to|failed to|cannot|can ?not|could not|does not (?:generali[sz]e|scale|hold|work)|struggles?|"
    r"suffers? from|limitation|drawback|shortcoming|degrad\w+|artifacts?|unstable|instabilit\w+|collapse|"
    r"contrary to|in contrast to (?:the claims?|their)|unlike .{0,40}(?:claim|report)|not reproduc\w+|"
    r"could not reproduce|inconsistent with|overestimat\w+|cherry-?pick\w*|only (?:works|evaluated)|"
    r"at the cost of|sacrific\w+|noticeable (?:quality )?(?:loss|drop)|quality (?:loss|drop))\b", re.IGNORECASE)
SUBJECT_CUE = re.compile(
    r"\b(?:fails?|failed|struggles?|suffers?|degrades?|collapses?|is (?:limited|unstable|slow|prone)|"
    r"(?:can ?not|cannot|could not|does not|did not) (?:\w+ ){0,2}(?:handle|generali[sz]e|scale|preserve|maintain|capture|"
    r"produce|work|converge)|produces? (?:noticeable )?(?:artifacts?|blur\w*|flicker\w*)|introduces? (?:\w+ )?(?:artifacts?|error)|"
    r"leads? to (?:\w+ )?(?:degradation|artifacts?|quality (?:loss|drop))|sacrific\w+|at the cost of)\b", re.IGNORECASE)
COMPARATIVE = re.compile(r"\b(outperform\w*|surpass\w*|better than|superior to|compared (?:to|with)|"
                         r"improv\w+ (?:over|upon)|beat\w*|faster than|slower than)\b", re.IGNORECASE)


def method_name(title: str) -> str | None:
    """'SageAttention: Accurate 8-Bit ...' -> 'SageAttention'; None when the title has no short method name."""
    head = title.split(":")[0].strip()
    return head if 2 <= len(head) <= 30 and len(head.split()) <= 3 else None


def stance(context: str, name: str | None = None) -> str:
    """critical = a failure/limitation cue within 80 chars of the method's name; possible_critical = cue without
    the name nearby (often about another method in the same sentence); comparative; neutral."""
    cues = list(CRITICAL.finditer(context))
    if cues:
        # Directional: "<Method> [12] ... fails / suffers from / struggles" — the method is the subject of the cue.
        if name and re.search(re.escape(name) + r"(?:\s*\[[\d,\s–-]+\])?[^.;]{0,50}?" + SUBJECT_CUE.pattern, context, re.IGNORECASE):
            return "critical"
        return "possible_critical"
    if COMPARATIVE.search(context):
        return "comparative"
    return "neutral"


def restance(db: Database) -> dict[str, int]:
    """Re-score stored contexts (after rule changes) using each cited paper's method name."""
    titles = {r["id"]: r["title"] for r in db.conn.execute(
        "SELECT id, title FROM papers WHERE id IN (SELECT DISTINCT paper_id FROM citation_contexts)")}
    rows = db.conn.execute("SELECT rowid, paper_id, context FROM citation_contexts").fetchall()
    for r in rows:
        db.conn.execute("UPDATE citation_contexts SET stance = ? WHERE rowid = ?",
                        (stance(r["context"], method_name(titles.get(r["paper_id"], ""))), r["rowid"]))
    db.commit()
    return {r[0]: r[1] for r in db.conn.execute("SELECT stance, COUNT(*) FROM citation_contexts GROUP BY 1")}


def collect(db: Database, limit: int = 200, per_paper: int = 200, grades: str = "AB",
            paper_ids: list[str] | None = None, interactive: bool = False) -> dict[str, int]:
    from kb.collect.http import RateLimitedClient

    headers = {"x-api-key": settings.s2_api_key} if settings.s2_api_key else {}
    http = RateLimitedClient(settings.s2_min_interval, headers=headers, max_retries=2 if interactive else 6,
                             max_backoff=8.0 if interactive else 60.0)
    if paper_ids is not None:
        rows = db.conn.execute(
            f"""SELECT id, arxiv_id, s2_id, doi, title FROM papers
                WHERE id IN ({",".join("?" * len(paper_ids))})""", paper_ids).fetchall()
    else:
        rows = db.conn.execute(
            f"""SELECT p.id, p.arxiv_id, p.s2_id, p.doi, p.title FROM papers p LEFT JOIN critique c ON c.source_id = p.id
                WHERE (p.arxiv_id IS NOT NULL OR p.s2_id IS NOT NULL OR p.doi IS NOT NULL)
                  AND COALESCE(c.grade, 'C') IN ({",".join("?" * len(grades))})
                  AND p.id NOT IN (SELECT DISTINCT paper_id FROM citation_contexts)
                  AND p.id NOT IN (SELECT key FROM meta WHERE value = 'citctx-done')
                ORDER BY COALESCE(p.citation_count, 0) DESC LIMIT ?""",
            (*grades, limit)).fetchall()
    counts = {"papers": 0, "contexts": 0, "critical": 0}
    try:
        for r in rows:
            if r["arxiv_id"]:
                ref = f"arXiv:{re.sub(r'v\\d+$', '', r['arxiv_id'])}"
            elif r["s2_id"]:
                ref = r["s2_id"]
            elif r["doi"]:
                ref = f"DOI:{r['doi']}"
            else:
                continue
            offset = 0
            while offset < per_paper:
                try:
                    data = http.get(f"{settings.s2_api}/paper/{ref}/citations",
                                    params={"fields": "title,year,externalIds,intents,contexts,isInfluential",
                                            "limit": 100, "offset": offset}).json()
                except Exception as e:
                    log.warning("citations %s: %s", ref, e)
                    counts["failed"] = counts.get("failed", 0) + 1
                    break
                for c in data.get("data", []):
                    cp = c.get("citingPaper") or {}
                    if not cp.get("paperId"):
                        continue
                    for ctx in c.get("contexts") or []:
                        st = stance(ctx, method_name(r["title"]))
                        db.conn.execute(
                            """INSERT OR IGNORE INTO citation_contexts (paper_id, citing_s2_id, citing_title, citing_year,
                                   citing_arxiv, influential, intents, context, stance) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (r["id"], cp["paperId"], cp.get("title"), cp.get("year"),
                             (cp.get("externalIds") or {}).get("ArXiv"), int(bool(c.get("isInfluential"))),
                             json.dumps(c.get("intents") or []), ctx[:1500], st))
                        counts["contexts"] += 1
                        counts["critical"] += st == "critical"
                if not data.get("next"):
                    break
                offset = data["next"]
            if not counts.get("failed"):
                db.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, 'citctx-done')", (r["id"],))
            db.commit()
            counts["papers"] += 1
            if counts["papers"] % 20 == 0:
                log.info("citation contexts: %s", counts)
    finally:
        http.close()
    return counts


def what_others_say(db: Database, paper_id: str, stance_filter: str | None = None, limit: int = 20) -> dict[str, Any]:
    paper = db.get_paper(paper_id)
    if paper is None:
        return {"error": f"unknown paper {paper_id}: find it with find_papers first"}
    pid = paper["id"]
    fetched: dict[str, int] | None = None
    have = db.conn.execute("SELECT 1 FROM citation_contexts WHERE paper_id = ? LIMIT 1", (pid,)).fetchone()
    done = db.conn.execute("SELECT 1 FROM meta WHERE key = ? AND value = 'citctx-done'", (pid,)).fetchone()
    if not have and not done:  # first time: fetch from Semantic Scholar now
        fetched = collect(db, per_paper=300, paper_ids=[pid], interactive=True)
    sql = "SELECT * FROM citation_contexts WHERE paper_id = ?"
    params: list[Any] = [pid]
    if stance_filter:
        sql += " AND stance = ?"
        params.append(stance_filter)
    sql += " ORDER BY CASE stance WHEN 'critical' THEN 0 WHEN 'possible_critical' THEN 1 WHEN 'comparative' THEN 2 ELSE 3 END, influential DESC, citing_year DESC LIMIT ?"
    params.append(limit)
    rows = [dict(r) for r in db.conn.execute(sql, params)]
    totals = {r[0]: r[1] for r in db.conn.execute(
        "SELECT stance, COUNT(*) FROM citation_contexts WHERE paper_id = ? GROUP BY stance", (pid,))}
    for r in rows:
        r["intents"] = json.loads(r["intents"] or "[]")
    out: dict[str, Any] = {"paper_id": pid, "title": paper["title"], "totals": totals, "contexts": rows}
    if fetched and fetched.get("failed"):
        out["warning"] = "Semantic Scholar did not answer (rate limit without a key) — try again in a minute"
    return {**out,
            "note": "Stance is keyword-based (evidence C): read each context; 'critical' often means a limitation "
                    "noted by others, not a refutation."}
